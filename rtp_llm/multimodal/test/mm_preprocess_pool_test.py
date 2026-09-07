"""CPU-only regression tests for pool recovery under concurrent video requests."""

import concurrent.futures
import multiprocessing
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.multimodal import mm_process_engine as engine


class FakePool:
    def __init__(self):
        self.tasks = []
        self.terminated = False
        self.closed = False
        self.submit_error = None

    def apply_async(self, func, args, callback, error_callback):
        if self.submit_error:
            raise self.submit_error
        if self.terminated or self.closed:
            raise AssertionError("submitted to a retired pool")
        self.tasks.append((callback, error_callback))

    def succeed(self, index=0, value="ok"):
        self.tasks[index][0]((value, 1.0, []))

    def fail(self, error, index=0):
        self.tasks[index][1](error)

    def terminate(self):
        self.terminated = True

    def close(self):
        self.closed = True

    def join(self):
        pass


def work_item(timeout_ms=120000, value="video"):
    return SimpleNamespace(
        embedding_result=None,
        preprocess_result=None,
        future=None,
        preprocess_pool_generation=None,
        mm_inputs=[value],
        mm_timeout_ms=timeout_ms,
    )


def spawn_preprocess(mm_inputs, vit_config, **kwargs):
    if mm_inputs == ["crash"]:
        os._exit(1)
    if mm_inputs == ["timeout"]:
        raise TimeoutError("video download timed out")
    return mm_inputs


class PreprocessPoolTest(unittest.TestCase):
    def setUp(self):
        self.pools = []
        self.context = mock.Mock()
        self.context.Pool.side_effect = self.make_pool
        self.executor = engine.MultiprocessPreprocessExecutor(
            self.context,
            SimpleNamespace(mm_preprocess_max_workers=2),
            {},
            spawn_preprocess,
        )
        self.addCleanup(self.executor.shutdown)
        self.report = mock.patch.object(engine.kmonitor, "report").start()
        self.addCleanup(mock.patch.stopall)

    def make_pool(self, **kwargs):
        pool = FakePool()
        self.pools.append(pool)
        return pool

    def submit(self, **kwargs):
        item = work_item(**kwargs)
        self.executor.submit(item)
        return item

    def test_success_and_metrics(self):
        item = self.submit()
        self.pools[0].succeed(value="decoded video")
        self.executor._consecutive_timeouts = 1
        self.executor.get_result(item)
        self.assertEqual(item.preprocess_result, "decoded video")
        self.assertEqual(self.executor._consecutive_timeouts, 0)
        self.report.assert_called_once()

    def test_cache_hit_needs_no_pool(self):
        item = work_item()
        item.embedding_result = "cached"
        self.executor.submit(item)
        self.executor.get_result(item)
        self.assertEqual(self.pools[0].tasks, [])

    def test_submit_waits_for_rebuild_publication(self):
        creating = threading.Event()
        release = threading.Event()
        submitting = threading.Event()

        def slow_create(**kwargs):
            creating.set()
            if not release.wait(5):
                raise AssertionError("test failed to release pool creation")
            return self.make_pool(**kwargs)

        self.context.Pool.side_effect = slow_create
        with concurrent.futures.ThreadPoolExecutor(2) as callers:
            rebuild = callers.submit(self.executor._rebuild_pool)
            try:
                self.assertTrue(creating.wait(5))
                self.assertIsNone(self.executor.pool)

                def submit():
                    submitting.set()
                    return self.submit()

                submission = callers.submit(submit)
                self.assertTrue(submitting.wait(5))
                with self.assertRaises(concurrent.futures.TimeoutError):
                    submission.result(timeout=0.05)
            finally:
                release.set()
            rebuild.result(timeout=5)
            item = submission.result(timeout=5)
        self.assertEqual(len(self.pools[0].tasks), 0)
        self.pools[1].succeed()
        self.executor.get_result(item)

    def test_one_rebuild_invalidates_400_old_requests(self):
        items = [self.submit(timeout_ms=0) for _ in range(400)]
        old = self.pools[0]

        def get_error(item):
            try:
                self.executor.get_result(item)
            except (TimeoutError, RuntimeError) as error:
                return str(error)
            self.fail("abandoned task unexpectedly succeeded")

        with mock.patch.object(engine.logging, "error"):
            with concurrent.futures.ThreadPoolExecutor(40) as callers:
                errors = list(callers.map(get_error, items))
        self.assertEqual(len(errors), 400)
        self.assertEqual(len(self.pools), 2)
        self.assertTrue(old.terminated)
        self.assertEqual(self.executor._consecutive_timeouts, 0)
        self.assertTrue(all(item.future.done() for item in items))
        self.assertEqual(len(items[0].preprocess_pool_generation._pending), 0)
        fresh = self.submit()
        self.pools[1].succeed()
        self.executor.get_result(fresh)
        self.assertEqual(fresh.preprocess_result, "ok")

    def test_rebuild_wakes_waiter_without_request_timeout(self):
        item = self.submit()  # The production 120-second request deadline.
        with concurrent.futures.ThreadPoolExecutor(1) as callers:
            waiter = callers.submit(self.executor.get_result, item)
            self.executor._rebuild_pool()
            with self.assertRaisesRegex(RuntimeError, "generation 0 was replaced"):
                waiter.result(timeout=5)

    def test_stale_wait_timeout_cannot_rebuild_new_pool(self):
        item = self.submit()
        self.executor._rebuild_pool()
        self.executor._consecutive_timeouts = 1
        # The old wait deadline expires just as another thread replaces the pool.
        with mock.patch.object(
            engine.concurrent.futures, "wait", return_value=([], [])
        ):
            with self.assertRaises(TimeoutError):
                self.executor.get_result(item)
        self.assertEqual(len(self.pools), 2)
        self.assertEqual(self.executor._consecutive_timeouts, 1)

    def test_old_success_cannot_clear_new_pool_timeout_counter(self):
        item = self.submit()
        self.pools[0].succeed()
        self.executor._rebuild_pool()
        self.executor._consecutive_timeouts = 1
        self.executor.get_result(item)
        self.assertEqual(item.preprocess_result, "ok")
        self.assertEqual(self.executor._consecutive_timeouts, 1)

    def test_old_broken_pipe_cannot_rebuild_new_pool(self):
        item = self.submit()
        self.pools[0].fail(BrokenPipeError("old pipe"))
        self.executor._rebuild_pool()
        with self.assertRaisesRegex(BrokenPipeError, "old pipe"):
            self.executor.get_result(item)
        self.assertEqual(len(self.pools), 2)

    def test_late_callback_cannot_resurrect_invalidated_result(self):
        item = self.submit()
        old = self.pools[0]
        self.executor._rebuild_pool()
        old.succeed()
        old.fail(OSError("late failure"))
        with self.assertRaisesRegex(RuntimeError, "was replaced"):
            self.executor.get_result(item)
        self.assertEqual(len(self.pools), 2)

    def test_callback_during_terminate_does_not_deadlock(self):
        self.submit()
        old = self.pools[0]
        joined = []

        def terminate():
            # Pool.terminate joins its result-handler thread. Its callback must
            # not wait for the lifecycle lock held by the rebuilding thread.
            callback = threading.Thread(target=old.succeed, daemon=True)
            callback.start()
            callback.join(timeout=5)
            joined.append(not callback.is_alive())

        old.terminate = terminate
        self.executor._rebuild_pool()
        self.assertEqual(joined, [True], "callback deadlocked on rebuild")
        self.assertEqual(len(self.pools), 2)

    def test_submit_broken_pipe_retries_once(self):
        self.pools[0].submit_error = BrokenPipeError("broken pipe")
        item = self.submit()
        self.assertEqual(len(self.pools), 2)
        self.pools[1].succeed()
        self.executor.get_result(item)

    def test_submit_retry_error_propagates(self):
        self.pools[0].submit_error = BrokenPipeError("first pipe")
        broken = FakePool()
        broken.submit_error = BrokenPipeError("retry pipe")
        self.context.Pool.side_effect = lambda **kwargs: broken
        with self.assertRaisesRegex(BrokenPipeError, "retry pipe"):
            self.submit()
        self.assertEqual(self.context.Pool.call_count, 2)

    def test_current_broken_pipe_rebuilds(self):
        item = self.submit()
        self.pools[0].fail(EOFError("worker exited"))
        with self.assertRaisesRegex(EOFError, "worker exited"):
            self.executor.get_result(item)
        self.assertEqual(len(self.pools), 2)

    def test_worker_timeout_is_not_a_pool_wait_timeout(self):
        for error in (TimeoutError("download"), multiprocessing.TimeoutError("decode")):
            with self.subTest(error=error):
                item = self.submit()
                self.pools[0].fail(error, index=len(self.pools[0].tasks) - 1)
                with self.assertRaises(type(error)) as raised:
                    self.executor.get_result(item)
                self.assertIs(raised.exception, error)
                self.assertEqual(self.executor._consecutive_timeouts, 0)
                self.assertEqual(len(self.pools), 1)

    def test_worker_value_error_propagates_without_rebuild(self):
        item = self.submit()
        self.pools[0].fail(ValueError("bad video"))
        with self.assertRaisesRegex(ValueError, "bad video"):
            self.executor.get_result(item)
        self.assertEqual(len(self.pools), 1)

    def test_failed_rebuild_can_retry_on_next_submit(self):
        item = self.submit(timeout_ms=0)
        self.executor._consecutive_timeouts = 1
        self.context.Pool.side_effect = OSError("spawn failed")
        with self.assertRaisesRegex(TimeoutError, "Preprocessing timeout"):
            self.executor.get_result(item)
        self.assertIsNone(self.executor.pool)
        self.assertEqual(self.executor._consecutive_timeouts, 0)
        self.context.Pool.side_effect = self.make_pool
        fresh = self.submit()
        self.pools[1].succeed()
        self.executor.get_result(fresh)

    def test_timeout_result_is_removed_and_late_success_ignored(self):
        item = self.submit(timeout_ms=0)
        with self.assertRaises(TimeoutError):
            self.executor.get_result(item)
        self.assertEqual(len(item.preprocess_pool_generation._pending), 0)
        self.pools[0].succeed()
        with self.assertRaises(TimeoutError):
            self.executor.get_result(item)
        self.assertEqual(self.executor._consecutive_timeouts, 1)

    def test_shutdown_wakes_pending_and_rejects_new_submissions(self):
        item = self.submit()
        self.executor.shutdown()
        self.executor.shutdown()
        with self.assertRaisesRegex(RuntimeError, "was shut down"):
            self.executor.get_result(item)
        with self.assertRaisesRegex(RuntimeError, "is shut down"):
            self.submit()
        with self.assertRaisesRegex(RuntimeError, "is shut down"):
            self.executor._rebuild_pool()
        self.assertIsNone(self.executor.pool)
        self.assertEqual(len(self.pools), 1)

    def test_stale_failure_after_shutdown_does_not_create_pool(self):
        item = self.submit()
        self.pools[0].fail(OSError("worker died"))
        self.executor.shutdown()
        with self.assertRaises(OSError):
            self.executor.get_result(item)
        self.assertEqual(len(self.pools), 1)


class PreprocessPoolSpawnTest(unittest.TestCase):
    def test_real_spawn_callbacks_crash_recovery_and_shutdown(self):
        executor = engine.MultiprocessPreprocessExecutor(
            multiprocessing.get_context("spawn"),
            SimpleNamespace(mm_preprocess_max_workers=1),
            {},
            spawn_preprocess,
        )
        self.addCleanup(executor.shutdown)
        with mock.patch.object(engine.kmonitor, "report"):
            first = work_item(timeout_ms=60000)
            executor.submit(first)
            executor.get_result(first)
            self.assertEqual(first.preprocess_result, ["video"])

            user_timeout = work_item(timeout_ms=10000, value="timeout")
            executor.submit(user_timeout)
            with self.assertRaisesRegex(TimeoutError, "video download timed out"):
                executor.get_result(user_timeout)
            self.assertEqual(executor._generation.number, 0)

            crash = work_item(timeout_ms=1000, value="crash")
            executor.submit(crash)
            with self.assertRaisesRegex(TimeoutError, "Preprocessing timeout"):
                executor.get_result(crash)
            self.assertEqual(executor._generation.number, 1)

            recovered = work_item(timeout_ms=60000)
            executor.submit(recovered)
            executor.get_result(recovered)
            self.assertEqual(recovered.preprocess_result, ["video"])
            executor.shutdown()
            self.assertIsNone(executor.pool)


if __name__ == "__main__":
    unittest.main()
