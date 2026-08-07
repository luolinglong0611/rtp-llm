import asyncio
import json
from typing import Any
from unittest import TestCase, main

from pydantic import BaseModel

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import init_controller, set_global_controller


class FakePipelinResponse(BaseModel):
    res: str


class FakeFrontendWorker(object):
    class FakeBackendRpcServerVisitor:
        def __init__(self):
            self.refresh_calls = []

        def is_backend_service_ready(self, refresh: bool = False):
            self.refresh_calls.append(refresh)
            return True

    def __init__(self):
        self.backend_rpc_server_visitor = self.FakeBackendRpcServerVisitor()
        self.close_called = False

    async def close(self):
        self.close_called = True

    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        response_generator = self._inference(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(
            response_generator, CompleteResponseAsyncGenerator.get_last_value
        )

    def tokenizer_encode(self, prompt: str):
        return [1, 2, 3, 4], ["b", "c", "d", "e"]

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kwargs: Any):
        return False


class FakeRawRequest(object):
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}

    async def is_disconnected(self):
        return False


async def failing_stream():
    yield FakePipelinResponse(res="partial")
    raise RuntimeError("backend stream failed")


class FrontendServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Create PyEnvConfigs with default values for testing
        py_env_configs = PyEnvConfigs()
        set_global_controller(init_controller(py_env_configs.concurrency_config))
        py_env_configs.server_config.start_port = 0
        py_env_configs.server_config.rank_id = 0
        self.frontend_server = FrontendServer(
            rank_id=0,
            server_id=0,
            py_env_configs=py_env_configs,
        )
        self.frontend_server._frontend_worker = FakeFrontendWorker()

    async def _async_run(self, *args: Any, **kwargs: Any):
        res = await self.frontend_server.inference(*args, **kwargs)
        return res

    def test_simple(self):
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(
            self._async_run(req={"prompt": "hello"}, raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )
        res = loop.run_until_complete(
            self._async_run(req='{"prompt": "hello"}', raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )

    def test_encode(self):
        res = self.frontend_server.tokenizer_encode('{"prompt": "b c d e"}')
        self.assertEqual(
            res.body.decode("utf-8"),
            '{"token_ids":[1,2,3,4],"tokens":["b","c","d","e"],"error":""}',
        )
        # test error input
        res = self.frontend_server.tokenizer_encode('{"text": "b c d e"}')
        self.assertEqual(json.loads(res.body.decode("utf-8"))["error_code"], 514)

    def test_check_health_uses_cached_service_discovery(self):
        self.assertTrue(self.frontend_server.check_health())
        visitor = self.frontend_server._frontend_worker.backend_rpc_server_visitor
        self.assertEqual(visitor.refresh_calls, [False])
    def test_close_uses_production_frontend_server_contract(self):
        asyncio.run(self.frontend_server.close())

        self.assertTrue(self.frontend_server._frontend_worker.close_called)

    def test_openai_stream_error_has_error_envelope_and_done_event(self):
        async def collect_chunks():
            response = CompleteResponseAsyncGenerator(
                failing_stream(), CompleteResponseAsyncGenerator.get_last_value
            )
            self.frontend_server._global_controller.increment()
            return [
                chunk
                async for chunk in self.frontend_server.stream_response(
                    {"stream": True, "source": "test", "__request_id__": 1},
                    response,
                )
            ]

        chunks = asyncio.run(collect_chunks())

        self.assertEqual(chunks[0], 'data: {"res":"partial"}\r\n\r\n')
        error_payload = json.loads(chunks[1].removeprefix("data: "))
        self.assertIn("error", error_payload)
        self.assertIn("backend stream failed", error_payload["error"]["message"])
        self.assertEqual(chunks[2], "data: [DONE]\r\n\r\n")

    def test_internal_stream_error_keeps_legacy_envelope_and_done_event(self):
        async def collect_chunks():
            response = CompleteResponseAsyncGenerator(
                failing_stream(), CompleteResponseAsyncGenerator.get_last_value
            )
            self.frontend_server._global_controller.increment()
            return [
                chunk
                async for chunk in self.frontend_server.stream_response(
                    {"stream": False, "source": "test", "__request_id__": 2},
                    response,
                )
            ]

        chunks = asyncio.run(collect_chunks())

        error_payload = json.loads(chunks[1].removeprefix("data:"))
        self.assertNotIn("error", error_payload)
        self.assertIn("backend stream failed", error_payload["message"])
        self.assertEqual(chunks[2], "data:[done]\r\n\r\n")


main()
