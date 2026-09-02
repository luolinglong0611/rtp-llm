import unittest

import torch

from rtp_llm.ops.fused_rope_kvcache_op import _get_prefill_position_ids_keyword


class FusedRopeKVCacheCompatTest(unittest.TestCase):
    def test_uses_position_ids_for_existing_kernel_api(self):
        def prefill_op(*, position_ids=None) -> torch.Tensor:
            return position_ids

        self.assertEqual(
            _get_prefill_position_ids_keyword(prefill_op), "position_ids"
        )

    def test_uses_cp_position_ids_for_cuda13_arm_kernel_api(self):
        def prefill_op(*, cp_position_ids=None) -> torch.Tensor:
            return cp_position_ids

        self.assertEqual(
            _get_prefill_position_ids_keyword(prefill_op), "cp_position_ids"
        )

    def test_uses_legacy_keyword_for_generic_kwargs_api(self):
        def prefill_op(**kwargs) -> torch.Tensor:
            return kwargs["position_ids"]

        self.assertEqual(
            _get_prefill_position_ids_keyword(prefill_op), "position_ids"
        )


if __name__ == "__main__":
    unittest.main()
