import unittest
from types import SimpleNamespace

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8PerTensorQuantConfig,
)
from rtp_llm.model_loader.loader import ModelLoader


_MIB = 1024**2


class TestFastsafetensorMemoryEstimate(unittest.TestCase):
    def _make_loader(self, quant_config):
        model_config = SimpleNamespace(
            quant_algo=quant_config,
            eval_model_weight_size=lambda: 100 * _MIB,
        )
        database = SimpleNamespace(get_max_file_size=lambda: 10 * _MIB)
        exported_device = SimpleNamespace(
            get_mem_info=lambda: SimpleNamespace(free=160 * _MIB)
        )

        loader = ModelLoader.__new__(ModelLoader)
        loader._weights_info = SimpleNamespace(
            model_config=model_config,
            _quant_config=quant_config,
        )
        loader._load_config = SimpleNamespace(
            database=database,
            exported_device=exported_device,
            ep_size=1,
            tp_size=1,
        )
        return loader

    def test_online_fp8_per_block_uses_streaming_shard_reserve(self):
        loader = self._make_loader(Fp8BlockWiseQuantConfig(is_quanted=False))

        self.assertTrue(loader._is_memory_enough_for_fastsafetensor())

    def test_other_non_inline_online_fp8_keeps_conservative_double_estimate(self):
        loader = self._make_loader(Fp8PerTensorQuantConfig(is_quanted=False))

        self.assertFalse(loader._is_memory_enough_for_fastsafetensor())


if __name__ == "__main__":
    unittest.main()
