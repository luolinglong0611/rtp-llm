import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.block_map import get_group_tags_for_layers
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextGatedDeltaNetDecode,
    Qwen3NextMetadata,
    _maybe_write_cp_cache_store,
    _write_cp_cache_store,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MixedFMHAImpl


class FakeKVCache:
    def __init__(self, layer_tags: list[list[str]]):
        self.layer_tags = layer_tags

    def get_layer_cache_groups(self, layer_idx: int):
        return [SimpleNamespace(tag=tag) for tag in self.layer_tags[layer_idx]]


class RoutingModel(GptModelBase):
    def __init__(self, fmha_group_tags: list[str] | None):
        nn.Module.__init__(self)
        self.config = object()
        self.parallelism_config = object()
        self.weight = object()
        self.fmha_config = object()
        self.fmha_group_tags = fmha_group_tags

    def _get_fmha_group_tags(self) -> list[str] | None:
        return self.fmha_group_tags


class AttentionInputRoutingTest(unittest.TestCase):
    def test_qwen3_next_cuda_graph_uses_narrow_block_map_view(self):
        block_map = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            kv_cache_kernel_block_id_device=block_map,
        )
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)

        narrowed = decode._get_fla_block_map(attention_inputs)

        self.assertEqual(narrowed.shape, (3, 1))
        self.assertEqual(narrowed.stride(0), block_map.stride(0))
        self.assertEqual(narrowed[:, 0].tolist(), [0, 4, 8])

    def test_qwen3_next_non_graph_keeps_full_block_map(self):
        block_map = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=False,
            kv_cache_kernel_block_id_device=block_map,
        )
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)

        self.assertIs(decode._get_fla_block_map(attention_inputs), block_map)

    def test_cp_cache_store_uses_each_layer_tag_metadata(self):
        layer_inputs = {}
        for tag in ("full", "linear0", "linear1"):
            cache_store_inputs = SimpleNamespace(tag=tag)
            kv_cache = SimpleNamespace(tag=tag)
            cache_store_writer = Mock()
            layer_inputs[tag] = (
                SimpleNamespace(
                    cache_store_inputs=cache_store_inputs,
                    cache_store_writer=cache_store_writer,
                ),
                kv_cache,
            )

        for tag in ("full", "linear0", "linear1"):
            attention_inputs, kv_cache = layer_inputs[tag]
            _write_cp_cache_store(attention_inputs, kv_cache)
            attention_inputs.cache_store_writer.write.assert_called_once_with(
                attention_inputs.cache_store_inputs, kv_cache
            )

    def test_cp_cache_store_skips_layer_without_store_inputs(self):
        cache_store_writer = Mock()
        attention_inputs = SimpleNamespace(
            cache_store_inputs=None, cache_store_writer=cache_store_writer
        )

        _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

        cache_store_writer.write.assert_not_called()

    def test_cp_cache_store_skips_layer_without_writer(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            cache_store_writer=None,
        )

        _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

    def test_non_cp_linear_attention_does_not_write_cache_store(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            cache_store_writer=Mock(),
            context_parallel_info=SimpleNamespace(
                prefill_actual_input_lengths_cpu=torch.tensor([1], dtype=torch.int32)
            ),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            kv_cache_block_id=torch.tensor([[1]], dtype=torch.int32),
        )

        _maybe_write_cp_cache_store(
            attention_inputs,
            SimpleNamespace(tag="linear0"),
            Qwen3NextMetadata(),
        )

        attention_inputs.cache_store_writer.write.assert_not_called()

    def test_get_group_tags_for_model_selected_layers(self):
        kv_cache = FakeKVCache([["full"], ["linear0"], ["linear1"], ["full", "aux"]])

        self.assertEqual(get_group_tags_for_layers(kv_cache, [0, 3]), ["full", "aux"])

    def test_prepare_fmha_impl_only_for_model_selected_tags(self):
        inputs_by_tag = {
            "full": object(),
            "linear0": object(),
            "linear1": object(),
        }
        inputs = SimpleNamespace(attention_inputs=inputs_by_tag)
        model = RoutingModel(["full"])

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                group_inputs
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertEqual(fmha_impl, {"full": inputs_by_tag["full"]})
        factory.assert_called_once()

    def test_default_model_prepares_every_tag(self):
        inputs_by_tag = {"group0": object(), "group1": object()}
        inputs = SimpleNamespace(attention_inputs=inputs_by_tag)
        model = RoutingModel(None)

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                group_inputs
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs)

        self.assertEqual(fmha_impl, inputs_by_tag)
        self.assertEqual(factory.call_count, 2)

    def test_prepare_mixed_fmha_pairs_decode_and_context_by_tag(self):
        decode_inputs = {"full": SimpleNamespace(sequence_lengths=torch.tensor([7, 8]))}
        context_inputs = {
            "full": SimpleNamespace(sequence_lengths=torch.empty(0, dtype=torch.int32))
        }
        inputs = SimpleNamespace(
            attention_inputs=decode_inputs,
            is_mixed_batch=True,
            mixed_context_attention_inputs=context_inputs,
        )
        model = RoutingModel(["full"])

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                SimpleNamespace(source=group_inputs)
            ),
        ) as factory:
            mixed = model.prepare_fmha_impl(inputs)

        self.assertEqual(factory.call_count, 2)
        self.assertIsInstance(mixed["full"], MixedFMHAImpl)
        self.assertIs(mixed["full"].decode_impl.source, decode_inputs["full"])
        self.assertIs(mixed["full"].context_impl.source, context_inputs["full"])
        self.assertEqual(mixed["full"].decode_token_count, 2)

    def test_mixed_fmha_splits_attention_only_and_restores_token_order(self):
        class FakeImpl:
            def __init__(self, offset: int, return_heads: bool = False) -> None:
                self.offset = offset
                self.return_heads = return_heads
                self.calls: list[tuple[torch.Tensor, object, int]] = []

            def forward(self, qkv, kv_cache, layer_idx=0):
                self.calls.append((qkv, kv_cache, layer_idx))
                output = qkv + self.offset
                return output.unsqueeze(1) if self.return_heads else output

        decode_impl = FakeImpl(10)
        context_impl = FakeImpl(100, return_heads=True)
        mixed = MixedFMHAImpl(decode_impl, context_impl, decode_token_count=2)
        qkv = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        kv_cache = object()

        output = mixed.forward(qkv, kv_cache, layer_idx=3)

        self.assertTrue(torch.equal(output[:2], qkv[:2] + 10))
        self.assertTrue(torch.equal(output[2:], qkv[2:] + 100))
        self.assertEqual(decode_impl.calls[0][0].shape, (2, 2))
        self.assertEqual(context_impl.calls[0][0].shape, (3, 2))
        self.assertIs(decode_impl.calls[0][1], kv_cache)
        self.assertEqual(context_impl.calls[0][2], 3)


if __name__ == "__main__":
    unittest.main()
