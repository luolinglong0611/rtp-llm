from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.qwen3_next_config import (
    Qwen3NextConfig,
    Qwen35DenseConfig,
    Qwen35MoeConfig,
)
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    Qwen3NextWeight,
    Qwen35DenseWeight,
    Qwen35MoeWeight,
)


class Qwen3NextBase(BaseModel):
    _CONFIG_CLS = Qwen3NextConfig

    def _create_python_model(self):
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextModel

        self.py_model = Qwen3NextModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return True

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        return cls._CONFIG_CLS._create_config(ckpt_path)

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        cls._CONFIG_CLS._post_build_model_config(model_config)

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        cls._CONFIG_CLS._parse_rope_config(config_json, config)


class Qwen3Next(Qwen3NextBase):
    _CONFIG_CLS = Qwen3NextConfig

    @staticmethod
    def get_weight_cls():
        return Qwen3NextWeight


class Qwen35Moe(Qwen3NextBase):
    _CONFIG_CLS = Qwen35MoeConfig

    @staticmethod
    def get_weight_cls():
        return Qwen35MoeWeight

    def _create_python_model(self):
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        from rtp_llm.models_py.utils.arch import (
            get_device_type,
            is_cuda,
            is_hip,
            is_ppu,
        )

        # Per-model allowlist: a device belongs here once it has the attention,
        # MoE and MRoPE impls Qwen35Model needs. Naming the device in the message
        # keeps it truthful as the list grows.
        if not is_cuda() and not is_hip() and not is_ppu():
            raise RuntimeError(
                "Qwen3Next has no python-model implementation for "
                f"{get_device_type().name}"
            )
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen35Model

        self.py_model = Qwen35Model(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


class Qwen35Dense(Qwen35Moe):
    _CONFIG_CLS = Qwen35DenseConfig

    @staticmethod
    def get_weight_cls():
        return Qwen35DenseWeight


register_model("qwen3_next", Qwen3Next, ["Qwen3NextForCausalLM"])
register_model("qwen35_moe", Qwen35Moe, ["Qwen3_5MoeForConditionalGeneration"])
register_model("qwen35_dense", Qwen35Dense, ["Qwen3_5ForConditionalGeneration"])
