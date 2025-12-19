"""
Model configuration wrapper.

Wraps SGLang's ModelConfig and provides easy access to model parameters
needed for capacity calculations.
"""

from dataclasses import dataclass
from typing import Optional

import torch

# Set up mocks BEFORE any sglang imports
from sglcap._compat_setup import setup_sglang_compat
setup_sglang_compat()

# Re-export SGLang's ModelConfig for loading
from sglang.srt.configs.model_config import ModelConfig, AttentionArch


@dataclass
class ModelArchitecture:
    """
    Extracted model architecture parameters for capacity calculations.

    This provides a clean interface to the model parameters we need,
    extracted from SGLang's ModelConfig and hf_config.
    """

    # Basic info
    model_path: str
    architecture_name: str
    dtype: torch.dtype
    dtype_bytes: int

    # Attention
    attention_arch: AttentionArch
    num_attention_heads: int
    num_kv_heads: int  # Total, before TP sharding
    head_dim: int

    # Model dimensions
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    context_len: int

    # Quantization (detected or specified) - optional, must come after required fields
    quantization: Optional[str] = None  # e.g., "fp4", "int4", "awq", "gptq"

    # Sliding window attention (None if not used)
    sliding_window: Optional[int] = None  # Window size in tokens
    has_mixed_attention: bool = False  # True if model has both sliding and full attention layers

    # MLA-specific (None if not MLA)
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # MoE-specific (None if not MoE)
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

    # Embeddings
    tie_word_embeddings: bool = False

    @property
    def is_mla(self) -> bool:
        return self.attention_arch == AttentionArch.MLA

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 1

    @property
    def attention_type_str(self) -> str:
        """Return attention type string matching SGLang's log format.

        SGLang uses MLA vs MHA (no GQA distinction in logs).
        """
        if self.is_mla:
            return "MLA"
        else:
            return "MHA"

    @property
    def attention_type_detail(self) -> str:
        """Return detailed attention type (MLA/MHA/GQA) for model info."""
        if self.is_mla:
            return "MLA"
        elif self.num_kv_heads < self.num_attention_heads:
            return "GQA"
        else:
            return "MHA"

    def get_num_kv_heads_per_tp(self, tp_size: int) -> int:
        """Get number of KV heads per TP rank."""
        return max(1, self.num_kv_heads // tp_size)


def extract_model_architecture(
    model_config: ModelConfig,
    model_path: str,
) -> ModelArchitecture:
    """
    Extract model architecture from SGLang's ModelConfig.

    Args:
        model_config: SGLang ModelConfig (already loaded)
        model_path: Path to model (for reference)

    Returns:
        ModelArchitecture with all parameters needed for capacity calculation
    """
    hf_config = model_config.hf_config

    # Get architecture name
    arch_name = "Unknown"
    if hasattr(hf_config, "architectures") and hf_config.architectures:
        arch_name = hf_config.architectures[0]

    # Dtype
    dtype = model_config.dtype
    dtype_bytes = torch._utils._element_size(dtype)

    # Get quantization from SGLang's ModelConfig (it already handles detection)
    quantization = getattr(model_config, "quantization", None)

    # Get intermediate_size from hf_config
    intermediate_size = getattr(hf_config, "intermediate_size", model_config.hidden_size * 4)

    # MoE parameters from hf_config (different models use different names)
    num_experts = (
        getattr(hf_config, "num_local_experts", None) or
        getattr(hf_config, "num_experts", None) or
        getattr(hf_config, "n_routed_experts", None)  # DeepSeek
    )
    num_experts_per_tok = (
        getattr(hf_config, "num_experts_per_tok", None) or
        getattr(hf_config, "num_activated_experts", None)  # DeepSeek uses this
    )
    moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", intermediate_size)

    # MLA parameters from model_config (set by _derive_model_shapes)
    kv_lora_rank = getattr(model_config, "kv_lora_rank", None)
    qk_rope_head_dim = getattr(model_config, "qk_rope_head_dim", None)
    qk_nope_head_dim = getattr(model_config, "qk_nope_head_dim", None)
    v_head_dim = getattr(model_config, "v_head_dim", None)

    # Embeddings
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # Sliding window attention
    sliding_window = getattr(hf_config, "sliding_window", None)
    layer_types = getattr(hf_config, "layer_types", None)
    has_mixed_attention = (
        layer_types is not None and
        len(set(layer_types)) > 1  # More than one type of attention
    )

    return ModelArchitecture(
        model_path=model_path,
        architecture_name=arch_name,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        attention_arch=model_config.attention_arch,
        num_attention_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        hidden_size=model_config.hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        vocab_size=model_config.vocab_size,
        context_len=model_config.context_len,
        quantization=quantization,
        sliding_window=sliding_window,
        has_mixed_attention=has_mixed_attention,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        tie_word_embeddings=tie_word_embeddings,
    )


__all__ = [
    "ModelConfig",
    "AttentionArch",
    "ModelArchitecture",
    "extract_model_architecture",
]
