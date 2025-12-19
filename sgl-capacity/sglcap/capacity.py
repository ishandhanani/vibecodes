"""
Capacity calculator using ONLY SGLang functions.

This module is the single source of truth for all capacity calculations.
All formulas use SGLang's ModelConfig properties and helper functions directly.

The distributed state (attention_tp_size, world_group) is mocked to allow
capacity calculation without actual torch.distributed initialization.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# Setup mocks BEFORE any sglang imports
from sglcap._compat_setup import (
    setup_sglang_compat,
    call_profile_max_num_token,
    compute_attention_tp_size,
)
setup_sglang_compat()

# Import SGLang's actual classes and functions
from sglang.srt.configs.model_config import ModelConfig, AttentionArch
from sglang.srt.server_args import ServerArgs


# ============================================================================
# KV Cache Dtype Resolution
# ============================================================================

def resolve_kv_cache_dtype(
    kv_cache_dtype_arg: str,
    model_dtype: torch.dtype,
) -> torch.dtype:
    """
    Resolve KV cache dtype from CLI argument.

    Matches SGLang's ModelRunner.init_memory_pool() dtype resolution.
    """
    if kv_cache_dtype_arg == "auto":
        return model_dtype
    elif kv_cache_dtype_arg == "fp8_e5m2":
        return torch.float8_e5m2
    elif kv_cache_dtype_arg in ("fp8_e4m3", "fp8"):
        return torch.float8_e4m3fn
    elif kv_cache_dtype_arg in ("bf16", "bfloat16"):
        return torch.bfloat16
    elif kv_cache_dtype_arg in ("fp16", "float16"):
        return torch.float16
    elif kv_cache_dtype_arg == "fp4_e2m1":
        if hasattr(torch, "float4_e2m1fn_x2"):
            return torch.float4_e2m1fn_x2
        return model_dtype
    else:
        return model_dtype


# ============================================================================
# Cell Size Calculation
# ============================================================================

@dataclass
class CellSizeResult:
    """Result of cell size calculation."""
    cell_size: int  # Bytes per token
    is_mla: bool
    num_layers: int
    dtype_bytes: int
    formula: str
    breakdown: str

    # MLA params (only set if is_mla=True)
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    # MHA/GQA params (only set if is_mla=False)
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None


def calculate_cell_size(
    model_config: ModelConfig,
    kv_cache_dtype: torch.dtype,
    tp_size: int,
    dp_size: int = 1,
    enable_dp_attention: bool = False,
) -> CellSizeResult:
    """
    Calculate KV cache cell size (bytes per token) using SGLang's ModelConfig.

    This matches SGLang's ModelRunner.profile_max_num_token() cell_size calculation.
    Sets up mocked distributed state so SGLang's get_attention_tp_size() returns
    the correct value.

    Args:
        model_config: SGLang ModelConfig (from ServerArgs.get_model_config())
        kv_cache_dtype: KV cache dtype
        tp_size: Tensor parallel size
        dp_size: Data parallel size
        enable_dp_attention: Whether DP attention is enabled

    Returns:
        CellSizeResult with cell size and component details
    """
    # Get dtype size
    dtype_bytes = torch._utils._element_size(kv_cache_dtype)

    # Get layer count (handle hybrid models with full_attention_layer_ids)
    full_attention_layer_ids = getattr(model_config, 'full_attention_layer_ids', None)
    if full_attention_layer_ids:
        num_layers = len(full_attention_layer_ids)
    else:
        num_layers = model_config.num_hidden_layers

    # Compute attention TP size (simple formula, no mocking needed)
    attention_tp_size = compute_attention_tp_size(tp_size, dp_size, enable_dp_attention)

    # Check if MLA (Multi-head Latent Attention)
    is_mla = model_config.attention_arch == AttentionArch.MLA

    if is_mla:
        # MLA: compressed KV representation
        # Formula from SGLang's ModelRunner.profile_max_num_token():
        # cell_size = (kv_lora_rank + qk_rope_head_dim) * num_layers * dtype_bytes
        kv_lora_rank = model_config.kv_lora_rank
        qk_rope_head_dim = model_config.qk_rope_head_dim

        cell_size = (kv_lora_rank + qk_rope_head_dim) * num_layers * dtype_bytes

        formula = "(kv_lora_rank + qk_rope_head_dim) × layers × dtype_bytes"
        breakdown = f"({kv_lora_rank} + {qk_rope_head_dim}) × {num_layers} × {dtype_bytes} = {cell_size:,}"

        return CellSizeResult(
            cell_size=cell_size,
            is_mla=True,
            num_layers=num_layers,
            dtype_bytes=dtype_bytes,
            formula=formula,
            breakdown=breakdown,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    else:
        # MHA/GQA: standard K and V tensors
        # Formula from SGLang's ModelRunner.profile_max_num_token():
        # cell_size = num_kv_heads(attention_tp_size) * (head_dim + v_head_dim) * num_layers * dtype_bytes
        num_kv_heads = model_config.get_num_kv_heads(attention_tp_size)
        head_dim = model_config.head_dim
        v_head_dim = getattr(model_config, 'v_head_dim', None) or head_dim

        cell_size = num_kv_heads * (head_dim + v_head_dim) * num_layers * dtype_bytes

        formula = "kv_heads × (head_dim + v_head_dim) × layers × dtype_bytes"
        breakdown = f"{num_kv_heads} × ({head_dim} + {v_head_dim}) × {num_layers} × {dtype_bytes} = {cell_size:,}"

        return CellSizeResult(
            cell_size=cell_size,
            is_mla=False,
            num_layers=num_layers,
            dtype_bytes=dtype_bytes,
            formula=formula,
            breakdown=breakdown,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
        )


# ============================================================================
# KV Cache Profiling
# ============================================================================

@dataclass
class KVCacheProfile:
    """Result of KV cache capacity profiling."""
    cell_size_result: CellSizeResult
    total_gpu_memory_gb: float
    available_gpu_memory_gb: float
    mem_fraction_static: float
    rest_memory_gb: float  # Memory for KV cache after overhead
    max_total_num_tokens: int

    @property
    def cell_size(self) -> int:
        return self.cell_size_result.cell_size


def profile_kv_cache(
    model_config: ModelConfig,
    server_args: ServerArgs,
    total_gpu_memory_gb: float,
    available_gpu_memory_gb: float,
    kv_cache_dtype: torch.dtype,
) -> KVCacheProfile:
    """
    Profile KV cache capacity using SGLang's actual ModelRunner.profile_max_num_token().

    This function:
    1. Calculates cell_size for reporting (breakdown, formula, etc.)
    2. Calls SGLang's actual profile_max_num_token() for max_total_num_tokens

    Args:
        model_config: SGLang ModelConfig
        server_args: SGLang ServerArgs
        total_gpu_memory_gb: Total GPU memory in GB
        available_gpu_memory_gb: Available GPU memory after weights
        kv_cache_dtype: KV cache dtype

    Returns:
        KVCacheProfile with capacity information
    """
    # Get cell size for detailed breakdown in report
    cell_result = calculate_cell_size(
        model_config=model_config,
        kv_cache_dtype=kv_cache_dtype,
        tp_size=server_args.tp_size,
        dp_size=server_args.dp_size,
        enable_dp_attention=server_args.enable_dp_attention,
    )

    # Call SGLang's actual ModelRunner.profile_max_num_token()
    # This uses the real SGLang function with our mocked environment
    max_total_num_tokens = call_profile_max_num_token(
        model_config=model_config,
        server_args=server_args,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_gb=available_gpu_memory_gb,
    )

    # Calculate rest memory for reporting (matches SGLang's formula)
    rest_memory_gb = available_gpu_memory_gb - total_gpu_memory_gb * (1 - server_args.mem_fraction_static)

    return KVCacheProfile(
        cell_size_result=cell_result,
        total_gpu_memory_gb=total_gpu_memory_gb,
        available_gpu_memory_gb=available_gpu_memory_gb,
        mem_fraction_static=server_args.mem_fraction_static,
        rest_memory_gb=max(0, rest_memory_gb),
        max_total_num_tokens=max(0, max_total_num_tokens),
    )


# ============================================================================
# Request Pool Size Calculation
# ============================================================================

def calculate_req_to_token_pool_size(
    max_total_num_tokens: int,
    context_len: int,
) -> Tuple[int, int]:
    """
    Calculate req_to_token_pool size using SGLang's formula.

    Formula from ModelRunner.init_memory_pool():
        heuristic = max_total_tokens / context_len × 512
        size = clamp(heuristic, 2048, 4096)

    Args:
        max_total_num_tokens: Maximum tokens in KV cache
        context_len: Model context length

    Returns:
        Tuple of (pool_size, raw_heuristic_before_clamp)
    """
    if context_len <= 0:
        return 2048, 0

    heuristic = int(max_total_num_tokens / context_len * 512)
    pool_size = min(max(heuristic, 2048), 4096)
    return pool_size, heuristic


# ============================================================================
# TpWorker Limits Calculation
# ============================================================================

@dataclass
class TpWorkerLimits:
    """Limits computed by TpWorker."""
    max_running_requests: int
    max_running_requests_source: str  # "heuristic" or "arg"
    max_req_len: int
    max_req_input_len: int
    max_prefill_tokens: int
    max_queued_requests: Optional[int]


def calculate_tp_worker_limits(
    server_args: ServerArgs,
    model_config: ModelConfig,
    max_total_num_tokens: int,
    req_to_token_pool_size: int,
) -> TpWorkerLimits:
    """
    Calculate TpWorker limits using SGLang's formulas.

    Matches TpWorker.__init__() calculations.

    Args:
        server_args: SGLang ServerArgs
        model_config: SGLang ModelConfig
        max_total_num_tokens: Maximum tokens in KV cache
        req_to_token_pool_size: Size of req_to_token_pool

    Returns:
        TpWorkerLimits with all computed limits
    """
    # max_running_requests calculation from TpWorker.__init__()
    if server_args.max_running_requests is None:
        max_running_requests = min(
            max_total_num_tokens // 2 if max_total_num_tokens > 0 else 0,
            req_to_token_pool_size,
        )
        max_running_requests_source = "heuristic"
    else:
        dp_divisor = server_args.dp_size if server_args.enable_dp_attention else 1
        max_running_requests = min(
            server_args.max_running_requests // dp_divisor,
            req_to_token_pool_size,
        )
        max_running_requests_source = "arg"

    # max_req_len calculation from TpWorker.__init__()
    max_req_len = min(
        model_config.context_len - 1,
        max_total_num_tokens - 1,
    ) if max_total_num_tokens > 0 else 0

    # max_req_input_len (buffer of 5 for output)
    max_req_input_len = max(0, max_req_len - 5)

    # max_prefill_tokens
    max_prefill_tokens = server_args.max_prefill_tokens or 16384

    return TpWorkerLimits(
        max_running_requests=max(0, max_running_requests),
        max_running_requests_source=max_running_requests_source,
        max_req_len=max_req_len,
        max_req_input_len=max_req_input_len,
        max_prefill_tokens=max_prefill_tokens,
        max_queued_requests=server_args.max_queued_requests,
    )


# ============================================================================
# Weight Memory Estimation
# ============================================================================

def estimate_weight_memory_gb(
    model_config: ModelConfig,
    model_path: str,
    tp_size: int,
) -> Tuple[float, float, str]:
    """
    Estimate model weight memory.

    First tries HuggingFace safetensors metadata, then falls back to
    parameter-based estimation.

    Args:
        model_config: SGLang ModelConfig
        model_path: Path to model (for HF metadata lookup)
        tp_size: Tensor parallel size (weights are sharded)

    Returns:
        Tuple of (per_gpu_gb, total_gb, source_description)
    """
    # Try to get actual size from HuggingFace metadata first
    hf_result = _fetch_model_size_from_hf(model_path)
    if hf_result:
        total_gb, total_params = hf_result
        per_gpu_gb = total_gb / tp_size
        return per_gpu_gb, total_gb, "from HuggingFace metadata"

    # Fall back to parameter-based estimation
    total_params = _estimate_model_parameters(model_config)

    # Get quantization info
    quantization = getattr(model_config, "quantization", None)

    # Determine bytes per parameter
    if quantization:
        quant_lower = quantization.lower()
        if "fp4" in quant_lower or "nvfp4" in quant_lower:
            bytes_per_param = 0.5625  # ~4.5 bits for NVFP4
            source = f"estimated ({quantization}, ~4.5-bit)"
        elif "int4" in quant_lower or "w4" in quant_lower or "4bit" in quant_lower:
            bytes_per_param = 0.5
            source = f"estimated ({quantization}, 4-bit)"
        elif "int8" in quant_lower or "w8" in quant_lower or "8bit" in quant_lower:
            bytes_per_param = 1
            source = f"estimated ({quantization}, 8-bit)"
        elif "fp8" in quant_lower:
            bytes_per_param = 1
            source = f"estimated ({quantization})"
        elif "awq" in quant_lower or "gptq" in quant_lower:
            bytes_per_param = 0.5
            source = f"estimated ({quantization}, assumed 4-bit)"
        else:
            bytes_per_param = torch._utils._element_size(model_config.dtype)
            source = f"estimated ({model_config.dtype})"
    else:
        bytes_per_param = torch._utils._element_size(model_config.dtype)
        dtype_name = str(model_config.dtype).replace("torch.", "")
        source = f"estimated ({dtype_name})"

    total_gb = (total_params * bytes_per_param) / (1024**3)
    per_gpu_gb = total_gb / tp_size

    return per_gpu_gb, total_gb, source


def _fetch_model_size_from_hf(model_path: str) -> Optional[Tuple[float, int]]:
    """Fetch model size from HuggingFace safetensors metadata."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None

    # Skip local paths
    if "/" not in model_path or model_path.startswith("/") or model_path.startswith("."):
        return None

    try:
        api = HfApi()
        metadata = api.get_safetensors_metadata(model_path)

        if not metadata or not metadata.files_metadata:
            return None

        total_bytes = 0
        for file_meta in metadata.files_metadata.values():
            if file_meta.tensors:
                for tensor in file_meta.tensors.values():
                    tensor_size = tensor.data_offsets[1] - tensor.data_offsets[0]
                    total_bytes += tensor_size

        total_params = sum(metadata.parameter_count.values()) if metadata.parameter_count else 0
        size_gb = total_bytes / (1024**3)

        return size_gb, total_params
    except Exception:
        return None


def _estimate_model_parameters(model_config: ModelConfig) -> int:
    """Estimate total model parameters from ModelConfig."""
    hf_config = model_config.hf_config

    hidden = model_config.hidden_size
    layers = model_config.num_hidden_layers
    vocab = model_config.vocab_size
    intermediate = getattr(hf_config, "intermediate_size", hidden * 4)
    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    # Attention projections
    if model_config.attention_arch == AttentionArch.MLA:
        kv_lora_rank = model_config.kv_lora_rank
        qk_nope = getattr(model_config, "qk_nope_head_dim", 0) or 0
        qk_rope = model_config.qk_rope_head_dim
        v_head = model_config.v_head_dim or head_dim

        q_proj = hidden * num_heads * (qk_nope + qk_rope)
        kv_a_proj = hidden * kv_lora_rank
        kv_b_proj = kv_lora_rank * (kv_lora_rank + qk_rope)
        o_proj = num_heads * v_head * hidden
        attn_params = q_proj + kv_a_proj + kv_b_proj + o_proj
    else:
        attn_params = (
            hidden * (num_heads * head_dim)  # Q
            + hidden * (num_kv_heads * head_dim)  # K
            + hidden * (num_kv_heads * head_dim)  # V
            + (num_heads * head_dim) * hidden  # O
        )

    # MLP projections
    num_experts = (
        getattr(hf_config, "num_local_experts", None) or
        getattr(hf_config, "num_experts", None) or
        getattr(hf_config, "n_routed_experts", None)
    )

    if num_experts and num_experts > 1:
        moe_intermediate = getattr(hf_config, "moe_intermediate_size", intermediate)
        mlp_params = 3 * hidden * moe_intermediate * num_experts
        mlp_params += hidden * num_experts  # Router
    else:
        mlp_params = 3 * hidden * intermediate

    params_per_layer = attn_params + mlp_params

    # Embeddings
    embedding_params = vocab * hidden
    tie_embeddings = getattr(hf_config, "tie_word_embeddings", False)
    lm_head_params = 0 if tie_embeddings else vocab * hidden

    # LayerNorms
    layernorm_params = layers * hidden * 2 + hidden

    total_params = layers * params_per_layer + embedding_params + lm_head_params + layernorm_params
    return total_params


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Dtype
    "resolve_kv_cache_dtype",
    # Cell size
    "CellSizeResult",
    "calculate_cell_size",
    # KV cache
    "KVCacheProfile",
    "profile_kv_cache",
    # Request pool
    "calculate_req_to_token_pool_size",
    # TpWorker limits
    "TpWorkerLimits",
    "calculate_tp_worker_limits",
    # Weight memory
    "estimate_weight_memory_gb",
]
