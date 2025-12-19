"""
Main capacity calculator.

Uses SGLang's ServerArgs for CLI parsing and ModelConfig for loading.
All calculations are performed using functions from capacity.py which
use SGLang's ModelConfig directly.
"""

from dataclasses import dataclass, field
from typing import Optional, List

import torch

# Set up mocks BEFORE any sglang imports
from sglcap._compat_setup import setup_sglang_compat
setup_sglang_compat()

from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import AttentionArch

# Import all calculations from capacity.py (single source of truth)
from sglcap.capacity import (
    resolve_kv_cache_dtype,
    profile_kv_cache,
    calculate_req_to_token_pool_size,
    calculate_tp_worker_limits,
    estimate_weight_memory_gb,
)
from sglcap.model_config import ModelArchitecture, extract_model_architecture
from sglcap.server_args import get_gpu_tier_info


@dataclass
class CapacityReport:
    """Comprehensive capacity report."""

    # =========================================================================
    # Required fields (no defaults) - must come first
    # =========================================================================

    # Model Info
    model_path: str
    architecture: str
    dtype: torch.dtype
    dtype_str: str

    # GPU Info
    gpu_memory_gb: float
    gpu_memory_source: str  # "override", "detected", "default"
    gpu_tier_description: str

    # Parallelism
    tp_size: int
    dp_size: int
    enable_dp_attention: bool

    # Cluster topology (calculated from TP/DP and GPU type)
    total_gpus: int
    num_nodes: int
    gpus_per_node: int

    # Model Architecture
    attention_type: str  # "MHA" or "MLA" (matches SGLang log format)
    attention_type_detail: str  # "MHA", "GQA", or "MLA" (for model info)
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    context_len: int

    # =========================================================================
    # Optional fields (with defaults) - must come after required fields
    # =========================================================================

    # Quantization
    quantization: Optional[str] = None  # e.g., "fp4", "awq", "gptq"

    # MLA-specific
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    # MoE-specific
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None

    # Sliding window attention
    sliding_window: Optional[int] = None
    has_mixed_attention: bool = False

    # GPU Tier Heuristics (from ServerArgs)
    chunked_prefill_size: Optional[int] = None
    cuda_graph_max_bs: int = 256
    mem_fraction_static: float = 0.88

    # KV Cache
    kv_cache_dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype_str: str = "bfloat16"
    kv_cache_dtype_bytes: int = 2
    cell_size: int = 0
    cell_size_kb: float = 0.0
    cell_size_formula: str = ""
    cell_size_breakdown: str = ""

    # Memory Budget
    weight_memory_gb: float = 0.0  # Per GPU
    weight_memory_total_gb: float = 0.0  # Total across all GPUs
    weight_memory_source: str = "estimated"
    available_memory_gb: float = 0.0
    kv_cache_budget_gb: float = 0.0

    # Max tokens (with tracking for user override)
    max_total_num_tokens: int = 0  # Raw profiled value
    max_total_num_tokens_padded: int = 0  # Page-aligned final value
    max_total_num_tokens_profiled: int = 0  # What profiling calculated
    max_total_num_tokens_arg: Optional[int] = None  # User override if any
    max_total_num_tokens_source: str = "profiled"  # "profiled", "arg", or "min(arg, profiled)"

    # Request Limits
    req_to_token_pool_size: int = 0
    req_to_token_pool_heuristic: int = 0
    max_running_requests: int = 0
    max_running_requests_source: str = "heuristic"
    max_prefill_tokens: int = 0
    max_queued_requests: Optional[int] = None
    max_req_len: int = 0
    max_req_input_len: int = 0

    # Page Size
    page_size: int = 16

    # Capacity Summary
    full_context_requests: int = 0
    requests_at_1k_tokens: int = 0
    requests_at_4k_tokens: int = 0
    requests_at_8k_tokens: int = 0

    # Notes
    notes: List[str] = field(default_factory=list)


def calculate_capacity(
    server_args: ServerArgs,
    gpu_memory_gb: Optional[float] = None,
    weight_memory_gb: Optional[float] = None,
    gpus_per_node: int = 8,
) -> CapacityReport:
    """
    Calculate capacity from ServerArgs.

    All calculations use SGLang's ModelConfig directly via functions in capacity.py.

    Args:
        server_args: Initialized ServerArgs (from CLI)
        gpu_memory_gb: Override GPU memory (None = use detected/default)
        weight_memory_gb: Override weight memory (None = estimate)
        gpus_per_node: GPUs per node (for calculating num_nodes)

    Returns:
        CapacityReport with all calculated values
    """
    notes = []

    # =========================================================================
    # GPU Memory
    # =========================================================================
    if gpu_memory_gb is not None:
        gpu_mem_gb = gpu_memory_gb
        gpu_source = "override"
    else:
        gpu_mem_gb = 80.0
        gpu_source = "default (80GB)"
        notes.append("GPU memory defaulted to 80GB. Use --gpu-memory-gb for accuracy.")

    # =========================================================================
    # Cluster Topology
    # =========================================================================
    if server_args.enable_dp_attention:
        total_gpus = server_args.tp_size
    else:
        total_gpus = server_args.tp_size * server_args.dp_size

    num_nodes = (total_gpus + gpus_per_node - 1) // gpus_per_node

    # =========================================================================
    # Load Model Config via SGLang
    # =========================================================================
    model_config = server_args.get_model_config()

    # Extract architecture into our clean dataclass (for report fields)
    arch = extract_model_architecture(model_config, server_args.model_path)

    # Add note for sliding window attention models
    if arch.sliding_window is not None or arch.has_mixed_attention:
        notes.append(
            f"Model uses sliding window attention (window={arch.sliding_window}). "
            "SGLang does not yet optimize KV cache for SWA - calculations assume full attention "
            "and may overestimate memory requirements."
        )

    # =========================================================================
    # KV Cache Dtype
    # =========================================================================
    kv_cache_dtype = resolve_kv_cache_dtype(server_args.kv_cache_dtype, model_config.dtype)
    kv_dtype_bytes = torch._utils._element_size(kv_cache_dtype)
    kv_dtype_str = str(kv_cache_dtype).replace("torch.", "")

    # =========================================================================
    # Weight Memory (using capacity.py which uses ModelConfig directly)
    # =========================================================================
    if weight_memory_gb is not None:
        weight_mem_gb = weight_memory_gb
        weight_mem_total_gb = weight_memory_gb * server_args.tp_size
        weight_source = "override"
    else:
        weight_mem_gb, weight_mem_total_gb, weight_source = estimate_weight_memory_gb(
            model_config, server_args.model_path, server_args.tp_size
        )
        notes.append(f"Weight memory is {weight_source}. Use --weight-memory-gb for precise value.")

    # =========================================================================
    # KV Cache Profile (using capacity.py which uses ModelConfig directly)
    # =========================================================================
    available_mem_gb = gpu_mem_gb - weight_mem_gb

    kv_profile = profile_kv_cache(
        model_config=model_config,
        server_args=server_args,
        total_gpu_memory_gb=gpu_mem_gb,
        available_gpu_memory_gb=available_mem_gb,
        kv_cache_dtype=kv_cache_dtype,
    )

    # =========================================================================
    # Max Total Tokens - handle user override
    # =========================================================================
    profiled_max_tokens = kv_profile.max_total_num_tokens
    max_tokens_arg = server_args.max_total_tokens

    if max_tokens_arg is not None:
        if max_tokens_arg < profiled_max_tokens:
            max_tokens = max_tokens_arg
            max_tokens_source = "arg"
        else:
            max_tokens = profiled_max_tokens
            max_tokens_source = "min(arg, profiled)"
    else:
        max_tokens = profiled_max_tokens
        max_tokens_source = "profiled"

    max_tokens_padded = (max_tokens // server_args.page_size) * server_args.page_size

    # =========================================================================
    # Request Pool Size (using capacity.py)
    # =========================================================================
    req_pool_size, req_pool_heuristic = calculate_req_to_token_pool_size(
        profiled_max_tokens,
        model_config.context_len,
    )

    # =========================================================================
    # TpWorker Limits (using capacity.py which uses ModelConfig directly)
    # =========================================================================
    tp_limits = calculate_tp_worker_limits(
        server_args=server_args,
        model_config=model_config,
        max_total_num_tokens=max_tokens_padded,
        req_to_token_pool_size=req_pool_size,
    )

    # =========================================================================
    # GPU Tier Info
    # =========================================================================
    tier_info = get_gpu_tier_info(gpu_mem_gb, server_args.tp_size)

    # =========================================================================
    # Attention TP Size (for report display)
    # =========================================================================
    attention_tp_size = (
        server_args.tp_size // server_args.dp_size
        if server_args.enable_dp_attention and server_args.dp_size > 1
        else server_args.tp_size
    )

    # =========================================================================
    # Context length
    # =========================================================================
    ctx_len = model_config.context_len

    # =========================================================================
    # Build Report
    # =========================================================================
    return CapacityReport(
        # Model info
        model_path=server_args.model_path,
        architecture=arch.architecture_name,
        dtype=model_config.dtype,
        dtype_str=str(model_config.dtype).replace("torch.", ""),
        quantization=arch.quantization,

        # GPU info
        gpu_memory_gb=gpu_mem_gb,
        gpu_memory_source=gpu_source,
        gpu_tier_description=tier_info["description"],

        # Parallelism
        tp_size=server_args.tp_size,
        dp_size=server_args.dp_size,
        enable_dp_attention=server_args.enable_dp_attention,

        # Cluster topology
        total_gpus=total_gpus,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,

        # Architecture (from arch dataclass for easy access)
        attention_type=arch.attention_type_str,
        attention_type_detail=arch.attention_type_detail,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.get_num_kv_heads(attention_tp_size),
        head_dim=model_config.head_dim,
        hidden_size=model_config.hidden_size,
        intermediate_size=arch.intermediate_size,
        vocab_size=model_config.vocab_size,
        context_len=ctx_len,

        # MLA
        kv_lora_rank=arch.kv_lora_rank,
        qk_rope_head_dim=arch.qk_rope_head_dim,

        # MoE
        is_moe=arch.is_moe,
        num_experts=arch.num_experts,
        num_experts_per_tok=arch.num_experts_per_tok,

        # Sliding window
        sliding_window=arch.sliding_window,
        has_mixed_attention=arch.has_mixed_attention,

        # GPU tier
        chunked_prefill_size=server_args.chunked_prefill_size,
        cuda_graph_max_bs=server_args.cuda_graph_max_bs,
        mem_fraction_static=server_args.mem_fraction_static,

        # KV cache
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_dtype_str=kv_dtype_str,
        kv_cache_dtype_bytes=kv_dtype_bytes,
        cell_size=kv_profile.cell_size,
        cell_size_kb=kv_profile.cell_size / 1024,
        cell_size_formula=kv_profile.cell_size_result.formula,
        cell_size_breakdown=kv_profile.cell_size_result.breakdown,

        # Memory
        weight_memory_gb=weight_mem_gb,
        weight_memory_total_gb=weight_mem_total_gb,
        weight_memory_source=weight_source,
        available_memory_gb=available_mem_gb,
        kv_cache_budget_gb=kv_profile.rest_memory_gb,
        max_total_num_tokens=max_tokens,
        max_total_num_tokens_padded=max_tokens_padded,
        max_total_num_tokens_profiled=profiled_max_tokens,
        max_total_num_tokens_arg=max_tokens_arg,
        max_total_num_tokens_source=max_tokens_source,

        # Request limits
        req_to_token_pool_size=req_pool_size,
        req_to_token_pool_heuristic=req_pool_heuristic,
        max_running_requests=tp_limits.max_running_requests,
        max_running_requests_source=tp_limits.max_running_requests_source,
        max_prefill_tokens=tp_limits.max_prefill_tokens,
        max_queued_requests=tp_limits.max_queued_requests,
        max_req_len=tp_limits.max_req_len,
        max_req_input_len=tp_limits.max_req_input_len,

        # Page size
        page_size=server_args.page_size,

        # Capacity summary
        full_context_requests=max_tokens_padded // ctx_len if ctx_len > 0 else 0,
        requests_at_1k_tokens=max_tokens_padded // 1000 if max_tokens_padded >= 1000 else 0,
        requests_at_4k_tokens=max_tokens_padded // 4000 if max_tokens_padded >= 4000 else 0,
        requests_at_8k_tokens=max_tokens_padded // 8000 if max_tokens_padded >= 8000 else 0,

        # Notes
        notes=notes,
    )


__all__ = [
    "CapacityReport",
    "calculate_capacity",
]
