"""Shared fixtures and helpers for sglcap tests."""

import pytest
from dataclasses import dataclass
from typing import Optional, List

from sglang.srt.server_args import ServerArgs

from sglcap.server_args import override_gpu_memory
from sglcap.calculator import calculate_capacity, CapacityReport
from sglcap.gpu_db import get_gpu_spec


@dataclass
class ExpectedValues:
    """Expected values from SGLang logs for comparison."""
    # Cell size
    cell_size_kb: float
    attention_type: str  # "MHA" or "MLA"

    # KV cache params (MHA)
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # KV cache params (MLA)
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    # Common
    num_layers: int = 0
    context_len: int = 0

    # req_to_token_pool
    req_pool_slots: int = 0

    # Limits
    max_running_requests: int = 0
    max_req_len: int = 0
    max_prefill_tokens: int = 0


def run_sglcap(
    model_path: str,
    gpu_type: str = "l40s",
    tp_size: int = 1,
    dp_size: int = 1,
    enable_dp_attention: bool = False,
    mem_fraction_static: float = 0.88,
    max_total_tokens: Optional[int] = None,
    max_prefill_tokens: Optional[int] = None,
    kv_cache_dtype: str = "auto",
    chunked_prefill_size: Optional[int] = None,
    attention_backend: Optional[str] = None,
) -> CapacityReport:
    """
    Run sglcap calculator and return the report.

    This mirrors CLI usage but returns the report directly for testing.
    """
    gpu_spec = get_gpu_spec(gpu_type)
    if gpu_spec is None:
        raise ValueError(f"Unknown GPU type: {gpu_type}")

    gpu_memory_gb = gpu_spec.memory_gb
    gpus_per_node = gpu_spec.gpus_per_node

    # Build args list matching CLI
    args = [
        "--model-path", model_path,
        "--tp-size", str(tp_size),
        "--dp-size", str(dp_size),
        "--mem-fraction-static", str(mem_fraction_static),
        "--kv-cache-dtype", kv_cache_dtype,
    ]

    if enable_dp_attention:
        args.append("--enable-dp-attention")

    if max_total_tokens is not None:
        args.extend(["--max-total-tokens", str(max_total_tokens)])

    if max_prefill_tokens is not None:
        args.extend(["--max-prefill-tokens", str(max_prefill_tokens)])

    if chunked_prefill_size is not None:
        args.extend(["--chunked-prefill-size", str(chunked_prefill_size)])

    if attention_backend is not None:
        args.extend(["--attention-backend", attention_backend])

    # Create ServerArgs with GPU memory override
    import argparse
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed_args = parser.parse_args(args)

    gpu_memory_mb = int(gpu_memory_gb * 1024)
    with override_gpu_memory(gpu_memory_mb):
        server_args = ServerArgs.from_cli_args(parsed_args)

    # Calculate capacity
    report = calculate_capacity(
        server_args=server_args,
        gpu_memory_gb=gpu_memory_gb,
        gpus_per_node=gpus_per_node,
    )

    return report


def assert_report_matches(report: CapacityReport, expected: ExpectedValues, tolerance: float = 0.01):
    """
    Assert that report values match expected values from SGLang.

    Args:
        report: Calculated capacity report
        expected: Expected values from SGLang logs
        tolerance: Relative tolerance for float comparisons (default 1%)
    """
    # Cell size (must match exactly - this is the core calculation)
    assert abs(report.cell_size_kb - expected.cell_size_kb) < 0.01, \
        f"Cell size mismatch: {report.cell_size_kb} != {expected.cell_size_kb}"

    # Attention type
    assert report.attention_type == expected.attention_type, \
        f"Attention type mismatch: {report.attention_type} != {expected.attention_type}"

    # KV cache params
    if expected.attention_type == "MHA":
        assert report.num_kv_heads == expected.num_kv_heads, \
            f"KV heads mismatch: {report.num_kv_heads} != {expected.num_kv_heads}"
        assert report.head_dim == expected.head_dim, \
            f"Head dim mismatch: {report.head_dim} != {expected.head_dim}"
    else:  # MLA
        assert report.kv_lora_rank == expected.kv_lora_rank, \
            f"KV lora rank mismatch: {report.kv_lora_rank} != {expected.kv_lora_rank}"
        assert report.qk_rope_head_dim == expected.qk_rope_head_dim, \
            f"QK rope head dim mismatch: {report.qk_rope_head_dim} != {expected.qk_rope_head_dim}"

    # Layers
    assert report.num_hidden_layers == expected.num_layers, \
        f"Layers mismatch: {report.num_hidden_layers} != {expected.num_layers}"

    # Context length
    assert report.context_len == expected.context_len, \
        f"Context len mismatch: {report.context_len} != {expected.context_len}"

    # req_to_token_pool slots
    assert report.req_to_token_pool_size == expected.req_pool_slots, \
        f"Req pool slots mismatch: {report.req_to_token_pool_size} != {expected.req_pool_slots}"

    # max_running_requests
    assert report.max_running_requests == expected.max_running_requests, \
        f"Max running requests mismatch: {report.max_running_requests} != {expected.max_running_requests}"

    # max_req_len
    assert report.max_req_len == expected.max_req_len, \
        f"Max req len mismatch: {report.max_req_len} != {expected.max_req_len}"

    # max_prefill_tokens
    assert report.max_prefill_tokens == expected.max_prefill_tokens, \
        f"Max prefill tokens mismatch: {report.max_prefill_tokens} != {expected.max_prefill_tokens}"
