"""
Server arguments - GPU memory override for SGLang's ServerArgs.

This allows running ServerArgs.__post_init__() with a custom GPU memory value,
enabling offline capacity calculations without actual GPU detection.

Usage:
    # Use SGLang's CLI parser directly
    from sglang.srt.server_args import ServerArgs
    from sglcap.server_args import override_gpu_memory

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    with override_gpu_memory(gpu_memory_mb=81920):  # 80GB
        server_args = ServerArgs.from_cli_args(args)
"""

import contextlib
from typing import Any, Dict

# Re-export SGLang's ServerArgs for convenience
from sglang.srt.server_args import ServerArgs
import sglang.srt.utils.common as sglang_utils


@contextlib.contextmanager
def override_gpu_memory(gpu_memory_mb: int):
    """
    Context manager to override GPU memory detection.

    This patches get_device_memory_capacity to return a fixed value,
    allowing ServerArgs.__post_init__ to run with simulated GPU memory.

    Args:
        gpu_memory_mb: GPU memory in MB (e.g., 81920 for 80GB)
    """
    original_func = sglang_utils.get_device_memory_capacity

    def patched_get_device_memory_capacity(device=None):
        return gpu_memory_mb

    try:
        sglang_utils.get_device_memory_capacity = patched_get_device_memory_capacity
        yield
    finally:
        sglang_utils.get_device_memory_capacity = original_func


# GPU tier reference for documentation
GPU_TIERS = {
    "T4": 16,
    "RTX_4080": 16,
    "A10": 24,
    "RTX_4090": 24,
    "RTX_5090": 32,
    "L4": 24,
    "L40": 48,
    "A100_40GB": 40,
    "A100_80GB": 80,
    "H100": 80,
    "H20": 96,
    "H200": 141,
    "B200": 192,
    "MI300X": 192,
}


def get_gpu_tier_info(gpu_memory_gb: float, tp_size: int = 1) -> Dict[str, Any]:
    """
    Get expected heuristics for a GPU memory tier.

    This shows what values ServerArgs._handle_gpu_memory_settings() will compute.

    Returns dict with:
        - chunked_prefill_size: Token budget for chunked prefill
        - cuda_graph_max_bs: Maximum batch size for CUDA graphs
        - description: Human-readable tier description
    """
    gpu_mem_mb = gpu_memory_gb * 1024

    if gpu_mem_mb < 20 * 1024:
        return {
            "chunked_prefill_size": 2048,
            "cuda_graph_max_bs": 8,
            "description": "Low-end (T4, 4080)",
        }
    elif gpu_mem_mb < 35 * 1024:
        return {
            "chunked_prefill_size": 2048,
            "cuda_graph_max_bs": 24 if tp_size < 4 else 80,
            "description": "Mid-range (A10, 4090, 5090)",
        }
    elif gpu_mem_mb < 60 * 1024:
        return {
            "chunked_prefill_size": 4096,
            "cuda_graph_max_bs": 32 if tp_size < 4 else 160,
            "description": "High-end (A100-40GB, L40)",
        }
    elif gpu_mem_mb < 90 * 1024:
        return {
            "chunked_prefill_size": 8192,
            "cuda_graph_max_bs": 256 if tp_size < 4 else 512,
            "description": "Premium (H100, A100-80GB)",
        }
    elif gpu_mem_mb < 160 * 1024:
        return {
            "chunked_prefill_size": 8192,
            "cuda_graph_max_bs": 256 if tp_size < 4 else 512,
            "description": "Enterprise (H20, H200)",
        }
    else:
        return {
            "chunked_prefill_size": 16384,
            "cuda_graph_max_bs": 512,
            "description": "Top-tier (B200, MI300)",
        }


__all__ = [
    "ServerArgs",
    "override_gpu_memory",
    "GPU_TIERS",
    "get_gpu_tier_info",
]
