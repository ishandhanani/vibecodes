"""
sglcap - SGLang Capacity Calculator

Offline capacity and sizing estimation for SGLang deployments.
Uses SGLang's actual classes and calculation formulas.

Usage:
    from sglcap import calculate_capacity, GPUSimulation, get_gpu_spec

    # Simulate GB200 GPU
    gpu_spec = get_gpu_spec("gb200")
    with GPUSimulation(gpu_spec):
        from sglcap.server_args import create_server_args
        server_args = create_server_args(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            tp_size=4,
        )
        report = calculate_capacity(server_args, gpu_memory_gb=gpu_spec.memory_gb)
        print(report)
"""

__version__ = "0.1.0"

# Must set up mocks BEFORE any sglang imports
from sglcap._compat_setup import setup_sglang_compat, GPUSimulation
setup_sglang_compat()

from sglcap.calculator import calculate_capacity, CapacityReport
from sglcap.report import format_report
from sglcap.gpu_db import GPUSpec, get_gpu_spec, list_gpus

__all__ = [
    "calculate_capacity",
    "CapacityReport",
    "format_report",
    "GPUSimulation",
    "GPUSpec",
    "get_gpu_spec",
    "list_gpus",
]
