#!/usr/bin/env python3
"""
sglcap CLI - SGLang Capacity Calculator

Uses SGLang's ServerArgs CLI parser directly to ensure perfect compatibility.
Adds --gpu and --gpu-memory-gb for offline capacity estimation.

Usage:
    # Specify GPU type
    sglcap --model-path meta-llama/Llama-3.1-8B-Instruct --gpu h100

    # Or specify memory directly
    sglcap --model-path meta-llama/Llama-3.1-8B-Instruct --gpu-memory-gb 80

    # Multi-GPU with FP8 KV cache
    sglcap --model-path deepseek-ai/DeepSeek-V3 \\
        --tp-size 8 --kv-cache-dtype fp8 --gpu a100-80

    # Verbose output
    sglcap --model-path meta-llama/Llama-3.1-8B-Instruct --gpu h100 -v
"""

# Must set up compat layer FIRST before any sglang imports
from sglcap._compat_setup import setup_sglang_compat
setup_sglang_compat()

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional

import torch

from sglang.srt.server_args import ServerArgs

from sglcap.server_args import override_gpu_memory
from sglcap.calculator import calculate_capacity, CapacityReport
from sglcap.report import format_report
from sglcap.gpu_db import get_gpu_spec, list_gpus, GPU_DB


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser using SGLang's CLI args."""

    # Build GPU choices string
    gpu_choices = ", ".join(GPU_DB.keys())

    parser = argparse.ArgumentParser(
        prog="sglcap",
        description="SGLang Capacity Calculator - Estimate capacity before deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Specify GPU type
    sglcap --model-path meta-llama/Llama-3.1-8B-Instruct --gpu h100

    # Or specify memory directly
    sglcap --model-path meta-llama/Llama-3.1-8B-Instruct --gpu-memory-gb 80

    # Multi-GPU DeepSeek V3 on A100-80
    sglcap --model-path deepseek-ai/DeepSeek-V3 \\
        --tp-size 8 --trust-remote-code --gpu a100-80

    # With FP8 KV cache on L40S
    sglcap --model-path meta-llama/Llama-3.1-70B-Instruct \\
        --gpu l40s --kv-cache-dtype fp8 --tp-size 4

    # Verbose with calculation details
    sglcap --model-path meta-llama/Llama-3.1-8B-Instruct --gpu h100 -v

Supported GPUs: {gpu_choices}
        """,
    )

    # Use SGLang's CLI args - this adds ALL server args
    ServerArgs.add_cli_args(parser)

    # Add sglcap-specific args
    sglcap_group = parser.add_argument_group("sglcap options")
    sglcap_group.add_argument(
        "--gpu",
        type=str,
        help=f"GPU type ({gpu_choices})",
    )
    sglcap_group.add_argument(
        "--gpu-memory-gb",
        type=float,
        help="GPU memory in GB per device. Overrides --gpu if both provided.",
    )
    sglcap_group.add_argument(
        "--weight-memory-gb",
        type=float,
        help="Override model weight memory in GB (per GPU).",
    )
    sglcap_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed calculation breakdowns",
    )
    sglcap_group.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    sglcap_group.add_argument(
        "--list-gpus",
        action="store_true",
        help="List supported GPU types and exit",
    )
    sglcap_group.add_argument(
        "--tui",
        action="store_true",
        help="Launch interactive TUI with GPU memory visualization",
    )
    sglcap_group.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs in node for TUI visualization (default: 8)",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle --list-gpus
    if args.list_gpus:
        print(list_gpus())
        return 0

    # Extract sglcap-specific args
    gpu_type = args.gpu
    gpu_memory_gb = args.gpu_memory_gb
    weight_memory_gb = args.weight_memory_gb
    verbose = args.verbose
    output_json = args.json
    use_tui = args.tui
    num_gpus = args.num_gpus

    # Resolve GPU memory and node configuration
    gpu_name = None
    gpu_spec = None
    if gpu_memory_gb is not None:
        # Explicit memory takes precedence
        pass
    elif gpu_type is not None:
        # Look up GPU type
        gpu_spec = get_gpu_spec(gpu_type)
        if gpu_spec is None:
            print(f"Error: Unknown GPU type '{gpu_type}'", file=sys.stderr)
            print(list_gpus(), file=sys.stderr)
            return 1
        gpu_memory_gb = gpu_spec.memory_gb
        gpu_name = gpu_spec.name
        # Use GPU-specific node size if user didn't override
        if num_gpus == 8 and gpu_spec.gpus_per_node != 8:
            num_gpus = gpu_spec.gpus_per_node

    try:
        # Create ServerArgs with GPU memory override
        if gpu_memory_gb is not None:
            gpu_memory_mb = int(gpu_memory_gb * 1024)
            with override_gpu_memory(gpu_memory_mb):
                server_args = ServerArgs.from_cli_args(args)
        else:
            server_args = ServerArgs.from_cli_args(args)

        # Use tp_size from server_args as the number of GPUs if larger
        if server_args.tp_size > num_gpus:
            num_gpus = server_args.tp_size

        # Calculate capacity
        gpus_per_node = gpu_spec.gpus_per_node if gpu_spec else 8
        report = calculate_capacity(
            server_args=server_args,
            gpu_memory_gb=gpu_memory_gb,
            weight_memory_gb=weight_memory_gb,
            gpus_per_node=gpus_per_node,
        )

        # Update GPU source if we used a named GPU
        if gpu_name:
            # Monkey-patch the source for display
            object.__setattr__(report, 'gpu_memory_source', gpu_name)

        # Output
        if use_tui:
            from sglcap.tui import run_tui  # Now imports from sglcap/tui/
            run_tui(
                report,
                num_gpus=num_gpus,
                gpu_type=gpu_type or "h100",
                gpus_per_node=gpus_per_node,
            )
        elif output_json:
            data = asdict(report)
            # Convert torch dtypes to strings for JSON
            data["dtype"] = report.dtype_str
            data["kv_cache_dtype"] = report.kv_cache_dtype_str
            # Remove torch.dtype objects that can't be JSON serialized
            del data["kv_cache_dtype"]  # Use kv_cache_dtype_str instead
            print(json.dumps(data, indent=2, default=str))
        else:
            print(format_report(report, verbose=verbose))

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
