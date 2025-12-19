"""
sglcap TUI - Interactive visualization for SGLang capacity calculations.

This module provides a Textual-based TUI for visualizing:
- GPU memory allocation (weights, KV cache, overhead)
- KV cache pool structure (per-layer 3D tensors)
- Request-to-token pool mapping
- Capacity estimates and limits
"""

from sglcap.tui.app import CapacityTUI, run_tui

__all__ = [
    "CapacityTUI",
    "run_tui",
]
