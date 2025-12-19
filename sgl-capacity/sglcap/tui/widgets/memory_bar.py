"""
GPU Memory bar visualization widget.
"""

from textual.widgets import Static
from rich.text import Text

from sglcap.calculator import CapacityReport


class MemoryBarWidget(Static):
    """
    Horizontal bar showing GPU memory allocation.

    Visualizes:
    - Model weights (cyan)
    - KV cache budget (green)
    - Available/overhead (dim)
    """

    DEFAULT_CSS = """
    MemoryBarWidget {
        width: 100%;
        height: auto;
        padding: 1;
    }
    """

    def __init__(self, report: CapacityReport, **kwargs) -> None:
        super().__init__(**kwargs)
        self.report = report

    def render(self) -> Text:
        r = self.report

        # Calculate percentages
        total = r.gpu_memory_gb
        weights_pct = (r.weight_memory_gb / total) * 100 if total > 0 else 0
        kv_pct = (max(0, r.kv_cache_budget_gb) / total) * 100 if total > 0 else 0
        overhead_gb = max(0, total - r.weight_memory_gb - max(0, r.kv_cache_budget_gb))

        # Build a horizontal bar (50 chars wide)
        bar_width = 50
        weights_chars = max(1, int((weights_pct / 100) * bar_width)) if weights_pct > 0 else 0
        kv_chars = max(1, int((kv_pct / 100) * bar_width)) if kv_pct > 0 else 0

        # Ensure we don't exceed bar width
        if weights_chars + kv_chars > bar_width:
            scale = bar_width / (weights_chars + kv_chars)
            weights_chars = max(1, int(weights_chars * scale))
            kv_chars = bar_width - weights_chars

        avail_chars = bar_width - weights_chars - kv_chars

        bar = (
            f"[cyan]{'█' * weights_chars}[/cyan]"
            f"[green]{'█' * kv_chars}[/green]"
            f"[dim]{'░' * avail_chars}[/dim]"
        )

        lines = [
            f"[bold]GPU Memory[/bold] ({r.gpu_memory_gb:.0f} GB × {r.tp_size} GPUs = {r.gpu_memory_gb * r.tp_size:.0f} GB total)",
            "",
            f"  ┌{'─' * bar_width}┐",
            f"  │{bar}│",
            f"  └{'─' * bar_width}┘",
            "",
            f"  [cyan]█[/cyan] Weights:  {r.weight_memory_gb:>5.1f} GB/GPU ({weights_pct:>4.0f}%)  "
            f"│ {r.weight_memory_total_gb:>6.1f} GB total",
            f"  [green]█[/green] KV Cache: {max(0, r.kv_cache_budget_gb):>5.1f} GB/GPU ({kv_pct:>4.0f}%)  "
            f"│ {max(0, r.kv_cache_budget_gb) * r.tp_size:>6.1f} GB total",
            f"  [dim]░[/dim] Overhead: {overhead_gb:>5.1f} GB/GPU ({100 - weights_pct - kv_pct:>4.0f}%)",
        ]

        return Text.from_markup("\n".join(lines))
