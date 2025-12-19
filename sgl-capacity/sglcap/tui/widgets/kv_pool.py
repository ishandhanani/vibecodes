"""
KV Cache Pool visualization widget.

Visualizes the structure of the KV cache:
- MHA: List of 3D tensors [token_slots × heads × head_dim] per layer
- MLA: List of 3D tensors [token_slots × 1 × kv_cache_dim] per layer
"""

from textual.widgets import Static
from rich.text import Text

from sglcap.calculator import CapacityReport


class KVPoolWidget(Static):
    """
    Visualize KV cache pool structure.

    Shows the per-layer tensor organization and memory layout.
    """

    DEFAULT_CSS = """
    KVPoolWidget {
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

        # Calculate per-layer memory
        layer_kv_bytes = r.cell_size * r.max_total_num_tokens_padded
        layer_kv_mb = layer_kv_bytes / (1024 * 1024)
        total_kv_gb = (layer_kv_bytes * r.num_hidden_layers) / (1024**3)

        lines = [
            f"[bold]KV Cache Pool[/bold] ({r.attention_type})",
            "",
        ]

        # Show tensor shape based on attention type
        if r.attention_type == "MLA":
            kv_dim = (r.kv_lora_rank or 0) + (r.qk_rope_head_dim or 0)
            shape = f"[{r.max_total_num_tokens_padded:,} × 1 × {kv_dim}]"
            lines.append(f"  Structure:      List of {r.num_hidden_layers} tensors")
            lines.append(f"  Per-layer:      {shape} × {r.kv_cache_dtype_bytes}B")
            lines.append(f"  kv_dim:         kv_lora + rope = {r.kv_lora_rank} + {r.qk_rope_head_dim} = {kv_dim}")
        else:
            shape = f"[{r.max_total_num_tokens_padded:,} × {r.num_kv_heads} × {r.head_dim}]"
            lines.append(f"  Structure:      K + V buffers, {r.num_hidden_layers} layers each")
            lines.append(f"  Per-layer:      {shape} × {r.kv_cache_dtype_bytes}B")

        lines.extend([
            "",
            f"  Token slots:    {r.max_total_num_tokens_padded:,}",
            f"  Cell size:      {r.cell_size:,} bytes/token",
            f"  Per-layer mem:  {layer_kv_mb:.1f} MB",
            f"  Total KV mem:   {total_kv_gb:.2f} GB/GPU",
            f"  Layers:         {r.num_hidden_layers}",
        ])

        # Visual layer stack
        lines.extend([
            "",
            "  [dim]┌───────────────┐[/dim]",
        ])

        num_layers = r.num_hidden_layers
        max_display = 6

        if num_layers <= max_display:
            for i in range(num_layers):
                lines.append(f"  [dim]│[/dim]   Layer {i:<3}    [dim]│[/dim]")
        else:
            show_each = max_display // 2
            for i in range(show_each):
                lines.append(f"  [dim]│[/dim]   Layer {i:<3}    [dim]│[/dim]")
            lines.append(f"  [dim]│[/dim]     ...       [dim]│[/dim]")
            for i in range(num_layers - show_each, num_layers):
                lines.append(f"  [dim]│[/dim]   Layer {i:<3}    [dim]│[/dim]")

        lines.append("  [dim]└───────────────┘[/dim]")

        # Formula
        lines.extend([
            "",
            "  [bold]Cell size formula:[/bold]",
            f"    {r.cell_size_formula}",
            f"    = {r.cell_size_breakdown}",
        ])

        # Show max tokens source if user override
        if r.max_total_num_tokens_arg is not None:
            lines.extend([
                "",
                f"  [yellow]User override:[/yellow]",
                f"    --max-total-tokens = {r.max_total_num_tokens_arg:,}",
                f"    profiled = {r.max_total_num_tokens_profiled:,}",
                f"    using = {r.max_total_num_tokens_padded:,} ({r.max_total_num_tokens_source})",
            ])

        return Text.from_markup("\n".join(lines))
