"""
Request-to-Token Pool visualization widget.

Visualizes the ReqToTokenPool structure:
- 2D tensor of shape (slots, max_context_len)
- Maps each request slot to its token positions in the KV cache
"""

from textual.widgets import Static
from rich.text import Text

from sglcap.calculator import CapacityReport


class ReqPoolWidget(Static):
    """
    Visualize request-to-token pool structure.

    Shows how requests map to token slots in the KV cache.
    """

    DEFAULT_CSS = """
    ReqPoolWidget {
        width: 100%;
        height: auto;
        padding: 1;
    }
    """

    def __init__(self, report: CapacityReport, **kwargs) -> None:
        super().__init__(**kwargs)
        self.report = report

    def _render_slot_grid(self, slots: int, max_display: int = 8) -> list[str]:
        """Render a visual grid of request slots."""
        lines = []

        # Calculate how many slots to show
        if slots <= max_display:
            display_slots = list(range(slots))
            show_ellipsis = False
        else:
            half = max_display // 2
            display_slots = list(range(half)) + list(range(slots - half, slots))
            show_ellipsis = True

        # Render slots
        slot_strs = []
        for i, slot in enumerate(display_slots):
            if show_ellipsis and i == max_display // 2:
                slot_strs.append("...")
            slot_strs.append(f"[{slot:>4}]")

        # Arrange in rows of 4
        row_size = 4
        for i in range(0, len(slot_strs), row_size):
            row = slot_strs[i:i + row_size]
            lines.append("    " + " ".join(row))

        return lines

    def render(self) -> Text:
        r = self.report

        # Calculate pool memory
        pool_mem_bytes = r.req_to_token_pool_size * r.context_len * 4  # int32
        pool_mem_mb = pool_mem_bytes / (1024 * 1024)

        lines = [
            f"[bold]Request-to-Token Pool[/bold]",
            "",
            f"  Shape: ({r.req_to_token_pool_size:,} slots × {r.context_len:,} context_len) int32",
            f"  Memory: {pool_mem_mb:.1f} MB",
            "",
            f"  [dim]┌─ Request Slots ────────────────────────────┐[/dim]",
        ]

        # Visual representation
        slot_width = min(r.req_to_token_pool_size, 16)
        ctx_height = 3

        # Header row showing slot indices
        header = "  [dim]│[/dim] "
        for i in range(min(slot_width, 8)):
            header += f"[cyan]R{i:<3}[/cyan]"
        if slot_width > 8:
            header += "..."
        lines.append(header)

        # Token position rows (showing the mapping concept)
        for row in range(ctx_height):
            row_str = "  [dim]│[/dim] "
            for col in range(min(slot_width, 8)):
                if row == 0:
                    row_str += "[green]████[/green]"  # Active tokens
                elif row == 1:
                    row_str += "[green]██[/green][dim]░░[/dim]"  # Partial
                else:
                    row_str += "[dim]░░░░[/dim]"  # Empty
            if slot_width > 8:
                row_str += "..."
            lines.append(row_str)

        lines.extend([
            f"  [dim]└────────────────────────────────────────────┘[/dim]",
            "",
            f"  Each slot holds up to {r.context_len:,} token indices",
            "",
            "  [dim]Heuristic:[/dim]",
            f"    slots = clamp(max_tokens / context_len × 512, 2048, 4096)",
            f"          = clamp({r.max_total_num_tokens_padded:,} / {r.context_len:,} × 512, 2048, 4096)",
            f"          = clamp({r.req_to_token_pool_heuristic}, 2048, 4096) = {r.req_to_token_pool_size}",
        ])

        return Text.from_markup("\n".join(lines))
