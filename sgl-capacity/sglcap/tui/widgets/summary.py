"""
Summary widget showing model info and request limits.
"""

from textual.widgets import Static
from rich.text import Text

from sglcap.calculator import CapacityReport


class SummaryWidget(Static):
    """
    Display model info and request limits.
    """

    DEFAULT_CSS = """
    SummaryWidget {
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

        lines = [
            "[bold]Model[/bold]",
            f"  {r.model_path}",
            f"  {r.architecture} | {r.attention_type_detail} | {r.dtype_str}",
        ]

        if r.quantization:
            lines.append(f"  Quantization: {r.quantization}")

        if r.is_moe:
            lines.append(f"  MoE: {r.num_experts} experts, {r.num_experts_per_tok} active/token")

        lines.extend([
            f"  TP={r.tp_size} DP={r.dp_size}",
            "",
        ])

        # Request limits
        token_limit = r.max_total_num_tokens_padded // 2

        lines.extend([
            "[bold]Request Limits[/bold]",
            f"  max_total_tokens:     {r.max_total_num_tokens_padded:,} ({r.max_total_num_tokens_source})",
            f"  max_running_requests: {r.max_running_requests:,} ({r.max_running_requests_source})",
        ])

        if r.max_running_requests_source != "arg":
            lines.append(f"    [dim]= min(token_limit={token_limit:,}, pool_slots={r.req_to_token_pool_size})[/dim]")

        lines.extend([
            f"  max_prefill_tokens:   {r.max_prefill_tokens:,}",
            f"  max_req_len:          {r.max_req_len:,}",
            f"  max_req_input_len:    {r.max_req_input_len:,}",
        ])

        if r.max_queued_requests is not None:
            lines.append(f"  max_queued_requests:  {r.max_queued_requests:,}")

        # Notes
        if r.notes:
            lines.append("")
            lines.append("[bold]Notes[/bold]")
            for note in r.notes:
                # Truncate long notes for display
                if len(note) > 70:
                    note = note[:67] + "..."
                lines.append(f"  [yellow]![/yellow] {note}")

        return Text.from_markup("\n".join(lines))
