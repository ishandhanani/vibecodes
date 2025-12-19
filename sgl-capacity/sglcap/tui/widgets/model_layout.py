"""
Model parallelism layout visualization widget.

Visualizes how TP, DP, and EP are organized across GPUs:
- DP groups contain TP ranks (with enable_dp_attention)
- EP distributes MoE experts across ranks
"""

from textual.widgets import Static
from rich.text import Text

from sglcap.calculator import CapacityReport


class ModelLayoutWidget(Static):
    """
    Visualize GPU layout for TP/DP/EP parallelism.

    With enable_dp_attention:
    - Total GPUs = tp_size
    - DP groups divide TP ranks for attention
    - EP distributes experts (usually ep_size = tp_size)
    """

    DEFAULT_CSS = """
    ModelLayoutWidget {
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
        lines = []

        tp_size = r.tp_size
        dp_size = r.dp_size
        enable_dp_attn = r.enable_dp_attention

        # Calculate layout
        if enable_dp_attn and dp_size > 1:
            total_gpus = tp_size
            gpus_per_dp_group = tp_size // dp_size
        else:
            total_gpus = tp_size * dp_size
            gpus_per_dp_group = tp_size

        lines.extend([
            f"[bold]Model Parallelism Layout[/bold]",
            "",
            f"  TP={tp_size}  DP={dp_size}  DP_Attn={'On' if enable_dp_attn else 'Off'}  "
            f"Total={total_gpus} GPUs",
        ])

        if r.is_moe:
            experts_per_gpu = r.num_experts // tp_size if r.num_experts else 0
            lines.append(f"  MoE: {r.num_experts} experts → ~{experts_per_gpu}/GPU")

        lines.append("")

        # Determine grid layout
        if enable_dp_attn and dp_size > 1:
            # Grid of DP groups, each containing gpus_per_dp_group GPUs
            cols = 4  # 4 DP groups per row
            rows = (dp_size + cols - 1) // cols

            lines.append(f"  [dim]DP Groups ({dp_size} groups × {gpus_per_dp_group} GPU each)[/dim]")
            lines.append("")

            for row in range(rows):
                # Build each line of the grid row
                top_line = "  "
                gpu_line = "  "
                bot_line = "  "

                for col in range(cols):
                    dp_idx = row * cols + col
                    if dp_idx >= dp_size:
                        # Empty cell
                        top_line += "         "
                        gpu_line += "         "
                        bot_line += "         "
                    else:
                        start_gpu = dp_idx * gpus_per_dp_group

                        # Build GPU string for this DP group
                        if gpus_per_dp_group == 1:
                            gpu_str = f"G{start_gpu:<2}"
                            width = 7
                        elif gpus_per_dp_group <= 2:
                            gpus = " ".join(f"G{start_gpu+i}" for i in range(gpus_per_dp_group))
                            gpu_str = gpus
                            width = len(gpu_str) + 4
                        else:
                            # Show range
                            gpu_str = f"G{start_gpu}-G{start_gpu + gpus_per_dp_group - 1}"
                            width = len(gpu_str) + 4

                        top_line += f"┌─D{dp_idx:<2}" + "─" * (width - 5) + "┐ "
                        gpu_line += f"│[green]{gpu_str:^{width-2}}[/green]│ "
                        bot_line += "└" + "─" * (width - 2) + "┘ "

                lines.append(top_line)
                lines.append(gpu_line)
                lines.append(bot_line)

                if row < rows - 1:
                    lines.append("")  # Space between rows
        else:
            # Standard TP layout - grid of individual GPUs
            cols = 8  # 8 GPUs per row
            rows = (total_gpus + cols - 1) // cols

            lines.append(f"  [dim]GPUs ({total_gpus} total, TP={tp_size})[/dim]")
            lines.append("")

            for row in range(rows):
                row_start = row * cols
                row_end = min(row_start + cols, total_gpus)
                actual_cols = row_end - row_start

                # Top border
                top = "  "
                for i in range(actual_cols):
                    top += "┌────┐"
                lines.append(top)

                # GPU labels
                gpus = "  "
                for i in range(row_start, row_end):
                    gpus += f"│[green]G{i:<3}[/green]│"
                lines.append(gpus)

                # Bottom border
                bot = "  "
                for i in range(actual_cols):
                    bot += "└────┘"
                lines.append(bot)

        # Compact legend
        lines.extend([
            "",
            f"  [dim]G#=GPU  D#=DP group[/dim]",
        ])

        if enable_dp_attn and dp_size > 1:
            lines.append(f"  [dim]Attention: independent per DP group | FFN/MoE: all GPUs via TP[/dim]")

        return Text.from_markup("\n".join(lines))
