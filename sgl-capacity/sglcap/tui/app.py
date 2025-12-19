"""
Main TUI application for SGLang Capacity Calculator.

Composes widgets to provide an interactive visualization of:
- GPU memory allocation
- KV cache pool structure
- Request-to-token pool
- Capacity estimates
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, TabbedContent, TabPane

from sglcap.calculator import CapacityReport
from sglcap.tui.widgets import (
    MemoryBarWidget,
    KVPoolWidget,
    ReqPoolWidget,
    SummaryWidget,
    ModelLayoutWidget,
)


class CapacityTUI(App):
    """
    Interactive TUI for visualizing SGLang capacity calculations.

    Uses a tabbed interface to show different aspects:
    - Overview: Memory bar + summary
    - KV Cache: Detailed KV pool structure
    - Request Pool: Request-to-token mapping
    """

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        width: 100%;
        height: 1fr;
        padding: 1 2;
    }

    .panel {
        width: 100%;
        height: auto;
        border: solid $primary;
        margin-bottom: 1;
        padding: 0;
    }

    .panel-title {
        dock: top;
        width: 100%;
        background: $primary;
        color: $text;
        padding: 0 1;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1;
    }

    #overview-left {
        width: 1fr;
        height: auto;
    }

    #overview-right {
        width: 1fr;
        height: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("1", "show_tab('overview')", "Overview"),
        ("2", "show_tab('layout')", "Layout"),
        ("3", "show_tab('req-pool')", "Req Pool"),
        ("4", "show_tab('kv-cache')", "KV Cache"),
    ]

    def __init__(
        self,
        report: CapacityReport,
        num_gpus: int = 8,
        gpu_type: str = "h100",
        gpus_per_node: int = 8,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.report = report
        self.num_gpus = num_gpus
        self.gpu_type = gpu_type
        self.gpus_per_node = gpus_per_node

    def compose(self) -> ComposeResult:
        yield Header()

        with ScrollableContainer(id="main-container"):
            with TabbedContent():
                # Tab 1: Overview
                with TabPane("Overview", id="overview"):
                    yield MemoryBarWidget(self.report)
                    yield SummaryWidget(self.report)

                # Tab 2: Model Layout
                with TabPane("Layout", id="layout"):
                    yield ModelLayoutWidget(self.report)

                # Tab 3: Request Pool
                with TabPane("Req Pool", id="req-pool"):
                    yield ReqPoolWidget(self.report)

                # Tab 4: KV Cache Pool
                with TabPane("KV Cache", id="kv-cache"):
                    yield KVPoolWidget(self.report)

        yield Footer()

    def action_quit(self) -> None:
        self.exit()

    def action_show_tab(self, tab_id: str) -> None:
        """Switch to specified tab."""
        tabbed = self.query_one(TabbedContent)
        tabbed.active = tab_id


def run_tui(
    report: CapacityReport,
    num_gpus: int = 8,
    gpu_type: str = "h100",
    gpus_per_node: int = 8,
) -> None:
    """
    Run the TUI with the given capacity report.

    Args:
        report: Pre-calculated CapacityReport from the calculator
        num_gpus: Total number of GPUs
        gpu_type: GPU type string (e.g., "h100", "l40s")
        gpus_per_node: GPUs per node for layout
    """
    app = CapacityTUI(
        report=report,
        num_gpus=num_gpus,
        gpu_type=gpu_type,
        gpus_per_node=gpus_per_node,
    )
    app.run()
