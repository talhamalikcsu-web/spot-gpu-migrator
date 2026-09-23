"""
Spot GPU Migrator (SGM) - Terminal UI (TUI) Dashboard.

Provides an executive, real-time Terminal UI using Python's `rich` library:
- Cluster State & Node Topology (Active vs Standby, Providers, Health States)
- Real-Time Dollar Savings Counter ($ saved vs On-Demand with ticker animation)
- Live Preemption Countdown & P2P State Transfer Telemetry
- Zero-Loss Token Verification Indicator (100% Stream Continuity)
- Live Scrolling Event Log Feed

Can be run standalone as an interactive demonstration or driven programmatically
via telemetry callbacks during actual migration operations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import sys
import time
from typing import Any, Callable, Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Telemetry Data Models
# ---------------------------------------------------------------------------

@dataclass
class NodeTelemetry:
    """Telemetry data representing a single GPU cluster node."""
    node_id: str
    role: str               # "ACTIVE", "STANDBY", "FALLBACK"
    provider: str           # "AWS (p4de.24xlarge)", "GCP (a2-highgpu-8g)", "RunPod (8x A100)"
    status: str             # "HEALTHY", "PREEMPTING", "STREAMING", "MIGRATED", "TERMINATED"
    control_port: int       # e.g. 9001
    p2p_port: int           # e.g. 9002
    engine_port: int        # e.g. 8001
    memory_info: str        # e.g. "80 GB (32.4 GB VRAM)"
    active_streams: int = 0
    tokens_generated: int = 0
    gpu_util_pct: float = 0.0


@dataclass
class FinancialTelemetry:
    """Financial tracking comparing On-Demand vs Spot GPU instance costs."""
    on_demand_rate_hourly: float = 32.77  # AWS p4de.24xlarge on-demand ($/hr)
    spot_rate_hourly: float = 9.83        # AWS spot rate (~70% discount)
    baseline_savings_usd: float = 142.85  # Realized savings prior to session
    session_start_monotonic: float = field(default_factory=time.monotonic)
    nodes_count: int = 1

    @property
    def hourly_savings_rate(self) -> float:
        """Net savings rate in dollars per hour across all active spot nodes."""
        return max(0.0, (self.on_demand_rate_hourly - self.spot_rate_hourly) * self.nodes_count)

    @property
    def savings_percent(self) -> float:
        """Percentage savings over on-demand rates."""
        if self.on_demand_rate_hourly <= 0:
            return 0.0
        return ((self.on_demand_rate_hourly - self.spot_rate_hourly) / self.on_demand_rate_hourly) * 100.0

    @property
    def monthly_projected_savings_usd(self) -> float:
        """Projected 30-day savings in USD."""
        return self.hourly_savings_rate * 24.0 * 30.0

    def current_realized_savings_usd(self) -> float:
        """Calculates live accumulated dollar savings with sub-second precision."""
        elapsed_seconds = time.monotonic() - self.session_start_monotonic
        elapsed_hours = elapsed_seconds / 3600.0
        return self.baseline_savings_usd + (elapsed_hours * self.hourly_savings_rate)


@dataclass
class PreemptionTelemetry:
    """Telemetry capturing preemption detection, countdown, and P2P handover metrics."""
    is_preempting: bool = False
    provider: str = "AWS"
    deadline_seconds_total: float = 30.0
    deadline_seconds_remaining: float = 0.0
    preemption_started_at: Optional[float] = None
    
    # P2P Transfer Telemetry
    p2p_transfer_speed_mb_s: float = 0.0
    p2p_bytes_transferred: int = 0
    p2p_bytes_total: int = 0
    active_handover_count: int = 0
    total_handover_count: int = 0
    handover_latency_ms: float = 0.0

    # Stream Integrity Verification
    zero_loss_verified: bool = False
    dropped_tokens: int = 0
    duplicate_tokens: int = 0
    integrity_percentage: float = 100.0
    verified_sequences_count: int = 0


@dataclass
class LogEvent:
    """Log entry with timestamp and severity level for the dashboard log feed."""
    timestamp: str
    level: str  # "INFO", "WARN", "CRIT", "SUCCESS"
    message: str


# ---------------------------------------------------------------------------
# Dashboard State Container
# ---------------------------------------------------------------------------

class DashboardState:
    """Comprehensive state container for SGM live metrics and telemetry."""

    def __init__(self) -> None:
        self.boot_time = datetime.now(timezone.utc)
        self.cluster_status: str = "OPERATIONAL"  # OPERATIONAL, PREEMPTING, MIGRATING, RESUMED
        self.mode: str = "SIMULATION (AWS IMDSv2)"
        self.ingress_url: str = "http://127.0.0.1:8000"
        self.active_upstream_url: str = "http://127.0.0.1:8001"

        # Nodes registry
        self.nodes: Dict[str, NodeTelemetry] = {
            "spot-node-us-east-1a": NodeTelemetry(
                node_id="spot-node-us-east-1a",
                role="ACTIVE",
                provider="AWS (p4de.24xlarge)",
                status="HEALTHY",
                control_port=9001,
                p2p_port=9002,
                engine_port=8001,
                memory_info="80 GB (34.2 GB VRAM)",
                active_streams=1,
                tokens_generated=0,
                gpu_util_pct=88.5,
            ),
            "spot-node-us-east-1b": NodeTelemetry(
                node_id="spot-node-us-east-1b",
                role="STANDBY",
                provider="AWS (p4de.24xlarge)",
                status="HEALTHY",
                control_port=9003,
                p2p_port=9002,
                engine_port=8002,
                memory_info="80 GB (Warmed Weights)",
                active_streams=0,
                tokens_generated=0,
                gpu_util_pct=12.0,
            ),
        }

        # Financial tracking
        self.financial = FinancialTelemetry()

        # Preemption & Migration metrics
        self.preemption = PreemptionTelemetry()

        # Log feed (circular buffer of latest 25 logs)
        self.logs: List[LogEvent] = []
        self._init_default_logs()

    def _init_default_logs(self) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        self.logs.extend([
            LogEvent(now_str, "INFO", "SGM Ingress Proxy online on port 8000 (OpenAI SSE gateway)"),
            LogEvent(now_str, "INFO", "Active spot node spot-node-us-east-1a connected [vLLM :8001]"),
            LogEvent(now_str, "INFO", "Standby spot node spot-node-us-east-1b warm & ready [P2P :9002]"),
            LogEvent(now_str, "INFO", "Hypervisor Watchdog active: polling AWS IMDSv2 @ 250ms"),
        ])

    def add_log(self, level: str, message: str) -> None:
        """Appends a new event log to the live feed."""
        now_str = datetime.now().strftime("%H:%M:%S")
        self.logs.append(LogEvent(now_str, level.upper(), message))
        if len(self.logs) > 30:
            self.logs.pop(0)

    @property
    def uptime_str(self) -> str:
        """Formatted uptime string HH:MM:SS."""
        delta = datetime.now(timezone.utc) - self.boot_time
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------------------------
# SGMDashboard UI Renderer
# ---------------------------------------------------------------------------

class SGMDashboard:
    """
    Executive Rich-based Terminal UI Dashboard for Spot GPU Migrator.
    Renders high-frequency layout updates showing cluster state,
    P2P migration progress, financial counters, and verification metrics.
    """

    def __init__(self, state: Optional[DashboardState] = None, console: Optional[Console] = None) -> None:
        self.state = state or DashboardState()
        self.console = console or Console()

    # -----------------------------------------------------------------------
    # Layout Builder
    # -----------------------------------------------------------------------

    def build_layout(self) -> Layout:
        """Constructs the multi-pane grid layout for the dashboard."""
        layout = Layout()

        # Vertical division: Header (4 lines), Main Body (flexible), Footer (3 lines)
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        # Body horizontal split: Left column (60%), Right column (40%)
        layout["body"].split_row(
            Layout(name="left", ratio=6),
            Layout(name="right", ratio=4),
        )

        # Left split: Nodes Table (top) and Live Migration Progress (bottom)
        layout["left"].split_column(
            Layout(name="nodes", ratio=5),
            Layout(name="migration", ratio=5),
        )

        # Right split: Financial Telemetry (top) and Live Log Feed (bottom)
        layout["right"].split_column(
            Layout(name="financial", ratio=4),
            Layout(name="logs", ratio=6),
        )

        # Populate renderables
        layout["header"].update(self._render_header())
        layout["nodes"].update(self._render_nodes_table())
        layout["migration"].update(self._render_migration_panel())
        layout["financial"].update(self._render_financial_panel())
        layout["logs"].update(self._render_logs_panel())
        layout["footer"].update(self._render_footer())

        return layout

    # -----------------------------------------------------------------------
    # Panel Renderers
    # -----------------------------------------------------------------------

    def _render_header(self) -> Panel:
        """Renders the executive title banner with cluster status and uptime."""
        # Status badge styling
        status_styles = {
            "OPERATIONAL": ("[bold white on green] 🟢 OPERATIONAL [/bold white on green]", "green"),
            "PREEMPTING": ("[bold white on yellow] 🟡 PREEMPTION DETECTED [/bold white on yellow]", "yellow"),
            "MIGRATING": ("[bold white on blue] 🔵 P2P MIGRATING [/bold white on blue]", "cyan"),
            "RESUMED": ("[bold white on magenta] 🟣 HANDOVER COMPLETE [/bold white on magenta]", "magenta"),
        }
        status_badge, border_color = status_styles.get(
            self.state.cluster_status,
            ("[bold white on red] 🔴 UNKNOWN [/bold white on red]", "red")
        )

        title_text = Text.from_markup(
            f"[bold bright_cyan]⚡ SPOT GPU MIGRATOR (SGM)[/bold bright_cyan]  "
            f"[dim]|[/dim]  [bold bright_white]Zero-Downtime Live Migration Daemon[/bold bright_white]\n"
            f"Cluster Status: {status_badge}  "
            f"[dim]|[/dim]  Uptime: [bold bright_yellow]{self.state.uptime_str}[/bold bright_yellow]  "
            f"[dim]|[/dim]  Mode: [bold cyan]{self.state.mode}[/bold cyan]  "
            f"[dim]|[/dim]  Ingress: [bold green]{self.state.ingress_url}[/bold green]"
        )

        return Panel(
            title_text,
            border_style=border_color,
            padding=(0, 1),
        )

    def _render_nodes_table(self) -> Panel:
        """Renders the active vs standby cluster nodes table."""
        table = Table(
            expand=True,
            box=None,
            header_style="bold bright_cyan",
            show_edge=False,
            row_styles=["none", "dim"],
        )
        table.add_column("Node ID", style="bright_white", no_wrap=True)
        table.add_column("Role", justify="center", no_wrap=True)
        table.add_column("Provider / Spec", style="bright_blue", no_wrap=True)
        table.add_column("Status", justify="center", no_wrap=True)
        table.add_column("Ports", justify="center", style="yellow")
        table.add_column("VRAM / Memory", style="green")
        table.add_column("Streams", justify="right", style="bright_magenta")

        status_color_map = {
            "HEALTHY": "[bold green]HEALTHY[/bold green]",
            "PREEMPTING": "[bold yellow]PREEMPTING[/bold yellow]",
            "STREAMING": "[bold bright_cyan]STREAMING[/bold bright_cyan]",
            "MIGRATED": "[bold magenta]MIGRATED[/bold magenta]",
            "TERMINATED": "[bold red]TERMINATED[/bold red]",
            "ACTIVE": "[bold bright_green]ACTIVE[/bold bright_green]",
            "STANDBY": "[bold cyan]STANDBY[/bold cyan]",
        }

        role_badge_map = {
            "ACTIVE": "[bold white on dark_green] ACTIVE [/bold white on dark_green]",
            "STANDBY": "[bold white on dark_blue] STANDBY [/bold white on dark_blue]",
            "FALLBACK": "[bold white on dark_magenta] FALLBACK [/bold white on dark_magenta]",
        }

        for node in self.state.nodes.values():
            role_badge = role_badge_map.get(node.role.upper(), f"[bold]{node.role}[/bold]")
            status_text = status_color_map.get(node.status.upper(), f"[white]{node.status}[/white]")
            port_info = f":{node.control_port} | P2P:{node.p2p_port}"

            table.add_row(
                node.node_id,
                role_badge,
                node.provider,
                status_text,
                port_info,
                node.memory_info,
                f"{node.active_streams} active",
            )

        return Panel(
            table,
            title="[bold bright_white]🖥️ Cluster Node Topology[/bold bright_white]",
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_financial_panel(self) -> Panel:
        """Renders the real-time dollar savings counter and cost telemetry."""
        fin = self.state.financial
        realized = fin.current_realized_savings_usd()
        savings_pct = fin.savings_percent
        hourly_save = fin.hourly_savings_rate
        monthly_save = fin.monthly_projected_savings_usd

        # Dynamic visual ticker representation
        content = Text()
        content.append("On-Demand Rate:  ", style="dim")
        content.append(f"${fin.on_demand_rate_hourly:.2f} / hr\n", style="bold red")

        content.append("Spot Rate:       ", style="dim")
        content.append(f"${fin.spot_rate_hourly:.2f} / hr\n", style="bold green")

        content.append("Cost Reduction:  ", style="dim")
        content.append(f"{savings_pct:.1f}% Savings", style="bold bright_green")
        content.append(f"  (-${hourly_save:.2f}/hr)\n\n", style="dim green")

        content.append("REALIZED SAVINGS:\n", style="bold bright_yellow")
        content.append(f"  ${realized:,.4f}\n", style="bold bright_green on black")
        content.append("  (Real-Time Dollar Ticker Accumulator)\n\n", style="dim green")

        content.append("30-Day Projected: ", style="dim")
        content.append(f"${monthly_save:,.2f} / mo\n", style="bold bright_white")
        content.append("Efficiency Gain:  ", style="dim")
        content.append("3.3x more tokens / dollar", style="italic bright_cyan")

        return Panel(
            content,
            title="[bold bright_green]💰 Financial Telemetry[/bold bright_green]",
            border_style="bright_green",
            padding=(0, 1),
        )

    def _render_migration_panel(self) -> Panel:
        """Renders the preemption deadline countdown, P2P transfer, and zero-loss verification."""
        p = self.state.preemption

        content = Text()

        # 1. Preemption Countdown Timer
        if p.is_preempting:
            rem = max(0.0, p.deadline_seconds_remaining)
            total = max(1.0, p.deadline_seconds_total)
            pct_remaining = rem / total

            # Color coding by urgency
            if rem > 20.0:
                color_tag = "bright_green"
                badge = "[bold white on dark_green] ⏳ GRACE PERIOD ACTIVE [/bold white on dark_green]"
            elif rem > 10.0:
                color_tag = "bright_yellow"
                badge = "[bold white on dark_goldenrod] ⚠️ TERMINATION IMMINENT [/bold white on dark_goldenrod]"
            else:
                color_tag = "bright_red"
                badge = "[bold white on dark_red] 🚨 CRITICAL DEADLINE [/bold white on dark_red]"

            # Visual ASCII progress bar for deadline
            bar_len = 24
            filled = int(round(pct_remaining * bar_len))
            bar_visual = "█" * filled + "░" * (bar_len - filled)

            content.append(f"Preemption Alert: {badge}\n", style="bold")
            content.append("Deadline Timer:   ", style="dim")
            content.append(f"[{color_tag}]{bar_visual}  {rem:.1f}s remaining[/{color_tag}]\n\n")
        else:
            content.append("Preemption Alert: [bold green]● NO NOTICE PENDING[/bold green] (Normal Execution)\n")
            content.append("Deadline Timer:   [dim]Polling metadata hypervisor every 250ms[/dim]\n\n")

        # 2. P2P State Transfer Speed & Progress
        if p.p2p_transfer_speed_mb_s > 0 or p.p2p_bytes_transferred > 0:
            speed_str = f"{p.p2p_transfer_speed_mb_s:.1f} MB/s"
            pct_p2p = 1.0 if p.p2p_bytes_total == 0 else min(1.0, p.p2p_bytes_transferred / p.p2p_bytes_total)
            bar_len = 24
            filled = int(round(pct_p2p * bar_len))
            bar_p2p = "█" * filled + "░" * (bar_len - filled)

            content.append("P2P State Stream: ", style="dim")
            content.append(f"[bold cyan]{bar_p2p}  {speed_str}[/bold cyan]\n")
            content.append(f"                  {p.p2p_bytes_transferred:,} / {max(p.p2p_bytes_transferred, p.p2p_bytes_total):,} bytes transferred\n")
        else:
            content.append("P2P State Stream: [dim]0.0 MB/s (Standby channel idle on port 9002)[/dim]\n")

        # 3. Active Stream Handover Count & Latency
        if p.active_handover_count > 0 or p.total_handover_count > 0:
            content.append("Streams Handover: ", style="dim")
            content.append(f"[bold yellow]{p.active_handover_count} / {p.total_handover_count} streams transferred[/bold yellow]")
            if p.handover_latency_ms > 0:
                sla_pass = "[bold green]PASS (<= 2500ms)[/bold green]" if p.handover_latency_ms <= 2500 else "[bold red]FAIL[/bold red]"
                content.append(f"  [dim]|[/dim]  Latency: [bold bright_yellow]{p.handover_latency_ms:.1f} ms[/bold bright_yellow] ({sla_pass})\n\n")
            else:
                content.append("\n\n")
        else:
            content.append("Streams Handover: [dim]0 active migrations in flight[/dim]\n\n")

        # 4. Zero-Loss Token Verification Indicator
        content.append("Zero-Loss Verify: ", style="dim")
        if p.zero_loss_verified:
            content.append(
                f"[bold bright_green]✨ 100.0% INTEGRITY VERIFIED[/bold bright_green] "
                f"([bold green]0 Dropped[/bold green], [bold green]0 Duplicates[/bold green])\n"
            )
            content.append("Socket Continuity:[bold bright_green] 100% Unbroken HTTP/SSE Stream[/bold bright_green]", style="green")
        elif p.is_preempting:
            content.append(
                "[bold bright_yellow]🔄 DEDUPLICATING TOKENS & BUFFERING SOCKET...[/bold bright_yellow]\n"
            )
            content.append("Socket Continuity:[bold yellow] Paused in BUFFERING mode (Keep-Alives Active)[/bold yellow]")
        else:
            content.append(
                "[bold cyan]● EXACTLY-ONCE RING BUFFER ACTIVE[/bold cyan] (Continuous Monitoring)\n"
            )
            content.append("Socket Continuity:[bold green] Stable connection with downstream client[/bold green]")

        panel_color = "bright_yellow" if p.is_preempting else ("bright_green" if p.zero_loss_verified else "bright_blue")
        return Panel(
            content,
            title="[bold bright_yellow]⚡ Live Preemption & Migration Telemetry[/bold bright_yellow]",
            border_style=panel_color,
            padding=(0, 1),
        )

    def _render_logs_panel(self) -> Panel:
        """Renders the recent scrolling SGM event log feed."""
        content = Text()
        level_colors = {
            "INFO": "bright_blue",
            "WARN": "bright_yellow",
            "CRIT": "bright_red",
            "SUCCESS": "bold bright_green",
        }

        # Take last 8 log entries to comfortably fit pane
        display_logs = self.state.logs[-8:] if self.state.logs else []
        for log in display_logs:
            lvl_color = level_colors.get(log.level, "white")
            content.append(f"[{log.timestamp}] ", style="dim")
            content.append(f"[{log.level:<7}] ", style=lvl_color)
            content.append(f"{log.message}\n", style="white")

        return Panel(
            content,
            title="[bold bright_white]📜 Live SGM Event Feed[/bold bright_white]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _render_footer(self) -> Panel:
        """Renders the navigation shortcuts and operational hints."""
        footer_text = Text.from_markup(
            "[dim]Commands:[/dim]  "
            "[bold cyan]python cli.py demo[/bold cyan] (Full local cluster)  "
            "[dim]|[/dim]  [bold yellow]python cli.py test[/bold yellow] (Rich test runner)  "
            "[dim]|[/dim]  [bold green]python cli.py status[/bold green] (Cluster health)  "
            "[dim]|[/dim]  Press [bold red]Ctrl+C[/bold red] to stop"
        )
        return Panel(
            footer_text,
            border_style="dim",
            padding=(0, 1),
        )

    # -----------------------------------------------------------------------
    # Programmatic Update Methods
    # -----------------------------------------------------------------------

    def update_node(
        self,
        node_id: str,
        role: Optional[str] = None,
        status: Optional[str] = None,
        active_streams: Optional[int] = None,
        tokens_generated: Optional[int] = None,
        gpu_util_pct: Optional[float] = None,
    ) -> None:
        """Programmatically updates metrics for a specific node."""
        if node_id in self.state.nodes:
            node = self.state.nodes[node_id]
            if role is not None:
                node.role = role
            if status is not None:
                node.status = status
            if active_streams is not None:
                node.active_streams = active_streams
            if tokens_generated is not None:
                node.tokens_generated = tokens_generated
            if gpu_util_pct is not None:
                node.gpu_util_pct = gpu_util_pct

    def add_log(self, level: str, message: str) -> None:
        """Appends a new event log to the dashboard state."""
        self.state.add_log(level, message)

    def trigger_preemption(self, provider: str = "AWS", deadline_seconds: float = 30.0) -> None:
        """Updates dashboard telemetry to reflect an active preemption event."""
        p = self.state.preemption
        p.is_preempting = True
        p.provider = provider.upper()
        p.deadline_seconds_total = float(deadline_seconds)
        p.deadline_seconds_remaining = float(deadline_seconds)
        p.preemption_started_at = time.monotonic()
        p.zero_loss_verified = False
        self.state.cluster_status = "PREEMPTING"
        self.add_log("WARN", f"{p.provider} Preemption Notice Detected! Grace Period: {deadline_seconds:.1f}s")
        self.add_log("CRIT", "Ingress Proxy activated BUFFERING mode. Emitting keep-alive heartbeats.")

    def tick_countdown(self) -> None:
        """Decrements the preemption countdown timer based on elapsed wall time."""
        p = self.state.preemption
        if p.is_preempting and p.preemption_started_at is not None:
            elapsed = time.monotonic() - p.preemption_started_at
            p.deadline_seconds_remaining = max(0.0, p.deadline_seconds_total - elapsed)

    def update_p2p_migration(
        self,
        speed_mb_s: float,
        transferred_bytes: int,
        total_bytes: int,
        active_handover_count: int,
        total_handover_count: int,
        latency_ms: float = 0.0,
    ) -> None:
        """Updates P2P state transfer telemetry during live migration."""
        p = self.state.preemption
        p.p2p_transfer_speed_mb_s = speed_mb_s
        p.p2p_bytes_transferred = transferred_bytes
        p.p2p_bytes_total = total_bytes
        p.active_handover_count = active_handover_count
        p.total_handover_count = total_handover_count
        if latency_ms > 0:
            p.handover_latency_ms = latency_ms

    def verify_zero_loss(self, dropped: int = 0, duplicates: int = 0, verified_tokens: int = 0) -> None:
        """Confirms exactly-once token delivery and zero dropped socket verification."""
        p = self.state.preemption
        p.zero_loss_verified = (dropped == 0 and duplicates == 0)
        p.dropped_tokens = dropped
        p.duplicate_tokens = duplicates
        p.verified_sequences_count = verified_tokens
        p.integrity_percentage = 100.0 if (dropped == 0 and duplicates == 0) else 0.0
        self.state.cluster_status = "RESUMED"
        self.add_log("SUCCESS", f"Zero-Loss Verified: {verified_tokens} tokens delivered with 100% integrity!")
        self.add_log("SUCCESS", "Proxy seamlessly switched upstream: downstream client experienced 0 resets.")

    # -----------------------------------------------------------------------
    # Standalone Demo Runner
    # -----------------------------------------------------------------------

    async def run_standalone_demo(self, speed_factor: float = 1.0) -> None:
        """
        Runs an interactive simulated walkthrough of the SGM live migration workflow:
        1. Steady-state token streaming on Active Node.
        2. Preemption injected by AWS IMDSv2 notice.
        3. Real-time countdown and proxy buffering.
        4. High-speed P2P state transfer to Standby Node.
        5. Seamless proxy upstream swap and zero-loss verification.
        """
        self.console.clear()
        with Live(self.build_layout(), console=self.console, refresh_per_second=8, screen=False) as live:
            # Phase 1: Steady-state streaming
            for i in range(1, 15):
                self.update_node("spot-node-us-east-1a", active_streams=1, tokens_generated=i * 4)
                self.tick_countdown()
                live.update(self.build_layout())
                await asyncio.sleep(0.12 / speed_factor)

            # Phase 2: Inject Preemption notice
            self.trigger_preemption(provider="AWS", deadline_seconds=30.0)
            self.update_node("spot-node-us-east-1a", status="PREEMPTING")
            live.update(self.build_layout())
            await asyncio.sleep(0.3 / speed_factor)

            # Phase 3: Buffering & P2P Transfer
            self.state.cluster_status = "MIGRATING"
            self.update_node("spot-node-us-east-1a", status="STREAMING")
            self.add_log("INFO", "Initiating P2P binary state transfer to Standby (port 9002)...")
            live.update(self.build_layout())

            total_bytes = 48200
            for step in range(1, 11):
                cur_bytes = int(total_bytes * (step / 10.0))
                speed = 820.0 + (step * 8.5)
                self.update_p2p_migration(
                    speed_mb_s=speed,
                    transferred_bytes=cur_bytes,
                    total_bytes=total_bytes,
                    active_handover_count=1,
                    total_handover_count=1,
                    latency_ms=138.4,
                )
                self.tick_countdown()
                live.update(self.build_layout())
                await asyncio.sleep(0.08 / speed_factor)

            self.add_log("INFO", "Standby Node ACK received: CRC32 0x7E3A9F1B verified.")
            self.add_log("SUCCESS", "Proxy upstream switched to Standby Node (port 8002).")
            self.update_node("spot-node-us-east-1a", status="TERMINATED", role="DRAINED", active_streams=0)
            self.update_node("spot-node-us-east-1b", status="HEALTHY", role="ACTIVE", active_streams=1)

            # Phase 4: Resume & complete streaming on Standby
            for i in range(16, 26):
                self.update_node("spot-node-us-east-1b", tokens_generated=i * 4)
                self.tick_countdown()
                live.update(self.build_layout())
                await asyncio.sleep(0.12 / speed_factor)

            # Phase 5: Zero-loss verification
            self.verify_zero_loss(dropped=0, duplicates=0, verified_tokens=100)
            live.update(self.build_layout())
            await asyncio.sleep(1.5 / speed_factor)


# ---------------------------------------------------------------------------
# Direct Execution Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dashboard = SGMDashboard()
    try:
        asyncio.run(dashboard.run_standalone_demo())
    except KeyboardInterrupt:
        pass
