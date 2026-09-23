"""
Spot GPU Migrator (SGM) UI Module.

Provides Terminal UI (TUI) dashboards and Rich-based visualization telemetry.
"""

from ui.dashboard import (
    SGMDashboard,
    DashboardState,
    NodeTelemetry,
    FinancialTelemetry,
    PreemptionTelemetry,
    ClusterMonitor,
)

__all__ = [
    "SGMDashboard",
    "DashboardState",
    "NodeTelemetry",
    "FinancialTelemetry",
    "PreemptionTelemetry",
    "ClusterMonitor",
]

