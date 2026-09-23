"""
Core module re-exporting NodeDaemon for daemon/core.py compatibility.
"""

from daemon.core.node_daemon import NodeDaemon

__all__ = ["NodeDaemon"]
