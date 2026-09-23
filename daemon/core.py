"""
Core module re-exporting NodeDaemon and main for daemon/core.py compatibility.
"""

from daemon.core.node_daemon import NodeDaemon, main

__all__ = ["NodeDaemon", "main"]

if __name__ == "__main__":
    main()
