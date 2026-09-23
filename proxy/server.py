"""
Proxy server module re-exporting IngressProxy for proxy/server.py compatibility.
"""

from proxy.ingress import IngressProxy

__all__ = ["IngressProxy"]
