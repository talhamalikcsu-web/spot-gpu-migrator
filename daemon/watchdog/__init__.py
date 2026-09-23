"""
SGM Watchdog Package.
Provides hypervisor preemption monitors for AWS, GCP, and RunPod.
"""

from daemon.watchdog.base import (
    AbstractPreemptionWatchdog,
    PreemptionCallback,
)
from daemon.watchdog.aws import AWSPreemptionWatchdog
from daemon.watchdog.gcp import GCPPreemptionWatchdog
from daemon.watchdog.runpod import RunPodPreemptionWatchdog

__all__ = [
    "AbstractPreemptionWatchdog",
    "PreemptionCallback",
    "AWSPreemptionWatchdog",
    "GCPPreemptionWatchdog",
    "RunPodPreemptionWatchdog",
]
