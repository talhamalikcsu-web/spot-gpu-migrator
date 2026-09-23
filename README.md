# Spot GPU Migrator (SGM) ⚡

Zero-downtime migration daemon and reverse proxy for LLM inference on spot and preemptible cloud GPUs (AWS, GCP, RunPod, Lambda Labs).

## Overview
Spot GPU instances offer **65% to 80% cost savings** over on-demand GPU instances, but cloud providers can reclaim them with short notices (30 to 120 seconds). **Spot GPU Migrator** turns volatile spot instances into resilient production infrastructure by:
1. **Detecting Preemption Instantly**: Watchdogs monitor cloud metadata endpoints (AWS IMDSv2, GCP metadata) every 250ms.
2. **Buffering & Rerouting Ingress**: Reverse proxy seamlessly holds active streaming sockets and transitions upstream targets.
3. **Streaming State Delve**: Live-migrates inference request states, uncommitted token buffers, and KV-cache pointers to warm standby spot instances in < 10 seconds.
4. **Zero Dropped Connections**: End users see continuous streaming responses with zero dropped requests.

## Architecture
```
[ Client / Web App ]
         │
         ▼
[ SGM Ingress Proxy ] ── (Routes traffic, drains dying nodes, switches upstreams)
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
[ Active Spot Node ] ═════════► [ Standby Spot Node ]
 (Watchdog catches                (Receives streamed state,
  preemption signal)               takes over inference)
```

## Directory Structure
- `specs/`: Technical specifications and RFCs (Product Owner)
- `daemon/`: Node daemon, metadata watchdog, memory/state serializer
- `proxy/`: Ingress traffic proxy and seamless socket drainer
- `simulator/`: Cloud metadata and preemption chaos injection harness
- `tests/`: Automated unit, integration, and chaos test suites (QA Engineer)
- `ui/`: Terminal UI (TUI) and real-time telemetry dashboard (UI/UX Developer)

## Team Organization
- **Project Manager**: Main interface with user; orchestrates team deliverables.
- **Product Owner**: Architecture specs, data contracts, and acceptance criteria.
- **Software Developer**: Core daemon, proxy, and networking implementation.
- **QA Engineer**: Automated chaos test suite and zero-loss verification.
- **UI/UX Developer**: Real-time CLI / TUI dashboard and live metrics.
