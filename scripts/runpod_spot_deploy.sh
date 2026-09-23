#!/usr/bin/env bash
# =============================================================================
# SGM RunPod Spot Automation & Webhook Receiver Script
# Component: scripts/runpod_spot_deploy.sh
# Hardware: RunPod Spot GPU Instances (RTX 4090, A100 SXM, H100 PCIe)
# Role: active
# =============================================================================
set -euo pipefail

echo "[SGM-RUNPOD] Initializing RunPod Spot Worker..."

POD_ID="${RUNPOD_POD_ID:-unknown_pod}"
POD_PUBLIC_IP="${RUNPOD_PUBLIC_IP:-127.0.0.1}"
STANDBY_PEER_IP="${SGM_STANDBY_HOST:-10.0.1.200}"
PROXY_URL="${SGM_PROXY_URL:-http://10.0.1.10:8000}"

echo "[SGM-RUNPOD] Pod ID: ${POD_ID}, Public IP: ${POD_PUBLIC_IP}"

# 1. Validate NVIDIA Drivers
if ! command -v nvidia-smi &> /dev/null; then
  echo "[SGM-RUNPOD] FATAL: nvidia-smi not detected. Ensure GPU is attached."
  exit 1
fi
nvidia-smi

# 2. Launch Local vLLM Engine in Background with Prefix Caching
echo "[SGM-RUNPOD] Launching vLLM Engine on Port 8001 with --enable-prefix-caching..."
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME:-meta-llama/Llama-3-8b-instruct}" \
  --port 8001 \
  --host 0.0.0.0 \
  --enable-prefix-caching \
  --block-size 16 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 \
  --disable-log-requests &
VLLM_PID=$!

VLLM_PORT="${VLLM_PORT:-8001}"

# Wait for vLLM to become healthy
echo "[SGM-RUNPOD] Waiting for vLLM to initialize weights..."
until curl -s "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null; do
  sleep 2
done
echo "[SGM-RUNPOD] vLLM is healthy and ready for inference."

# 3. Trap Signals for Graceful Preemption Handling
cleanup() {
  echo "[SGM-RUNPOD] SIGTERM/SIGINT received! Triggering emergency drain..."
  kill -TERM "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  exit 0
}
trap cleanup SIGTERM SIGINT

# 4. Launch SGM Node Daemon
echo "[SGM-RUNPOD] Launching SGM Node Daemon..."
exec python -m daemon.core \
  --node-id "${POD_ID}" \
  --role "${SGM_ROLE:-active}" \
  --control-port 9001 \
  --p2p-port 9002 \
  --standby-host "${STANDBY_PEER_IP}" \
  --standby-p2p-port 9002 \
  --proxy-url "${PROXY_URL}" \
  --engine-url "http://127.0.0.1:8001" \
  --cloud-provider runpod
