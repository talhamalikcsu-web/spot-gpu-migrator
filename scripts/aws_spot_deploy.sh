#!/usr/bin/env bash
# =============================================================================
# SGM AWS EC2 Spot Auto-Recovery & Provisioning Script
# Component: scripts/aws_spot_deploy.sh
# Target OS: Ubuntu 22.04 LTS / 24.04 LTS (x86_64)
# Hardware: AWS EC2 Spot Instances (g5.xlarge, g6.xlarge, p4d.24xlarge)
# Role: active
# =============================================================================
set -euo pipefail
IFS=$'\n\t'

echo "[SGM-INIT] Starting Spot GPU Migrator node initialization..."

# -----------------------------------------------------------------------------
# 1. IMDSv2 Security Verification
# -----------------------------------------------------------------------------
echo "[SGM-INIT] Verifying IMDSv2 token acquisition..."
IMDS_TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)

if [ -z "$IMDS_TOKEN" ]; then
  echo "[SGM-INIT] ERROR: Unable to acquire IMDSv2 token. Ensure HttpEndpoint=enabled and HttpTokens=required."
  echo "[SGM-INIT] Ensure http-put-response-hop-limit 2 is configured for containerized access."
  exit 1
fi

INSTANCE_ID=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_TYPE=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type)
AVAILABILITY_ZONE=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/availability-zone)

echo "[SGM-INIT] Detected Instance: ID=${INSTANCE_ID}, Type=${INSTANCE_TYPE}, AZ=${AVAILABILITY_ZONE}"

# -----------------------------------------------------------------------------
# 2. System Packages & Docker Installation
# -----------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  pciutils \
  jq

# Install Docker CE if not installed
if ! command -v docker &> /dev/null; then
  echo "[SGM-INIT] Installing Docker CE..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# -----------------------------------------------------------------------------
# 3. NVIDIA Driver & NVIDIA Container Toolkit Setup
# -----------------------------------------------------------------------------
if ! command -v nvidia-smi &> /dev/null; then
  echo "[SGM-INIT] Installing NVIDIA Drivers (nvidia-driver-550-server)..."
  apt-get install -y nvidia-driver-550-server
fi

if ! command -v nvidia-ctk &> /dev/null; then
  echo "[SGM-INIT] Configuring NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

echo "[SGM-INIT] GPU Hardware Topology:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# -----------------------------------------------------------------------------
# 4. SGM Daemon Systemd Service Configuration
# -----------------------------------------------------------------------------
cat << 'EOF' > /etc/systemd/system/sgm-daemon.service
[Unit]
Description=Spot GPU Migrator (SGM) Node Supervisor Daemon
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
Restart=always
RestartSec=5s
EnvironmentFile=-/etc/sgm/environment
ExecStart=/usr/bin/docker run --rm --name sgm-node-daemon \
  --net=host \
  --ipc=host \
  -e SGM_NODE_ID=${SGM_NODE_ID} \
  -e SGM_ROLE=${SGM_ROLE} \
  -e SGM_CONTROL_PORT=${SGM_CONTROL_PORT:-9001} \
  -e SGM_P2P_PORT=${SGM_P2P_PORT:-9002} \
  -e SGM_STANDBY_HOST=${SGM_STANDBY_HOST} \
  -e SGM_STANDBY_P2P_PORT=${SGM_STANDBY_P2P_PORT:-9002} \
  -e SGM_PROXY_URL=${SGM_PROXY_URL} \
  -e SGM_ENGINE_URL=${SGM_ENGINE_URL:-http://127.0.0.1:8001} \
  -e SGM_CLOUD_PROVIDER=aws \
  sgm-daemon:latest

ExecStop=/usr/bin/docker stop -t 30 sgm-node-daemon

[Install]
WantedBy=multi-user.target
EOF

mkdir -p /etc/sgm
cat << EOF > /etc/sgm/environment
SGM_NODE_ID=${INSTANCE_ID}
SGM_ROLE=${SGM_NODE_ROLE:-active}
SGM_CONTROL_PORT=9001
SGM_P2P_PORT=9002
SGM_STANDBY_HOST=${STANDBY_PEER_IP:-10.0.1.200}
SGM_STANDBY_P2P_PORT=9002
SGM_PROXY_URL=${PROXY_INGRESS_URL:-http://10.0.1.10:8000}
SGM_ENGINE_URL=http://127.0.0.1:8001
EOF

systemctl daemon-reload
systemctl enable --now sgm-daemon.service || true
echo "[SGM-INIT] Spot provisioning complete. SGM daemon configured."
