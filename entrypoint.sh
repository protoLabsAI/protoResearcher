#!/bin/bash
# protoResearcher — container entrypoint

echo "[entrypoint] Starting protoResearcher"

# Create dirs inside tmpfs home
mkdir -p /home/sandbox/.nanobot /home/sandbox/.local

# Ensure persistent volume dirs exist
mkdir -p /sandbox/audit /sandbox/knowledge /sandbox/papers

# Copy configs from read-only image
cp /opt/protoresearcher/config/nanobot-config.json /home/sandbox/.nanobot/config.json

# Copy persona into workspace (nanobot reads SOUL.md from workspace)
mkdir -p /sandbox
cp /opt/protoresearcher/config/SOUL.md /sandbox/SOUL.md

# Copy skills into workspace
cp -r /opt/protoresearcher/skills /sandbox/skills

# --- Claude credentials ---
mkdir -p /home/sandbox/.claude

if [ -n "$CLAUDE_OAUTH_CREDENTIALS" ]; then
    echo "$CLAUDE_OAUTH_CREDENTIALS" > /home/sandbox/.claude/.credentials.json
    chmod 600 /home/sandbox/.claude/.credentials.json
    echo "[entrypoint] Claude credentials loaded from env var"
elif [ -f /opt/claude-creds/.credentials.json ]; then
    cp /opt/claude-creds/.credentials.json /home/sandbox/.claude/.credentials.json
    chmod 600 /home/sandbox/.claude/.credentials.json
    echo "[entrypoint] Claude credentials loaded from mounted volume"
fi

# --- CLIProxyAPI — OpenAI-compatible proxy for Claude OAuth ---
mkdir -p /opt/.cliproxy
cp /opt/protoresearcher/config/cliproxy-config.yaml /opt/.cliproxy/config.yaml

# Inject OAuth token into CLIProxyAPI config
if [ -f /home/sandbox/.claude/.credentials.json ]; then
    python3 -c "
import json
import yaml

with open('/home/sandbox/.claude/.credentials.json') as f:
    creds = json.load(f)
token = creds.get('claudeAiOauth', {}).get('accessToken', '')

with open('/opt/.cliproxy/config.yaml') as f:
    cfg = yaml.safe_load(f)

if token:
    cfg['claude-api-key'] = [{'api-key': token}]
    with open('/opt/.cliproxy/config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print('[entrypoint] Injected Claude OAuth token into CLIProxyAPI config')
else:
    print('[entrypoint] No OAuth token found for CLIProxyAPI')
" 2>/dev/null
fi

cli-proxy-api --config /opt/.cliproxy/config.yaml &
echo "[entrypoint] CLIProxyAPI started on port 8317"

# Lab mode setup (if GPU available)
if [ -n "${LAB_GPU}" ] || command -v nvidia-smi &>/dev/null; then
    echo "[entrypoint] GPU detected — lab mode available (/lab on)"
    mkdir -p /sandbox/lab
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    fi
fi

# Start protoResearcher Gradio UI on port 7870
exec python /opt/protoresearcher/server.py --config /home/sandbox/.nanobot/config.json
