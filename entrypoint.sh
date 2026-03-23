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
# Two paths:
# 1. CLAUDE_OAUTH_CREDENTIALS env var (macOS keychain extraction)
# 2. Mounted ~/.claude/ at /opt/claude-creds/ (Linux)
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

# Export OAuth token as ANTHROPIC_API_KEY if not already set
if { [ -z "${ANTHROPIC_API_KEY:-}" ] || [ "$ANTHROPIC_API_KEY" = "" ]; } && [ -f /home/sandbox/.claude/.credentials.json ]; then
    TOKEN=$(python3 -c "import json; d=json.load(open('/home/sandbox/.claude/.credentials.json')); print(d.get('claudeAiOauth',{}).get('accessToken',''))" 2>/dev/null)
    if [ -n "$TOKEN" ]; then
        export ANTHROPIC_API_KEY="$TOKEN"
        echo "[entrypoint] Exported OAuth token as ANTHROPIC_API_KEY"
    fi
fi

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
