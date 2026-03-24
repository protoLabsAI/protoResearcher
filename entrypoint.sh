#!/bin/bash
# protoResearcher — container entrypoint

echo "[entrypoint] Starting protoResearcher"

# Create dirs inside tmpfs home
mkdir -p /home/sandbox/.nanobot /home/sandbox/.local

# Symlink persistent cron data into nanobot's expected location
if [ -d /opt/.cron ]; then
    ln -sf /opt/.cron /home/sandbox/.nanobot/cron
fi

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

# Function to inject token into CLIProxyAPI config
inject_token() {
    python3 -c "
import json
import yaml

with open('/opt/claude-creds/.credentials.json') as f:
    creds = json.load(f)
token = creds.get('claudeAiOauth', {}).get('accessToken', '')

with open('/opt/.cliproxy/config.yaml') as f:
    cfg = yaml.safe_load(f)

old_token = ''
if cfg.get('claude-api-key'):
    old_token = cfg['claude-api-key'][0].get('api-key', '')

if token and token != old_token:
    cfg['claude-api-key'] = [{'api-key': token}]
    with open('/opt/.cliproxy/config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print('[token-refresh] New OAuth token injected')
" 2>/dev/null
}

# Initial token injection
inject_token

cli-proxy-api --config /opt/.cliproxy/config.yaml &
echo "[entrypoint] CLIProxyAPI started on port 8317"

# Set env vars for litellm to route through CLIProxyAPI
export OPENAI_API_KEY="protoresearcher-internal"
export OPENAI_API_BASE="http://127.0.0.1:8317/v1"

# --- Token refresh loop ---
# Watches the mounted ~/.claude credentials for changes.
# CLIProxyAPI has a file watcher that auto-reloads config on change,
# so we just need to update the config file when the token changes.
(
    LAST_MTIME=0
    while true; do
        sleep 300  # Check every 5 minutes
        if [ -f /opt/claude-creds/.credentials.json ]; then
            CURRENT_MTIME=$(stat -c %Y /opt/claude-creds/.credentials.json 2>/dev/null || echo 0)
            if [ "$CURRENT_MTIME" != "$LAST_MTIME" ]; then
                LAST_MTIME=$CURRENT_MTIME
                inject_token
            fi
        fi
    done
) &
echo "[entrypoint] Token refresh watcher started (every 5m)"

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
