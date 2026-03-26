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

# Copy configs from read-only image, expanding env vars (e.g. MCP_AUTH_TOKEN)
envsubst < /opt/protoresearcher/config/nanobot-config.json > /home/sandbox/.nanobot/config.json

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
# Always writes the config to trigger CLIProxyAPI's file watcher reload,
# which forces it to re-validate auth state even if the token hasn't changed.
inject_token() {
    python3 -c "
import json, yaml, time

with open('/opt/claude-creds/.credentials.json') as f:
    creds = json.load(f)

oauth = creds.get('claudeAiOauth', {})
token = oauth.get('accessToken', '')
if not token:
    return

with open('/opt/.cliproxy/config.yaml') as f:
    cfg = yaml.safe_load(f)

old_token = ''
if cfg.get('claude-api-key'):
    old_token = cfg['claude-api-key'][0].get('api-key', '')

cfg['claude-api-key'] = [{'api-key': token}]

# Always rewrite to trigger file watcher (even if token unchanged)
with open('/opt/.cliproxy/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

if token != old_token:
    print(f'[token-refresh] New OAuth token injected at {time.strftime(\"%H:%M:%S\")}')
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
# Re-injects the OAuth token into CLIProxyAPI config every 60 seconds.
# CLIProxyAPI's file watcher auto-reloads when the config changes.
# We always re-inject (not just on mtime change) because:
#   1. The host's Claude Code may refresh the token without changing mtime
#   2. CLIProxyAPI's internal auth state can go stale even with a valid token
#   3. 60s interval keeps the window of staleness small
(
    while true; do
        sleep 60
        inject_token
    done
) &
echo "[entrypoint] Token refresh watcher started (every 60s)"

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
