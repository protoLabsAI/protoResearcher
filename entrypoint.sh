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
