FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root sandbox user
ARG SANDBOX_UID=1001
RUN useradd -m -s /bin/bash -u ${SANDBOX_UID} sandbox

# Node.js (for agent-browser)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Browser tool: agent-browser + Chromium
RUN npm install -g agent-browser \
    && (agent-browser install --with-deps 2>/dev/null \
        || (apt-get update && apt-get install -y --no-install-recommends chromium && rm -rf /var/lib/apt/lists/*))

# Install nanobot from submodule + Python deps
COPY nanobot/ /opt/nanobot/
RUN pip install --no-cache-dir /opt/nanobot/ \
    gradio sqlite-vec httpx uvicorn langfuse prometheus-client PyMuPDF

# Install protoResearcher
COPY tools/ /opt/protoresearcher/tools/
COPY knowledge/ /opt/protoresearcher/knowledge/
COPY skills/ /opt/protoresearcher/skills/
COPY audit.py /opt/protoresearcher/audit.py
COPY tracing.py /opt/protoresearcher/tracing.py
COPY metrics.py /opt/protoresearcher/metrics.py
COPY chat_ui.py /opt/protoresearcher/chat_ui.py
COPY server.py /opt/protoresearcher/server.py
COPY entrypoint.sh /opt/protoresearcher/entrypoint.sh
COPY config/ /opt/protoresearcher/config/
COPY static/ /opt/protoresearcher/static/

# Sandbox workspace + knowledge/audit/papers dirs
RUN mkdir -p /sandbox /tmp/sandbox /sandbox/audit /sandbox/knowledge /sandbox/papers \
    && chown -R sandbox:sandbox /sandbox /tmp/sandbox

# Nanobot data dir
RUN mkdir -p /home/sandbox/.nanobot \
    && chown -R sandbox:sandbox /home/sandbox/.nanobot

# Drop to sandbox user
USER sandbox
WORKDIR /sandbox

EXPOSE 7870
CMD ["/opt/protoresearcher/entrypoint.sh"]
