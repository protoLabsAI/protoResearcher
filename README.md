# protoResearcher

Autonomous AI research agent that tracks the latest developments in AI and machine learning. Built on the [nanobot](https://github.com/HKUDS/nanobot) agent framework with research-specialized tools.

## What it does

- **Scans Discord feeds** for research links, papers, model releases
- **Reads PDFs** — downloads and extracts text from papers
- **Monitors HuggingFace** for new models, datasets, and papers
- **Integrates with [rabbit-hole.io](https://github.com/protoLabsAI/rabbit-hole.io)** — knowledge graph, media ingestion, entity extraction via MCP
- **Tracks GitHub** trending AI/ML repositories and releases
- **Stores knowledge** — papers, findings, digests in a semantic knowledge base
- **Generates digests** — structured research summaries with significance ratings
- **Browses the web** for blog posts, conference pages, and more
- **Runs experiments** — autonomous GPU training via lab mode (inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch))

## Architecture

```
protoResearcher
├── nanobot/              # Agent framework (submodule)
├── tools/
│   ├── discord_feed.py   # Discord channel scanner + link classifier
│   ├── paper_reader.py   # PDF text extraction (PyMuPDF)
│   ├── huggingface.py    # HF Hub models, datasets, papers
│   ├── github_trending.py # GitHub search + releases
│   ├── research_memory.py # Knowledge store tool
│   ├── lab_bench.py      # GPU experiment runner (lab mode)
│   └── browser.py        # Web automation
├── knowledge/            # SQLite + sqlite-vec knowledge base
├── lab/                  # Experiment runner + templates
│   ├── runner.py         # Experiment lifecycle management
│   └── templates/        # LLaMA-Factory configs for Qwen 0.8B/2B
├── server.py             # Gradio UI server
├── chat_ui.py            # Chat interface
├── Dockerfile            # Multi-stage build (base + lab)
└── docker-compose.yml    # Orchestration (with lab GPU profile)
```

## Quick Start

### Prerequisites: Claude Code Authentication

protoResearcher uses **CLIProxyAPI** to access Claude models through your existing Claude Code subscription — no separate API key needed. It works by reading the OAuth token from Claude Code's credential file on your host.

**Setup:**

1. **Install and authenticate Claude Code** on the host machine:

   ```bash
   npm install -g @anthropic-ai/claude-code
   claude  # This opens a browser for OAuth login
   ```

2. **Ensure the credentials file is readable** by the container (runs as uid 1001):

   ```bash
   chmod 644 ~/.claude/.credentials.json
   ```

   This file is mounted read-only into the container at `/opt/claude-creds/`. The entrypoint extracts the OAuth token and injects it into CLIProxyAPI's config. A background watcher refreshes the token every 5 minutes if the file changes.

3. **Verify the file exists:**
   ```bash
   ls -la ~/.claude/.credentials.json
   # Should show: -rw-r--r-- ... .credentials.json
   ```

> **How it works:** CLIProxyAPI runs inside the container on port 8317, exposing an OpenAI-compatible API that routes requests to Anthropic using your Claude Code OAuth token. The nanobot agent is configured to use this as its LLM provider (`cliproxy` in `nanobot-config.json`). This means LLM calls use your Claude Code subscription, not a separate API key.

> **Alternative:** If you prefer to use an API key directly, set `ANTHROPIC_API_KEY` in your environment and change the nanobot config provider from `cliproxy` to `anthropic`.

### Docker (recommended)

```bash
# Clone with submodules (nanobot is a git submodule)
git clone --recursive https://github.com/protoLabsAI/protoResearcher
cd protoResearcher

# Start (basic mode)
docker compose up --build
# UI at http://localhost:7872
```

### With Lab Mode (GPU experiments)

```bash
docker compose --profile lab up --build
# UI at http://localhost:7872, then /lab on in chat
```

Requires NVIDIA GPU. Lab mode mounts LLaMA-Factory + HuggingFace model cache from the host.

### Local

```bash
pip install -r requirements.txt
python server.py --port 7870
```

Requires vLLM running on `localhost:8000` (or configure `config/nanobot-config.json`).

## Rabbit Hole Integration (MCP)

protoResearcher connects to [rabbit-hole.io](https://github.com/protoLabsAI/rabbit-hole.io)'s MCP server via the Streamable HTTP transport. This gives the agent access to 12 research and media processing tools:

| Tool               | What it does                                               |
| ------------------ | ---------------------------------------------------------- |
| `graph_search`     | Search existing entities in the knowledge graph            |
| `research_entity`  | Full research pipeline (multi-source → extract → validate) |
| `extract_entities` | LLM-based entity extraction from text                      |
| `validate_bundle`  | Bundle structural integrity check                          |
| `ingest_bundle`    | Push entities into the Neo4j knowledge graph               |
| `wikipedia_search` | Fetch Wikipedia articles                                   |
| `web_search`       | DuckDuckGo instant answers                                 |
| `tavily_search`    | Premium web search (requires TAVILY_API_KEY on MCP server) |
| `ingest_url`       | Ingest any URL (HTML, PDF, YouTube, audio)                 |
| `ingest_file`      | Ingest local files                                         |
| `transcribe_audio` | Audio transcription                                        |
| `extract_pdf`      | PDF text extraction                                        |

### Setup

1. **Start the rabbit-hole MCP server** (on the same host or network):

   ```bash
   cd /path/to/rabbit-hole.io
   pnpm --filter @proto/mcp-server build

   # Start with auth token
   MCP_AUTH_TOKEN=$(openssl rand -hex 32) \
   MCP_PORT=3398 \
   pm2 start packages/mcp-server/dist/http-server.js --name rabbit-hole-mcp
   ```

2. **Pass the token to protoResearcher** via env:

   ```bash
   MCP_AUTH_TOKEN=<your-token> docker compose up -d
   ```

3. **Verify connectivity** — the container reaches the MCP server at `host.docker.internal:3398`. The agent's tools list will show `mcp_rabbit-hole_*` prefixed tools when connected.

The MCP connection is configured in `config/nanobot-config.json` under `tools.mcpServers.rabbit-hole`. Nanobot connects lazily on the first agent loop.

### Firewall Note

If you're running UFW or iptables with a DROP policy, Docker containers may not be able to reach host ports. Allow traffic from Docker bridge networks:

```bash
sudo ufw allow from 10.0.0.0/8 to any port 3398 comment "MCP server from Docker"
sudo ufw allow from 10.0.0.0/8 to any port 3399 comment "rabbit-hole from Docker"
```

## API — Trigger from Other Agents

protoResearcher exposes an HTTP API at `http://localhost:7872/api/chat`. Any agent or script can trigger research tasks.

```
POST http://localhost:7872/api/chat
Content-Type: application/json

{"message": "<command or natural language>"}
```

```json
// Response
{
  "response": "Markdown-formatted response text",
  "messages": [{ "role": "assistant", "content": "..." }]
}
```

```bash
# Examples
curl -s http://localhost:7872/api/chat -H "Content-Type: application/json" \
  -d '{"message": "/agenda"}'

curl -s http://localhost:7872/api/chat -H "Content-Type: application/json" \
  -d '{"message": "What are the latest developments in MoE architectures?"}'
```

## Chat Commands

| Command                | Description                       |
| ---------------------- | --------------------------------- |
| `/topics`              | Show tracked research topics      |
| `/agenda`              | Research agenda with stats        |
| `/papers [query]`      | Search stored papers              |
| `/recent [n]`          | Show recent findings              |
| `/lab on\|off\|status` | Toggle lab mode (GPU experiments) |
| `/think <level>`       | Set reasoning effort              |
| `/tools`               | List registered tools             |
| `/help`                | Show all commands                 |

## Lab Mode

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). When enabled, the agent gets a `lab_bench` tool that runs autonomous training experiments on local GPUs.

### How it works

1. **Init** — create experiment workspace from a template (Qwen 0.8B or 2B DPO)
2. **Edit** — modify the LLaMA-Factory config (the single modifiable file)
3. **Run** — train with a fixed time budget, metrics auto-extracted
4. **Keep/Discard** — accept or revert via git (each hypothesis = commit)
5. **Repeat** — iterate until the metric stops improving

### Design principles (from autoresearch)

- **Single modifiable file** — `config.yaml` (LLaMA-Factory training config)
- **Fixed time budget** — default 5 minutes per experiment
- **Single metric** — eval_loss for optimization
- **Git-based tracking** — commit per hypothesis, `git reset` on discard
- **Results ledger** — `results.tsv` with commit, metrics, status, description

### Templates

| Template        | Model        | Method   | Description                 |
| --------------- | ------------ | -------- | --------------------------- |
| `dpo_qwen_0.8b` | Qwen3.5-0.8B | LoRA DPO | Fast iteration, tiny model  |
| `dpo_qwen_2b`   | Qwen3.5-2B   | LoRA DPO | More capacity, same dataset |

### Example session

```
/lab on
> Lab mode ON. lab_bench tool registered.

"Initialize a DPO experiment on Qwen 0.8B called baseline-run"
> Experiment initialized from template dpo_qwen_0.8b.

"Run the baseline"
> Experiment complete. eval_loss: 2.341, Peak VRAM: 8420 MB

"Try doubling the LoRA rank to 64"
> Updated lora_rank: 32 -> 64, lora_alpha: 64 -> 128

"Run it"
> Experiment complete. eval_loss: 2.298 (improved!)

"Keep it, then try learning rate 2e-5"
> Marked as KEEP. Updated learning_rate: 1e-5 -> 2e-5
```

## Research Topics (default)

- MoE scaling and efficiency
- Quantization and model compression
- Inference optimization (vLLM, speculative decoding)
- Agentic systems and tool use
- Video generation
- Training methods (DPO, RLHF, LoRA)
- Open-source model releases
- Multimodal models

## Environment Variables

| Variable              | Required                    | Description                                                 |
| --------------------- | --------------------------- | ----------------------------------------------------------- |
| `MCP_AUTH_TOKEN`      | For rabbit-hole integration | Bearer token for MCP server auth                            |
| `ANTHROPIC_API_KEY`   | No                          | Direct Anthropic API (alternative to CLIProxyAPI)           |
| `LANGFUSE_PUBLIC_KEY` | No                          | Langfuse tracing                                            |
| `LANGFUSE_SECRET_KEY` | No                          | Langfuse tracing                                            |
| `LANGFUSE_HOST`       | No                          | Langfuse host (default: `http://host.docker.internal:3001`) |
| `GITHUB_TOKEN`        | No                          | GitHub API (higher rate limits)                             |
| `DISCORD_BOT_TOKEN`   | No                          | Discord channel reading                                     |
| `DISCORD_WEBHOOK_URL` | No                          | Discord digest publishing                                   |
| `LAB_GPU`             | No                          | GPU ID for lab mode (default: `1`)                          |
| `AGENT_BACKEND`       | No                          | `nanobot` (default) or `langgraph`                          |

## Stack

- **Agent**: nanobot (tool-calling agent loop, sessions, LiteLLM provider)
- **LLM**: CLIProxyAPI → Claude Code OAuth (no API key needed) or direct Anthropic API
- **UI**: Gradio 5 (dark theme, PWA)
- **Knowledge**: SQLite + sqlite-vec (semantic search via Ollama embeddings)
- **Training**: LLaMA-Factory with LoRA DPO on Qwen3.5-0.8B/2B
- **Observability**: Langfuse tracing, Prometheus metrics, JSONL audit
- **Container**: Docker with seccomp, read-only root, tmpfs workspace

## Part of protoLabs

protoResearcher is part of the [protoLabs](https://protolabs.studio) autonomous development studio.

| Agent               | Role                                              |
| ------------------- | ------------------------------------------------- |
| **Ava**             | Chief of Staff — orchestration and strategy       |
| **Quinn**           | QA Engineer — verification and release management |
| **protoResearcher** | Research — AI/ML paper tracking and analysis      |
