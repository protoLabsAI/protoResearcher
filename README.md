# protoResearcher

Autonomous AI research agent that tracks the latest developments in AI and machine learning. Built on the [nanobot](https://github.com/HKUDS/nanobot) agent framework with research-specialized tools.

## What it does

- **Searches arxiv** for papers by topic, category, or recency
- **Reads PDFs** — downloads and extracts text from papers
- **Monitors HuggingFace** for new models, datasets, and papers
- **Tracks GitHub** trending AI/ML repositories and releases
- **Stores knowledge** — papers, findings, digests in a semantic knowledge base
- **Generates digests** — structured research summaries with significance ratings
- **Browses the web** for blog posts, conference pages, and more

## Architecture

```
protoResearcher
├── nanobot/              # Agent framework (submodule)
├── tools/
│   ├── arxiv.py          # Arxiv API search + paper metadata
│   ├── paper_reader.py   # PDF text extraction (PyMuPDF)
│   ├── huggingface.py    # HF Hub models, datasets, papers
│   ├── github_trending.py # GitHub search + releases
│   ├── research_memory.py # Knowledge store tool
│   └── browser.py        # Web automation
├── knowledge/
│   ├── store.py          # SQLite + sqlite-vec knowledge base
│   ├── schema.sql        # Database schema
│   └── models.py         # Data models
├── server.py             # Gradio UI server
├── chat_ui.py            # Chat interface
├── Dockerfile            # Container build
└── docker-compose.yml    # Orchestration
```

## Quick Start

### Docker (recommended)

```bash
docker compose up --build
# UI at http://localhost:7870
```

### Local

```bash
pip install -r requirements.txt
python server.py --port 7870
```

Requires vLLM running on `localhost:8000` (or configure `config/nanobot-config.json`).

## Chat Commands

| Command | Description |
|---------|-------------|
| `/topics` | Show tracked research topics |
| `/agenda` | Research agenda with stats |
| `/papers [query]` | Search stored papers |
| `/recent [n]` | Show recent findings |
| `/think <level>` | Set reasoning effort |
| `/tools` | List registered tools |
| `/help` | Show all commands |

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

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | Langfuse tracing |
| `LANGFUSE_HOST` | No | Langfuse host (default: `http://host.docker.internal:3001`) |
| `GITHUB_TOKEN` | No | GitHub API (higher rate limits) |

## Stack

- **Agent**: nanobot (tool-calling agent loop, sessions, LiteLLM provider)
- **UI**: Gradio 5 (dark theme, PWA)
- **Knowledge**: SQLite + sqlite-vec (semantic search via Ollama embeddings)
- **Observability**: Langfuse tracing, Prometheus metrics, JSONL audit
- **Container**: Docker with seccomp, read-only root, tmpfs workspace
- **LLM**: vLLM (local, OpenAI-compatible)

## Part of protoLabs

This is a research tool in the [protoLabs](https://protolabs.studio) AI stack, running on ava-ai (2x RTX PRO 6000 Blackwell, 192 GB VRAM).
