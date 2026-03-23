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
- **Runs experiments** — autonomous GPU training via lab mode (inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch))

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

### Docker (recommended)

```bash
docker compose up --build
# UI at http://localhost:7870
```

### With Lab Mode (GPU experiments)

```bash
docker compose --profile lab up --build
# UI at http://localhost:7870, then /lab on in chat
```

Requires NVIDIA GPU. Lab mode mounts LLaMA-Factory + HuggingFace model cache from the host.

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
| `/lab on\|off\|status` | Toggle lab mode (GPU experiments) |
| `/think <level>` | Set reasoning effort |
| `/tools` | List registered tools |
| `/help` | Show all commands |

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

| Template | Model | Method | Description |
|----------|-------|--------|-------------|
| `dpo_qwen_0.8b` | Qwen3.5-0.8B | LoRA DPO | Fast iteration, tiny model |
| `dpo_qwen_2b` | Qwen3.5-2B | LoRA DPO | More capacity, same dataset |

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

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | Langfuse tracing |
| `LANGFUSE_HOST` | No | Langfuse host (default: `http://host.docker.internal:3001`) |
| `GITHUB_TOKEN` | No | GitHub API (higher rate limits) |
| `LAB_GPU` | No | GPU ID for lab mode (default: `1`) |

## Stack

- **Agent**: nanobot (tool-calling agent loop, sessions, LiteLLM provider)
- **UI**: Gradio 5 (dark theme, PWA)
- **Knowledge**: SQLite + sqlite-vec (semantic search via Ollama embeddings)
- **Training**: LLaMA-Factory with LoRA DPO on Qwen3.5-0.8B/2B
- **Observability**: Langfuse tracing, Prometheus metrics, JSONL audit
- **Container**: Docker with seccomp, read-only root, tmpfs workspace
- **LLM**: vLLM (local, OpenAI-compatible)

## Part of protoLabs

This is a research tool in the [protoLabs](https://protolabs.studio) AI stack, running on ava-ai (2x RTX PRO 6000 Blackwell, 192 GB VRAM).
