"""LangGraph tool adapters for protoResearcher.

Wraps existing nanobot Tool classes as LangChain @tool functions.
All business logic stays in the original classes — these are thin adapters.
"""

from typing import Optional

from langchain_core.tools import tool

import os

from tools.paper_reader import PaperReaderTool
from tools.huggingface import HuggingFaceTool
from tools.github_trending import GitHubTrendingTool
from tools.research_memory import ResearchMemoryTool
from tools.browser import BrowserTool
from tools.lab_monitor import LabMonitorTool
from tools.rabbit_hole_bridge import RabbitHoleBridgeTool


# Instantiate underlying tool classes (stateless singletons)
_rabbit_hole_bridge = RabbitHoleBridgeTool()
_paper_reader = PaperReaderTool()
_huggingface = HuggingFaceTool()
_github_trending = GitHubTrendingTool()
_browser = BrowserTool()
_lab_monitor = LabMonitorTool()


# Discord tools — only loaded when DISCORD_BOT_TOKEN is set
_discord_feed_tool = None
if os.environ.get("DISCORD_BOT_TOKEN"):
    from tools.discord_feed import DiscordFeedTool
    _discord_feed_tool = DiscordFeedTool()

    @tool
    async def discord_feed(
        action: str,
        channel_id: str = "",
        guild_id: str = "",
        limit: int = 50,
        after: str = "",
        content: str = "",
        title: str = "",
    ) -> str:
        """Read Discord channels and publish research digests.

        READING (requires channel_id):
        - scan: Read recent messages and extract classified URLs
        - history: Get raw message history
        - channels: List channels in a server (guild_id required)
        - digest: Scan a channel and produce a structured link digest

        PUBLISHING (NO channel_id needed — uses pre-configured webhook):
        - publish: Post content to #protolabs-research via webhook.
          Just provide 'content' and optionally 'title'. The webhook is auto-configured.
        """
        return await _discord_feed_tool.execute(
            action=action, channel_id=channel_id, guild_id=guild_id,
            limit=limit, after=after, content=content, title=title,
        )


@tool
async def paper_reader(
    action: str,
    paper: str = "",
    pages: str = "",
) -> str:
    """Read PDF papers that have been downloaded.

    - read: Extract text from a paper (by path or paper ID)
    - list: List downloaded papers
    Tip: Use the 'browser' tool or rabbit-hole MCP to fetch PDFs first.
    """
    return await _paper_reader.execute(action=action, paper=paper, pages=pages)


@tool
async def huggingface(
    action: str,
    query: str = "",
    model_id: str = "",
    sort: str = "trending",
    limit: int = 10,
    filter_task: str = "",
) -> str:
    """Search HuggingFace Hub for models, datasets, and papers.

    - search_models: Find models by query, sorted by trending/downloads/created
    - search_datasets: Find datasets by query
    - model_card: Get the README/model card for a specific model
    - search_papers: Search HF papers
    """
    return await _huggingface.execute(
        action=action, query=query, model_id=model_id,
        sort=sort, limit=limit, filter_task=filter_task,
    )


@tool
async def github_trending(
    action: str,
    query: str = "",
    topic: str = "",
    language: str = "",
    min_stars: int = 100,
    created_after: str = "",
    repos: str = "",
    limit: int = 10,
    sort: str = "stars",
) -> str:
    """Search GitHub for trending and notable AI/ML repositories.

    - search: Search repos by query with star/activity filters
    - recent_repos: Find recently created repos with high engagement
    - releases: Check latest releases for tracked repos
    """
    return await _github_trending.execute(
        action=action, query=query, topic=topic, language=language,
        min_stars=min_stars, created_after=created_after, repos=repos,
        limit=limit, sort=sort,
    )


@tool
async def browser(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
    query: str = "",
) -> str:
    """Automate a web browser. Actions: open, snapshot, screenshot, click, fill, find, type, wait.

    Returns accessibility tree snapshots by default (token-efficient).
    Use 'open' first, then 'snapshot' to read page content.
    """
    return await _browser.execute(
        action=action, url=url, selector=selector, text=text, query=query,
    )


def create_research_memory_tool(store=None):
    """Factory: creates research_memory tool with injected KnowledgeStore."""
    from knowledge.store import KnowledgeStore
    _tool = ResearchMemoryTool(store or KnowledgeStore())

    @tool
    async def research_memory(
        action: str,
        query: str = "",
        arxiv_id: str = "",
        title: str = "",
        authors: str = "",
        abstract: str = "",
        summary: str = "",
        significance: str = "",
        tags: str = "",
        content: str = "",
        source: str = "",
        source_type: str = "",
        topic: str = "",
        finding_type: str = "insight",
        name: str = "",
        description: str = "",
        keywords: str = "",
        priority: int = 2,
        filter_table: str = "",
        k: int = 10,
    ) -> str:
        """Persistent research knowledge store with semantic search.

        - store_paper: Save a paper with metadata and summary
        - store_finding: Save a research insight or result
        - store_digest: Save a research digest/summary
        - search: Semantic search across all stored knowledge
        - get_topics: List tracked research topics
        - add_topic: Add a new research topic to track
        - stats: Show knowledge base statistics
        """
        return await _tool.execute(
            action=action, query=query, arxiv_id=arxiv_id, title=title,
            authors=authors, abstract=abstract, summary=summary,
            significance=significance, tags=tags, content=content,
            source=source, source_type=source_type, topic=topic,
            finding_type=finding_type, name=name, description=description,
            keywords=keywords, priority=priority, filter_table=filter_table, k=k,
        )

    return research_memory


def create_lab_bench_tool():
    """Factory: creates lab_bench tool (called at runtime when /lab on)."""
    from tools.lab_bench import LabBenchTool
    _tool = LabBenchTool()

    @tool
    async def lab_bench(
        action: str,
        experiment: str = "",
        template: str = "dpo_qwen_0.8b",
        key: str = "",
        value: str = "",
        description: str = "",
        gpu: str = "1",
        time_budget: int = 300,
        tail: int = 50,
    ) -> str:
        """Run autonomous training experiments on tiny Qwen models.

        Uses LLaMA-Factory with LoRA DPO training on local GPU.
        Workflow: init -> edit config -> run -> keep/discard -> repeat.
        """
        return await _tool.execute(
            action=action, experiment=experiment, template=template,
            key=key, value=value, description=description,
            gpu=gpu, time_budget=time_budget, tail=tail,
        )

    return lab_bench


@tool
async def lab_monitor(
    action: str,
    path: str = "",
    sha: str = "",
    days: int = 7,
    since: str = "",
    limit: int = 20,
) -> str:
    """Monitor protoLabsAI/lab for new experiments, docs, and changes.

    - recent_commits: Get commits since last check (or last N days)
    - read_file: Read a file from the repo (README, experiment index, etc.)
    - experiments: List active experiments from the lab index
    - diff: Show what changed in a specific commit
    - watch_paths: Show which paths are monitored
    - changes_since: Get all changes to watched paths since a date
    """
    return await _lab_monitor.execute(
        action=action, path=path, sha=sha,
        days=days, since=since, limit=limit,
    )


@tool
async def rabbit_hole_bridge(
    action: str,
    query: str = "",
    arxiv_id: str = "",
    model_id: str = "",
    text: str = "",
    focus_entity: str = "",
    paper_ids: Optional[list[str]] = None,
    model_ids: Optional[list[str]] = None,
    limit: int = 10,
) -> str:
    """Ship research data to rabbit-hole.io's knowledge graph.

    - search_graph: Check what's already in the graph (query required)
    - ingest_paper: Send a stored paper to the graph (arxiv_id required)
    - ingest_model: Send a stored model release to the graph (model_id required)
    - ingest_text: Extract entities from free text and ingest (text required)
    - ingest_batch: Send multiple papers/models at once (paper_ids and/or model_ids)
    """
    return await _rabbit_hole_bridge.execute(
        action=action, query=query, arxiv_id=arxiv_id, model_id=model_id,
        text=text, focus_entity=focus_entity, paper_ids=paper_ids,
        model_ids=model_ids, limit=limit,
    )


def get_all_tools(knowledge_store=None):
    """Get all research tools as LangChain tool objects."""
    tools = [
        rabbit_hole_bridge,
        paper_reader,
        huggingface,
        github_trending,
        browser,
        lab_monitor,
        create_research_memory_tool(knowledge_store),
    ]
    if _discord_feed_tool is not None:
        tools.insert(0, discord_feed)
    return tools
