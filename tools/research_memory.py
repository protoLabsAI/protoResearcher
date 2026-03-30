"""Research memory tool — nanobot Tool interface for the knowledge store.

Wraps KnowledgeStore as a tool the agent can call to store and search
papers, findings, and digests.
"""

import json
from typing import Any

from nanobot.agent.tools.base import Tool

from knowledge.store import KnowledgeStore


class ResearchMemoryTool(Tool):
    """Store and search research knowledge — papers, findings, digests."""

    def __init__(self, store: KnowledgeStore | None = None):
        self._store = store or KnowledgeStore()

    @property
    def name(self) -> str:
        return "research_memory"

    @property
    def description(self) -> str:
        return (
            "Persistent research knowledge store with hybrid search. Actions:\n"
            "- store_paper: Save a paper with metadata and summary\n"
            "- store_finding: Save a research insight or result\n"
            "- store_digest: Save a research digest/summary\n"
            "- search: Hybrid search (vector + keyword) across all stored knowledge\n"
            "- get_topics: List tracked research topics\n"
            "- add_topic: Add a new research topic to track\n"
            "- stats: Show knowledge base statistics"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "store_paper", "store_finding", "store_digest",
                        "search", "get_topics", "add_topic", "stats",
                    ],
                    "description": "Action to perform.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'search').",
                },
                # store_paper fields
                "arxiv_id": {"type": "string", "description": "Arxiv paper ID."},
                "title": {"type": "string", "description": "Paper or digest title."},
                "authors": {"type": "string", "description": "Comma-separated authors."},
                "abstract": {"type": "string", "description": "Paper abstract."},
                "summary": {"type": "string", "description": "Agent-generated summary."},
                "significance": {
                    "type": "string",
                    "enum": ["breakthrough", "significant", "incremental", "noise"],
                    "description": "Significance rating.",
                },
                "tags": {"type": "string", "description": "Comma-separated tags."},
                # store_finding fields
                "content": {"type": "string", "description": "Finding content."},
                "source": {"type": "string", "description": "Source (paper ID, URL, etc)."},
                "source_type": {"type": "string", "description": "Source type: paper/blog/github/model_release."},
                "topic": {"type": "string", "description": "Related topic."},
                "finding_type": {
                    "type": "string",
                    "enum": ["insight", "result", "method", "benchmark", "recommendation"],
                    "description": "Type of finding.",
                },
                # add_topic fields
                "name": {"type": "string", "description": "Topic name."},
                "description": {"type": "string", "description": "Topic description."},
                "keywords": {"type": "string", "description": "Comma-separated keywords."},
                "priority": {"type": "integer", "description": "Priority 0-4 (0=critical)."},
                # search
                "filter_table": {"type": "string", "description": "Filter search to: papers/findings/digests."},
                "k": {"type": "integer", "description": "Number of results (default 10)."},
                "search_mode": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "keyword"],
                    "description": "Search mode: hybrid (default, best quality), vector (semantic only), keyword (BM25 only).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]

        if action == "store_paper":
            arxiv_id = kwargs.get("arxiv_id", "")
            title = kwargs.get("title", "")
            if not arxiv_id or not title:
                return "Error: 'arxiv_id' and 'title' are required."
            authors_str = kwargs.get("authors", "")
            authors = [a.strip() for a in authors_str.split(",") if a.strip()] if authors_str else []
            tags_str = kwargs.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
            ok = self._store.add_paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=kwargs.get("abstract", ""),
                summary=kwargs.get("summary", ""),
                significance=kwargs.get("significance", "unknown"),
                tags=tags,
            )
            return "Paper stored." if ok else "Error: Failed to store paper."

        if action == "store_finding":
            content = kwargs.get("content", "")
            if not content:
                return "Error: 'content' is required."
            ok = self._store.add_finding(
                content=content,
                source=kwargs.get("source", ""),
                source_type=kwargs.get("source_type", ""),
                topic=kwargs.get("topic", ""),
                finding_type=kwargs.get("finding_type", "insight"),
                significance=kwargs.get("significance", ""),
            )
            return "Finding stored." if ok else "Error: Failed to store finding."

        if action == "store_digest":
            title = kwargs.get("title", "")
            content = kwargs.get("content", "")
            if not title or not content:
                return "Error: 'title' and 'content' are required."
            ok = self._store.add_digest(
                title=title,
                content=content,
                digest_type=kwargs.get("finding_type", "weekly"),
                topic=kwargs.get("topic", ""),
            )
            return "Digest stored." if ok else "Error: Failed to store digest."

        if action == "search":
            query = kwargs.get("query", "")
            if not query:
                return "Error: 'query' is required."
            k = kwargs.get("k", 10)
            filter_table = kwargs.get("filter_table")
            search_mode = kwargs.get("search_mode", "hybrid")
            if search_mode == "keyword":
                results = self._store.keyword_search(query, k=k, filter_table=filter_table)
            elif search_mode == "vector":
                results = self._store.search(query, k=k, filter_table=filter_table)
            else:
                results = self._store.hybrid_search(query, k=k, filter_table=filter_table)
            if not results:
                return "No results found."
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. [{r['table']}:{r['source_id']}] (dist: {r['distance']:.3f})\n"
                    f"   {r['preview']}"
                )
            return "\n".join(lines)

        if action == "get_topics":
            topics = self._store.get_topics()
            if not topics:
                return "No topics configured. Use 'add_topic' to start tracking research areas."
            lines = ["**Research Topics:**"]
            for t in topics:
                kw = json.loads(t.get("keywords", "[]"))
                kw_str = ", ".join(kw) if kw else ""
                scanned = t.get("last_scanned_at", "never") or "never"
                lines.append(
                    f"- **{t['name']}** (P{t['priority']}) — {t.get('description', '')}\n"
                    f"  Keywords: {kw_str}\n"
                    f"  Last scanned: {scanned}"
                )
            return "\n".join(lines)

        if action == "add_topic":
            name = kwargs.get("name", "")
            if not name:
                return "Error: 'name' is required."
            kw_str = kwargs.get("keywords", "")
            keywords = [k.strip() for k in kw_str.split(",") if k.strip()] if kw_str else []
            ok = self._store.add_topic(
                name=name,
                description=kwargs.get("description", ""),
                keywords=keywords,
                priority=kwargs.get("priority", 2),
            )
            return f"Topic '{name}' added." if ok else "Error: Failed to add topic."

        if action == "stats":
            stats = self._store.get_stats()
            if not stats:
                return "Knowledge base not initialized."
            lines = ["**Knowledge Base Stats:**"]
            for table, count in stats.items():
                lines.append(f"- {table}: {count}")
            return "\n".join(lines)

        return f"Error: Unknown action '{action}'."
