"""Arxiv search and paper retrieval tool for protoResearcher.

Uses the arxiv.org Atom API — no API key required.
Rate limit: 1 request per 3 seconds (enforced).
"""

import asyncio
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool

_ARXIV_API = "http://export.arxiv.org/api/query"
_PAPERS_DIR = Path("/sandbox/papers")
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_last_request_time = 0.0


async def _rate_limit():
    """Enforce 1 request per 3 seconds."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < 3.0:
        await asyncio.sleep(3.0 - elapsed)
    _last_request_time = time.monotonic()


def _parse_entry(entry: ET.Element) -> dict[str, Any]:
    """Parse an Atom entry into a paper dict."""
    ns = _ATOM_NS

    arxiv_id = entry.findtext("atom:id", "", ns).split("/abs/")[-1]
    title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
    abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
    published = entry.findtext("atom:published", "", ns)[:10]

    authors = []
    for author in entry.findall("atom:author", ns):
        name = author.findtext("atom:name", "", ns)
        if name:
            authors.append(name)

    categories = []
    for cat in entry.findall("arxiv:primary_category", ns):
        term = cat.get("term", "")
        if term:
            categories.append(term)
    for cat in entry.findall("atom:category", ns):
        term = cat.get("term", "")
        if term and term not in categories:
            categories.append(term)

    pdf_url = ""
    for link in entry.findall("atom:link", ns):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")

    return {
        "id": arxiv_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "published": published,
        "pdf_url": pdf_url,
        "categories": categories,
    }


class ArxivTool(Tool):
    """Search arxiv, fetch paper metadata, download PDFs."""

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def description(self) -> str:
        return (
            "Search and retrieve papers from arxiv.org. Actions:\n"
            "- search: Search by query, returns papers with titles, authors, abstracts\n"
            "- recent: Get newest papers in specified categories\n"
            "- paper: Get full metadata for a specific arxiv ID\n"
            "- download: Download a paper's PDF to local storage"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "recent", "paper", "download"],
                    "description": "Action to perform.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'search' action).",
                },
                "arxiv_id": {
                    "type": "string",
                    "description": "Arxiv paper ID, e.g. '2401.12345' (for 'paper'/'download').",
                },
                "categories": {
                    "type": "string",
                    "description": "Comma-separated arxiv categories, e.g. 'cs.AI,cs.LG' (for 'recent').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 10, max 50).",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort order (default: relevance for search, submittedDate for recent).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]

        if action == "search":
            return await self._search(kwargs)
        elif action == "recent":
            return await self._recent(kwargs)
        elif action == "paper":
            return await self._paper(kwargs)
        elif action == "download":
            return await self._download(kwargs)
        else:
            return f"Error: Unknown action '{action}'."

    async def _search(self, kwargs: dict) -> str:
        query = kwargs.get("query", "")
        if not query:
            return "Error: 'query' is required for search."
        max_results = min(kwargs.get("max_results", 10), 50)
        sort_by = kwargs.get("sort_by", "relevance")

        await _rate_limit()
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(_ARXIV_API, params=params)
                resp.raise_for_status()
        except Exception as e:
            return f"Error: arxiv API request failed: {e}"

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", _ATOM_NS)
        if not entries:
            return "No papers found."

        papers = [_parse_entry(e) for e in entries]
        return self._format_papers(papers)

    async def _recent(self, kwargs: dict) -> str:
        cats_str = kwargs.get("categories", "cs.AI,cs.LG,cs.CL")
        categories = [c.strip() for c in cats_str.split(",")]
        max_results = min(kwargs.get("max_results", 20), 50)

        cat_query = " OR ".join(f"cat:{c}" for c in categories)

        await _rate_limit()
        params = {
            "search_query": cat_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(_ARXIV_API, params=params)
                resp.raise_for_status()
        except Exception as e:
            return f"Error: arxiv API request failed: {e}"

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", _ATOM_NS)
        if not entries:
            return "No recent papers found."

        papers = [_parse_entry(e) for e in entries]
        return self._format_papers(papers)

    async def _paper(self, kwargs: dict) -> str:
        arxiv_id = kwargs.get("arxiv_id", "")
        if not arxiv_id:
            return "Error: 'arxiv_id' is required."

        await _rate_limit()
        params = {"id_list": arxiv_id}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(_ARXIV_API, params=params)
                resp.raise_for_status()
        except Exception as e:
            return f"Error: arxiv API request failed: {e}"

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", _ATOM_NS)
        if not entries:
            return f"Paper {arxiv_id} not found."

        paper = _parse_entry(entries[0])
        lines = [
            f"**{paper['title']}**",
            f"Authors: {', '.join(paper['authors'])}",
            f"Published: {paper['published']}",
            f"Categories: {', '.join(paper['categories'])}",
            f"PDF: {paper['pdf_url']}",
            "",
            "**Abstract:**",
            paper["abstract"],
        ]
        return "\n".join(lines)

    async def _download(self, kwargs: dict) -> str:
        arxiv_id = kwargs.get("arxiv_id", "")
        if not arxiv_id:
            return "Error: 'arxiv_id' is required."

        _PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = arxiv_id.replace("/", "_")
        pdf_path = _PAPERS_DIR / f"{safe_id}.pdf"

        if pdf_path.exists():
            return f"Paper already downloaded: {pdf_path}"

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                resp = await client.get(pdf_url)
                resp.raise_for_status()
                pdf_path.write_bytes(resp.content)
        except Exception as e:
            return f"Error downloading PDF: {e}"

        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        return f"Downloaded {arxiv_id} ({size_mb:.1f} MB) → {pdf_path}"

    def _format_papers(self, papers: list[dict]) -> str:
        lines = []
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors_str += f" +{len(p['authors']) - 3}"
            abstract_preview = p["abstract"][:200] + "..." if len(p["abstract"]) > 200 else p["abstract"]
            lines.append(
                f"{i}. **{p['title']}**\n"
                f"   [{p['id']}] {authors_str} ({p['published']})\n"
                f"   Categories: {', '.join(p['categories'][:4])}\n"
                f"   {abstract_preview}\n"
            )
        return "\n".join(lines)
