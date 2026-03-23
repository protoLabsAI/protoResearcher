"""System prompt composer for protoResearcher LangGraph agent.

Composes the system prompt from:
1. SOUL.md content (identity, personality, values)
2. Research skills (from skills/research/SKILL.md)
3. Subagent instructions (available types + delegation rules)
4. Dynamic research context (from KnowledgeMiddleware)
"""

from pathlib import Path

from graph.subagents.config import SUBAGENT_REGISTRY


def _read_file(path: str | Path) -> str:
    """Read a file if it exists, return empty string otherwise."""
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""


def build_system_prompt(
    workspace: str = "/sandbox",
    include_subagents: bool = True,
    research_context: str = "",
) -> str:
    """Build the complete system prompt for the lead agent."""
    parts = []

    # 1. Identity from SOUL.md
    soul = _read_file(f"{workspace}/SOUL.md")
    if soul:
        parts.append(soul)
    else:
        parts.append(
            "# protoResearcher 🔬\n\n"
            "You are protoResearcher, an autonomous AI research assistant built by protoLabs.\n"
            "You track, analyze, and synthesize the latest developments in AI and machine learning."
        )

    # 2. Research skills
    skill = _read_file(f"{workspace}/skills/research/SKILL.md")
    if skill:
        parts.append(f"\n# Research Methodology\n\n{skill}")

    # 3. Subagent instructions
    if include_subagents:
        parts.append(_build_subagent_section())

    # 4. Dynamic research context (injected by KnowledgeMiddleware)
    if research_context:
        parts.append(f"\n# Research Context\n\n{research_context}")

    # 5. Guidelines
    parts.append("""
# Guidelines

- Think before acting. Break down complex tasks.
- For multi-source research, delegate to subagents: Explorer scans, Analyst reads, Writer synthesizes.
- Rate significance of every finding: [breakthrough / significant / incremental / noise]
- Always store important findings in research_memory.
- When publishing to Discord, use discord_feed action=publish with content and title. NO channel_id needed.
- Reply directly with text for conversations. Use the task tool to delegate parallel work.
""")

    return "\n\n".join(parts)


def _build_subagent_section() -> str:
    """Build the subagent delegation instructions."""
    lines = [
        "# Subagent Delegation",
        "",
        "You can delegate tasks to specialized subagents using the `task` tool.",
        "Each subagent has focused tools and expertise:",
        "",
    ]

    for name, config in SUBAGENT_REGISTRY.items():
        lines.append(f"- **{name}**: {config.description}")
        lines.append(f"  Tools: {', '.join(config.tools)}")
        lines.append("")

    lines.extend([
        "**Rules:**",
        "- Delegate scanning/discovery to Explorer",
        "- Delegate deep reading/analysis to Analyst",
        "- Delegate writing/publishing to Writer",
        "- For simple questions, answer directly without delegation",
        "- Max 3 concurrent subagent tasks",
        "- Subagents cannot spawn further subagents",
    ])

    return "\n".join(lines)


def build_subagent_prompt(agent_name: str, workspace: str = "/sandbox") -> str:
    """Build system prompt for a specific subagent."""
    config = SUBAGENT_REGISTRY.get(agent_name)
    if not config:
        return f"You are a research subagent. Complete the delegated task efficiently."
    return config.system_prompt
