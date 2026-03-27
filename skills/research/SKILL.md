# Research Skill

You are a research agent. Follow these workflows when the user asks you to research a topic.

## Deep Dive Workflow

When asked to research a topic in depth:

1. **Scan Discord feeds** for recent links using `discord_feed` (scan/digest)
2. **Search HuggingFace** for related models and papers
3. **Check GitHub** for trending repositories in the area
4. **Browse the web** for blog posts, announcements, conference pages
5. **Read papers** found via links using `paper_reader` (download PDFs via `browser` first)
6. **Synthesize findings** into a structured summary:
   - What's the state of the art?
   - What are the key recent developments?
   - What's practical / deployable now?
   - What should we watch?
7. **Store findings** using `research_memory` (store_paper, store_finding)
8. **Rate significance** of each finding: breakthrough / significant / incremental / noise

## Quick Scan Workflow

When asked for a quick update:

1. Scan Discord feeds for recent links and discussion
2. Check HF trending models
3. Summarize top 5 most interesting items
4. Note anything relevant to protoLabs stack

## Paper Review Workflow

When asked to review a specific paper:

1. Fetch the PDF using `browser` tool (or use rabbit-hole MCP for ingestion)
2. Read the full paper using `paper_reader`
3. Provide structured analysis:
   - **Problem**: What problem does it solve?
   - **Method**: What's the approach?
   - **Results**: Key quantitative results
   - **Significance**: Why does this matter?
   - **Limitations**: What are the weaknesses?
   - **Relevance**: How does this relate to our work?
4. Store the paper and findings in research_memory

## Digest Generation

When generating a research digest:

1. Search research_memory for recent findings on the topic
2. Organize by theme/significance
3. Write a concise digest with:
   - Executive summary (3-5 sentences)
   - Key findings (bullet points)
   - Notable papers (with significance ratings)
   - Recommendations for the team
4. Store the digest using research_memory (store_digest)
5. **Publish to Discord** using `discord_feed` with `action=publish`:
   - Pass `content` with the full digest text and `title` with a descriptive heading
   - Do NOT pass `channel_id` — publish uses a pre-configured webhook automatically
   - Long content is auto-chunked into multiple embeds

## Publishing to Discord

To publish any content to the team's Discord:

```
discord_feed action=publish title="Weekly Digest — 2026-03-24" content="..."
```

**Important:** The `publish` action does NOT need a `channel_id`. It posts via a webhook to #protolabs-research automatically. Only `scan`, `history`, and `digest` actions need `channel_id`.

## Knowledge Graph Integration

All research findings should be shipped to rabbit-hole.io's knowledge graph so they're searchable and connected beyond this session. Use the `rabbit_hole_bridge` tool.

### Before Researching
Call `rabbit_hole_bridge action=search_graph query="<topic>"` to check what's already in the knowledge graph. If entities exist with recent data, build on them rather than re-researching from scratch.

### After Storing a Paper
After `research_memory store_paper`, also run:
```
rabbit_hole_bridge action=ingest_paper arxiv_id="<arxiv_id>"
```
This ships the paper + authors + relationships to the graph.

### After Storing a Model Release
After discovering a model, also run:
```
rabbit_hole_bridge action=ingest_model model_id="<model_id>"
```

### After Generating a Digest
Ship the digest text for entity extraction:
```
rabbit_hole_bridge action=ingest_text text="<digest content>" focus_entity="<main topic>"
```

### Batch Ingestion
After a bulk scan (Explorer workflow), collect all paper and model IDs, then ingest in one shot:
```
rabbit_hole_bridge action=ingest_batch paper_ids=["id1","id2"] model_ids=["org/model1"]
```

### Media Processing
For PDFs and URLs, you can also use rabbit-hole's MCP media tools (if MCP is connected):
- `mcp_rabbit-hole_ingest_url` — extract text from any URL
- `mcp_rabbit-hole_extract_pdf` — extract text from PDFs

## Significance Rating Guide

- **Breakthrough**: Changes how we think about the field. New SOTA by large margin. Novel architecture that could replace existing approaches.
- **Significant**: Meaningful improvement. Worth evaluating. Could influence our roadmap.
- **Incremental**: Small improvement on existing work. Good to know, not urgent.
- **Noise**: Marketing, minor variations, or hype without substance.
