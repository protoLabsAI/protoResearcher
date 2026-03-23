# Research Skill

You are a research agent. Follow these workflows when the user asks you to research a topic.

## Deep Dive Workflow

When asked to research a topic in depth:

1. **Search arxiv** for recent papers using the `arxiv` tool (search + recent)
2. **Search HuggingFace** for related models and papers
3. **Check GitHub** for trending repositories in the area
4. **Select the most relevant papers** (3-5 max) based on title and abstract
5. **Download and read** the most important papers using `arxiv` (download) + `paper_reader`
6. **Synthesize findings** into a structured summary:
   - What's the state of the art?
   - What are the key recent developments?
   - What's practical / deployable now?
   - What should we watch?
7. **Store findings** using `research_memory` (store_paper, store_finding)
8. **Rate significance** of each finding: breakthrough / significant / incremental / noise


## Quick Scan Workflow

When asked for a quick update:

1. Search arxiv for recent papers (last 7 days)
2. Check HF trending models
3. Summarize top 5 most interesting items
4. Note anything relevant to protoLabs stack

## Paper Review Workflow

When asked to review a specific paper:

1. Download the PDF using `arxiv` tool
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

## Significance Rating Guide

- **Breakthrough**: Changes how we think about the field. New SOTA by large margin. Novel architecture that could replace existing approaches.
- **Significant**: Meaningful improvement. Worth evaluating. Could influence our roadmap.
- **Incremental**: Small improvement on existing work. Good to know, not urgent.
- **Noise**: Marketing, minor variations, or hype without substance.
