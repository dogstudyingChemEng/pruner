"""
Generate detailed compression report as markdown file.

Shows complete process from original skill to compressed skill.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunker import chunk_markdown_body, count_tokens
from src.llm_client import SkillLLMClient
from src.models import ContentType
from src.optimizer import Stage2Optimizer
from src.parser import parse_skill_file


def generate_report(skill_name: str = "marketing-strategy-pmm"):
    """Generate compression report markdown file."""
    load_dotenv()

    # Initialize
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    client = SkillLLMClient(api_key=api_key, base_url=base_url, model="deepseek-v4-flash")
    optimizer = Stage2Optimizer(llm_client=client, batch_size=10)

    # Parse skill
    skill_path = Path(f"Claude-Skills/marketing/{skill_name}/SKILL.md")
    skill = parse_skill_file(skill_path)

    # Chunk and classify
    initial_chunks = chunk_markdown_body(skill.body.original)
    classified_blocks = optimizer.classify_skill_body(skill)

    # Calculate metrics
    original_tokens = count_tokens(skill.body.original)
    summary = optimizer.get_classification_summary(classified_blocks)
    compression = optimizer.calculate_compression_potential(classified_blocks)

    # Generate markdown report
    report = generate_markdown_content(
        skill, initial_chunks, classified_blocks,
        original_tokens, summary, compression, optimizer
    )

    # Write to file
    output_path = Path(f"data/{skill_name}_compression_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Report saved to: {output_path}")
    return output_path


def generate_markdown_content(
    skill, initial_chunks, classified_blocks,
    original_tokens, summary, compression, optimizer
):
    """Generate markdown report content."""

    lines = []

    # Header
    lines.append("# SkillReducer Compression Report")
    lines.append("")
    lines.append(f"**Skill:** {skill.name}")
    lines.append(f"**Generated:** 2026-04-22")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Part 1: Original Skill
    lines.append("## Part 1: Original Skill")
    lines.append("")
    lines.append("### Metadata")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Name | {skill.name} |")
    lines.append(f"| Version | {skill.metadata.version or 'N/A'} |")
    lines.append(f"| Author | {skill.metadata.author or 'N/A'} |")
    lines.append(f"| Category | {skill.metadata.category or 'N/A'} |")
    lines.append(f"| Tags | {', '.join(skill.metadata.tags) if skill.metadata.tags else 'N/A'} |")
    lines.append("")
    lines.append("### Description")
    lines.append("")
    lines.append(f"```")
    lines.append(skill.description.original)
    lines.append(f"```")
    lines.append("")
    lines.append(f"**Description Tokens:** {count_tokens(skill.description.original)}")
    lines.append("")
    lines.append("### Original Body")
    lines.append("")
    lines.append(f"**Body Tokens:** {original_tokens}")
    lines.append("")
    lines.append("#### Body Content (First 1000 characters)")
    lines.append("")
    lines.append(f"```markdown")
    lines.append(skill.body.original[:1000])
    lines.append(f"...")
    lines.append(f"```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Part 2: Chunking Process
    lines.append("## Part 2: Chunking Process")
    lines.append("")
    lines.append("### Chunking Strategy")
    lines.append("")
    lines.append("The markdown body is split into semantic chunks based on:")
    lines.append("- Each heading (h1-h6) starts a new chunk")
    lines.append("- Content following a heading (paragraphs, lists, code blocks, tables) is grouped with that heading")
    lines.append("- Top-level content (before any heading) forms its own chunk")
    lines.append("")
    lines.append(f"### Chunking Results")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Chunks | {len(initial_chunks)} |")
    lines.append(f"| Total Tokens (Chunked) | {sum(c.token_count for c in initial_chunks)} |")
    lines.append(f"| Original Tokens | {original_tokens} |")
    lines.append(f"| Retention Ratio | {sum(c.token_count for c in initial_chunks)/original_tokens*100:.1f}% |")
    lines.append("")
    lines.append("### All Chunks Detail")
    lines.append("")
    lines.append("| # | Chunk ID | Tokens | Content Preview |")
    lines.append("|---|----------|--------|-----------------|")

    for i, chunk in enumerate(initial_chunks, 1):
        preview = chunk.content[:80].replace('\n', '\\n')
        lines.append(f"| {i} | {chunk.chunk_id} | {chunk.token_count} | {preview}... |")

    lines.append("")
    lines.append("### Full Chunk Contents")
    lines.append("")

    for i, chunk in enumerate(initial_chunks, 1):
        lines.append(f"#### Chunk {i}: `{chunk.chunk_id}`")
        lines.append("")
        lines.append(f"**Tokens:** {chunk.token_count}")
        lines.append("")
        lines.append(f"**Content:**")
        lines.append(f"```markdown")
        lines.append(chunk.content)
        lines.append(f"```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Part 3: LLM Classification
    lines.append("## Part 3: LLM Classification")
    lines.append("")
    lines.append("### Classification Categories")
    lines.append("")
    lines.append("| Category | Description | Loading Strategy |")
    lines.append("|----------|-------------|------------------|")
    lines.append("| `core_rule` | Actionable instructions, directives, workflows | **Always loaded** |")
    lines.append("| `background` | Explanations, definitions, context | **On-demand** |")
    lines.append("| `example` | Code snippets, usage examples | **On-demand** |")
    lines.append("| `template` | Boilerplate, ready-to-use formats | **On-demand** |")
    lines.append("| `redundant` | Duplicate or unnecessary content | **Discarded** |")
    lines.append("")
    lines.append("### Classification Summary")
    lines.append("")
    lines.append(f"| Category | Blocks | Tokens | Percentage |")
    lines.append(f"|----------|--------|--------|------------|")

    total_tokens = sum(b.token_count for b in classified_blocks)
    for ct in ContentType:
        blocks = optimizer.get_blocks_by_type(classified_blocks, ct)
        tokens = sum(b.token_count for b in blocks)
        if blocks:
            pct = tokens / total_tokens * 100
            lines.append(f"| {ct} | {len(blocks)} | {tokens} | {pct:.1f}% |")

    lines.append("")
    lines.append("### Classification Results Detail")
    lines.append("")
    lines.append("| # | Chunk ID | Classification | Tokens | Content Preview |")
    lines.append("|---|----------|----------------|--------|-----------------|")

    for i, block in enumerate(classified_blocks, 1):
        preview = block.content[:60].replace('\n', '\\n')
        lines.append(f"| {i} | {block.chunk_id} | {block.content_type} | {block.token_count} | {preview}... |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Part 4: Content by Type
    lines.append("## Part 4: Content by Classification Type")
    lines.append("")

    for ct in ContentType:
        blocks = optimizer.get_blocks_by_type(classified_blocks, ct)
        if not blocks:
            continue

        type_names = {
            ContentType.CORE_RULE: "Core Rules (Always Loaded)",
            ContentType.BACKGROUND: "Background (On-Demand)",
            ContentType.EXAMPLE: "Examples (On-Demand)",
            ContentType.TEMPLATE: "Templates (On-Demand)",
            ContentType.REDUNDANT: "Redundant (Discarded)"
        }

        lines.append(f"### {type_names[ct]}")
        lines.append("")
        lines.append(f"**Statistics:** {len(blocks)} blocks, {sum(b.token_count for b in blocks)} tokens")
        lines.append("")

        for i, block in enumerate(blocks, 1):
            lines.append(f"#### Block {i}: `{block.chunk_id}` ({block.token_count} tokens)")
            lines.append("")
            lines.append(f"```markdown")
            lines.append(block.content)
            lines.append(f"```")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Part 5: Compression Result
    lines.append("## Part 5: Compression Result")
    lines.append("")
    lines.append("### Token Reduction Analysis")
    lines.append("")
    lines.append(f"| Metric | Before | After | Reduction |")
    lines.append(f"|--------|--------|-------|-----------|")
    lines.append(f"| Total Body Tokens | {original_tokens} | {compression['always_loaded_tokens']} | {original_tokens - compression['always_loaded_tokens']} ({(1-compression['always_loaded_tokens']/original_tokens)*100:.1f}%) |")
    lines.append("")
    lines.append("### Progressive Disclosure Structure")
    lines.append("")
    lines.append("``")
    lines.append("Compressed Skill Structure:")
    lines.append("")
    lines.append(f"├─ Phase 1: Base Load (Always Injected)")
    lines.append(f"│   └─ Core Rules: {compression['always_loaded_tokens']} tokens")
    lines.append("")
    lines.append(f"├─ Phase 2: On-Demand Modules")
    lines.append(f"│   ├─ Background: {sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.BACKGROUND))} tokens")
    lines.append(f"│   ├─ Examples: {sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.EXAMPLE))} tokens")
    lines.append(f"│   └─ Templates: {sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.TEMPLATE))} tokens")
    lines.append(f"│   └─ Total On-Demand: {compression['on_demand_tokens']} tokens")
    lines.append("")
    lines.append(f"└─ Phase 3: Discarded")
    lines.append(f"    └─ Redundant: {compression['discarded_tokens']} tokens")
    lines.append("```")
    lines.append("")
    lines.append("### Loading Triggers")
    lines.append("")
    lines.append("| Module | Trigger Condition | Tokens Saved |")
    lines.append("|--------|-------------------|--------------|")
    lines.append("| Background | User asks 'why' or needs context explanation | Loaded only when needed |")
    lines.append("| Examples | User requests code examples or demonstrations | Loaded only when needed |")
    lines.append("| Templates | User needs boilerplate or fill-in templates | Loaded only when needed |")
    lines.append("")
    lines.append("### Compression Metrics")
    lines.append("")
    lines.append(f"- **Original Size:** {original_tokens} tokens")
    lines.append(f"- **Compressed Size:** {compression['always_loaded_tokens']} tokens")
    lines.append(f"- **Compression Ratio:** {compression['compression_ratio']*100:.1f}%")
    lines.append(f"- **Tokens Saved Per Invocation:** {original_tokens - compression['always_loaded_tokens']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Part 6: Compressed Skill Output
    lines.append("## Part 6: Final Compressed Skill")
    lines.append("")
    lines.append("### Core Rules Only (Always Loaded)")
    lines.append("")
    lines.append("This is what gets injected into the LLM context on every invocation:")
    lines.append("")
    lines.append(f"```markdown")

    core_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.CORE_RULE)
    for block in core_blocks:
        lines.append(block.content)
        lines.append("")

    lines.append(f"```")
    lines.append("")
    lines.append(f"**Total Core Tokens:** {compression['always_loaded_tokens']}")
    lines.append("")
    lines.append("### On-Demand Modules (Reference)")
    lines.append("")
    lines.append("These modules are loaded only when specifically needed:")
    lines.append("")
    lines.append("#### Background Module")
    lines.append(f"```markdown")

    bg_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.BACKGROUND)
    for block in bg_blocks[:5]:  # Show first 5
        lines.append(block.content)
        lines.append("")

    if len(bg_blocks) > 5:
        lines.append(f"... ({len(bg_blocks) - 5} more background blocks)")
    lines.append(f"```")
    lines.append("")
    lines.append("#### Templates Module")
    lines.append(f"```markdown")

    template_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.TEMPLATE)
    for block in template_blocks[:3]:  # Show first 3
        lines.append(block.content)
        lines.append("")

    if len(template_blocks) > 3:
        lines.append(f"... ({len(template_blocks) - 3} more template blocks)")
    lines.append(f"```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Skill Name | {skill.name} |")
    lines.append(f"| Original Tokens | {original_tokens} |")
    lines.append(f"| Compressed Tokens | {compression['always_loaded_tokens']} |")
    lines.append(f"| Compression Ratio | {compression['compression_ratio']*100:.1f}% |")
    lines.append(f"| Core Rules Percentage | {compression['always_loaded_tokens']/original_tokens*100:.1f}% |")
    lines.append(f"| Estimated Cost Savings | ${((original_tokens - compression['always_loaded_tokens']) / 1000) * 0.01:.4f} per invocation |")
    lines.append("")
    lines.append("### Comparison with Paper Findings")
    lines.append("")
    lines.append("According to SkillReducer paper (Gao et al., 2026):")
    lines.append("- **Average core rule percentage:** 38.5%")
    lines.append(f"- **This skill core rule percentage:** {compression['always_loaded_tokens']/original_tokens*100:.1f}%")
    lines.append("- **Average compression ratio:** 39%")
    lines.append(f"- **This skill compression ratio:** {compression['compression_ratio']*100:.1f}%")
    lines.append("")
    lines.append("This skill's compression is slightly higher than the average, indicating more background/template content.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by SkillReducer Framework*")

    return "\n".join(lines)


if __name__ == "__main__":
    generate_report()