#!/usr/bin/env python3
"""
Batch compression script for Claude-Skills library.

Usage:
    python batch_compress.py [--limit N] [--output-dir DIR] [--skip-errors]

Examples:
    # Compress all 245 skills
    python batch_compress.py

    # Compress only first 10 skills (for testing)
    python batch_compress.py --limit 10

    # Custom output directory
    python batch_compress.py --output-dir ./compressed-skills
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.parser import parse_skill_file
from src.llm_client import SkillLLMClient
from src.quality_gates import run_pipeline, PipelineResult


def find_all_skills(base_dir: str = "Claude-Skills") -> list[str]:
    """Find all SKILL.md files in the directory."""
    skills = []
    for root, dirs, files in os.walk(base_dir):
        if "SKILL.md" in files:
            skills.append(os.path.join(root, "SKILL.md"))
    return sorted(skills)


def init_llm_client() -> SkillLLMClient:
    """Initialize LLM client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Auto-detect model based on base_url
    default_model = "gpt-4o-mini"
    if "deepseek" in base_url.lower():
        default_model = "deepseek-v4-flash"
    elif "qwen" in base_url.lower() or "dashscope" in base_url.lower():
        default_model = "qwen-plus"

    model = os.getenv("LLM_MODEL", default_model)

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return SkillLLMClient(api_key=api_key, base_url=base_url, model=model)


def compress_skill(
    skill_path: str,
    client: SkillLLMClient,
    enable_gate1: bool = True,
    enable_gate2: bool = False
) -> Optional[PipelineResult]:
    """Compress a single skill file."""
    try:
        skill = parse_skill_file(skill_path)
        result = run_pipeline(
            skill,
            client,
            enable_gate1=enable_gate1,
            enable_gate2=enable_gate2
        )
        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def save_compressed_skill(result: PipelineResult, output_dir: str, original_path: str):
    """
    Save compressed skill to output directory.

    Implements Progressive Disclosure architecture:
    - SKILL.md: Contains ONLY core rules (always loaded)
    - references/on-demand-*.md: Contains examples, templates, background (loaded on demand)
    """
    # Preserve directory structure
    rel_path = os.path.relpath(original_path, "Claude-Skills")
    output_path = os.path.join(output_dir, rel_path)
    skill_dir = os.path.dirname(output_path)

    # Create directories
    os.makedirs(skill_dir, exist_ok=True)
    os.makedirs(os.path.join(skill_dir, "references"), exist_ok=True)

    # Generate compressed SKILL.md content
    skill = result.skill

    # Build YAML frontmatter
    frontmatter = f"""---
name: {skill.name}
description: |
  {skill.description.compressed or skill.description.original}
metadata:
  version: {skill.metadata.version or '1.0.0'}
  category: {skill.metadata.category or ''}
  tags: {json.dumps(skill.metadata.tags)}
---

"""

    # Build body from content blocks (ONLY core rules)
    body_parts = []
    for block in skill.body.content_blocks:
        body_parts.append(block.content)

    body_content = "\n\n".join(body_parts)

    # Add reference to on-demand modules in body
    on_demand_note = """

---
## On-Demand Modules

The following modules are available in the `references/` directory and can be loaded when needed:
- `references/on-demand-examples.md` - Code examples and usage demonstrations
- `references/on-demand-templates.md` - Ready-to-use templates and boilerplate
- `references/on-demand-background.md` - Background knowledge and explanations

Use the `read_file` tool to load these when the user asks for examples, templates, or explanations.
"""
    body_content += on_demand_note

    # Write main SKILL.md
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter + body_content)

    # Save on-demand modules as reference files
    for ref_name, ref_content in skill.references.files.items():
        ref_path = os.path.join(skill_dir, "references", ref_name)
        with open(ref_path, 'w', encoding='utf-8') as f:
            f.write(f"# {ref_name.replace('.md', '').replace('-', ' ').title()}\n\n")
            f.write("This file is loaded on-demand when needed.\n\n---\n\n")
            f.write(ref_content)

    return output_path


def generate_report(results: list[dict], output_dir: str):
    """Generate compression report."""
    report_path = os.path.join(output_dir, "compression_report.md")

    total_skills = len(results)
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    # Calculate aggregate metrics
    total_original = sum(r.get("original_tokens", 0) for r in successful)
    total_final = sum(r.get("final_tokens", 0) for r in successful)

    avg_compression = 0
    if total_original > 0:
        avg_compression = 1 - (total_final / total_original)

    # Stage metrics
    stage1_success = sum(1 for r in successful if r.get("stage1_compressed"))
    stage2_success = sum(1 for r in successful if r.get("stage2_compressed"))
    gate1_failures = sum(1 for r in successful if r.get("rollback_performed"))

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# SkillReducer Compression Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Skills | {total_skills} |\n")
        f.write(f"| Successful | {len(successful)} |\n")
        f.write(f"| Failed | {len(failed)} |\n")
        f.write(f"| Stage 1 Compressed | {stage1_success} |\n")
        f.write(f"| Stage 2 Compressed | {stage2_success} |\n")
        f.write(f"| Gate 1 Rollbacks | {gate1_failures} |\n")
        f.write(f"| **Total Original Tokens** | {total_original:,} |\n")
        f.write(f"| **Total Final Tokens** | {total_final:,} |\n")
        f.write(f"| **Overall Compression** | {avg_compression:.2%} |\n")

        f.write("\n## Per-Skill Results\n\n")
        f.write("| Skill | Original | Final | Compression | Gate1 |\n")
        f.write("|-------|----------|-------|-------------|-------|\n")

        for r in successful:
            skill_name = r.get("skill_name", "unknown")
            orig = r.get("original_tokens", 0)
            final = r.get("final_tokens", 0)
            ratio = r.get("compression_ratio", 0)
            gate1 = "PASS" if not r.get("rollback_performed") else "ROLLBACK"
            f.write(f"| {skill_name} | {orig:,} | {final:,} | {ratio:.2%} | {gate1} |\n")

        if failed:
            f.write("\n## Failed Skills\n\n")
            for r in failed:
                f.write(f"- {r.get('skill_path', 'unknown')}: {r.get('error', 'unknown error')}\n")

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Batch compress Claude-Skills")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of skills to process")
    parser.add_argument("--output-dir", type=str, default="compressed-skills", help="Output directory")
    parser.add_argument("--skip-errors", action="store_true", help="Continue on errors")
    parser.add_argument("--no-gate1", action="store_true", help="Disable Gate 1 (faithfulness check)")
    parser.add_argument("--start-from", type=int, default=0, help="Start from N-th skill (for resuming)")
    args = parser.parse_args()

    # Find all skills
    print("Finding skills...")
    skill_paths = find_all_skills()
    print(f"Found {len(skill_paths)} skills")

    if args.limit:
        skill_paths = skill_paths[:args.limit]
        print(f"Processing first {args.limit} skills")

    if args.start_from > 0:
        skill_paths = skill_paths[args.start_from:]
        print(f"Starting from skill #{args.start_from + 1}")

    # Initialize client
    print("\nInitializing LLM client...")
    try:
        client = init_llm_client()
        print(f"Model: {client.model}")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set OPENAI_API_KEY in .env file or environment variables")
        sys.exit(1)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process skills
    results = []
    print(f"\nProcessing {len(skill_paths)} skills...\n")

    for i, skill_path in enumerate(skill_paths):
        print(f"[{i+1}/{len(skill_paths)}] {skill_path}")

        result = compress_skill(
            skill_path,
            client,
            enable_gate1=not args.no_gate1,
            enable_gate2=False
        )

        if result:
            # Save compressed skill
            output_path = save_compressed_skill(result, output_dir, skill_path)

            results.append({
                "success": True,
                "skill_path": skill_path,
                "skill_name": result.skill.name,
                "original_tokens": result.original_tokens,
                "final_tokens": result.final_tokens,
                "compression_ratio": result.overall_compression_ratio,
                "stage1_compressed": result.stage1_compressed,
                "stage2_compressed": result.stage2_compressed,
                "rollback_performed": result.rollback_performed,
                "errors": result.errors
            })

            print(f"  Original: {result.original_tokens:,} tokens")
            print(f"  Final: {result.final_tokens:,} tokens")
            print(f"  Compression: {result.overall_compression_ratio:.2%}")
            if result.rollback_performed:
                print(f"  ⚠️ Gate 1 rollback performed")
        else:
            results.append({
                "success": False,
                "skill_path": skill_path,
                "error": "Compression failed"
            })

            if not args.skip_errors:
                print("\nStopping due to error. Use --skip-errors to continue.")
                break

        print()

    # Generate report
    print("Generating report...")
    report_path = generate_report(results, output_dir)

    # Summary
    successful = [r for r in results if r.get("success")]
    print(f"\n{'='*50}")
    print(f"Compression Complete")
    print(f"{'='*50}")
    print(f"Processed: {len(results)} skills")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")

    if successful:
        total_orig = sum(r["original_tokens"] for r in successful)
        total_final = sum(r["final_tokens"] for r in successful)
        print(f"\nTotal tokens: {total_orig:,} → {total_final:,}")
        print(f"Overall compression: {1 - total_final/total_orig:.2%}")

    print(f"\nOutput saved to: {output_dir}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
