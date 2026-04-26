"""
Demonstration of SkillReducer compression pipeline.

Shows the complete Stage 2 body restructuring process with real LLM API.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunker import chunk_markdown_body, count_tokens
from src.llm_client import SkillLLMClient
from src.models import ContentType
from src.optimizer import Stage2Optimizer
from src.parser import parse_skill_file


def main():
    """Run compression demonstration."""
    # Load environment variables
    load_dotenv()

    print("=" * 70)
    print("SkillReducer Compression Demonstration")
    print("=" * 70)

    # Initialize LLM client with DeepSeek
    print("\n[1] Initializing LLM Client (DeepSeek)")
    print("-" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        return

    print(f"Base URL: {base_url}")
    print(f"Model: deepseek-v4-flash")

    client = SkillLLMClient(
        api_key=api_key,
        base_url=base_url,
        model="deepseek-v4-flash"
    )

    # Test connection
    print("Testing connection...")
    if client.test_connection():
        print("✓ Connection successful")
    else:
        print("✗ Connection failed")
        return

    # Parse skill file
    print("\n[2] Parsing Skill File")
    print("-" * 70)

    skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
    skill = parse_skill_file(skill_path)

    print(f"Skill Name: {skill.name}")
    print(f"Description: {skill.description.original[:100]}...")
    print(f"Original Body Tokens: {count_tokens(skill.body.original)}")

    # Chunk the body
    print("\n[3] Chunking Body Content")
    print("-" * 70)

    initial_blocks = chunk_markdown_body(skill.body.original)
    print(f"Total Chunks: {len(initial_blocks)}")
    print(f"Total Chunked Tokens: {sum(b.token_count for b in initial_blocks)}")

    # Initialize optimizer and classify
    print("\n[4] Classifying Content Blocks with LLM")
    print("-" * 70)

    optimizer = Stage2Optimizer(llm_client=client, batch_size=10)
    classified_blocks = optimizer.classify_skill_body(skill)

    # Get classification summary
    print("\n[5] Classification Results")
    print("-" * 70)

    summary = optimizer.get_classification_summary(classified_blocks)

    print("\nContent Type Distribution:")
    print(f"{'Type':<15} {'Count':<10} {'Tokens':<10} {'Percentage'}")
    print("-" * 50)

    type_tokens = {}
    for ct in ContentType:
        blocks_of_type = optimizer.get_blocks_by_type(classified_blocks, ct)
        tokens = sum(b.token_count for b in blocks_of_type)
        type_tokens[ct] = tokens
        if ct in summary:
            pct = tokens / sum(b.token_count for b in classified_blocks) * 100
            print(f"{ct.value:<15} {summary[ct]:<10} {tokens:<10} {pct:.1f}%")

    # Calculate compression potential
    print("\n[6] Compression Potential Analysis")
    print("-" * 70)

    compression = optimizer.calculate_compression_potential(classified_blocks)

    total = compression["total_tokens"]
    core = compression["always_loaded_tokens"]
    on_demand = compression["on_demand_tokens"]
    discarded = compression["discarded_tokens"]

    print(f"\nTotal Tokens: {total}")
    print(f"  ├─ Always Loaded (Core Rules): {core} ({core/total*100:.1f}%)")
    print(f"  ├─ On-Demand (Background, Examples, Templates): {on_demand} ({on_demand/total*100:.1f}%)")
    print(f"  └─ Discarded (Redundant): {discarded} ({discarded/total*100:.1f}%)")

    print(f"\nEstimated Compression Ratio: {compression['compression_ratio']*100:.1f}%")
    print(f"  → Reduces context from {total} to ~{core} tokens (core only)")
    print(f"  → Additional {on_demand} tokens available on-demand")

    # Show sample blocks by type
    print("\n[7] Sample Blocks by Content Type")
    print("-" * 70)

    for ct in [ContentType.CORE_RULE, ContentType.BACKGROUND, ContentType.EXAMPLE, ContentType.TEMPLATE]:
        blocks = optimizer.get_blocks_by_type(classified_blocks, ct)
        if blocks:
            print(f"\n--- {ct.value.upper()} (showing first 2) ---")
            for i, block in enumerate(blocks[:2], 1):
                preview = block.content[:80].replace('\n', ' ')
                print(f"  [{block.chunk_id}] {preview}...")
                print(f"      Tokens: {block.token_count}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Skill: {skill.name}")
    print(f"Original: {count_tokens(skill.body.original)} tokens")
    print(f"After Stage 2 restructuring:")
    print(f"  - Core (always loaded): {core} tokens")
    print(f"  - Compression achieved: {(1 - core/total)*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()