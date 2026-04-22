"""
Detailed demonstration of SkillReducer compression pipeline.

Shows step-by-step how a skill is compressed, including:
1. Original skill content
2. Chunking process with each chunk's content
3. LLM classification results with reasoning
4. Final compressed skill structure
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunker import chunk_markdown_body, count_tokens
from src.llm_client import SkillLLMClient, BlockClassification
from src.models import ContentType, Skill, Description, Body
from src.optimizer import Stage2Optimizer
from src.parser import parse_skill_file


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}\n")


def truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text.replace('\n', '\\n')
    return text[:max_len].replace('\n', '\\n') + "..."


def main():
    """Run detailed compression demonstration."""
    load_dotenv()

    print_section("SKILLREDUCER 详细压缩过程演示")

    # Step 1: Load original skill
    print_section("步骤 1: 加载原始 Skill")

    skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
    skill = parse_skill_file(skill_path)

    print(f"Skill 名称: {skill.name}")
    print(f"Skill 描述: {skill.description.original}")
    print(f"\n元数据:")
    print(f"  - 版本: {skill.metadata.version}")
    print(f"  - 作者: {skill.metadata.author}")
    print(f"  - 类别: {skill.metadata.category}")
    print(f"  - 标签: {skill.metadata.tags}")

    print_subsection("原始 Body 内容 (前 500 字符)")
    print(truncate_text(skill.body.original, 500))
    print(f"\n原始 Body Token 数: {count_tokens(skill.body.original)}")

    # Step 2: Chunk the body
    print_section("步骤 2: Markdown 切分")

    chunks = chunk_markdown_body(skill.body.original)

    print(f"切分结果: 共 {len(chunks)} 个文本块")
    print(f"切分后总 Token 数: {sum(c.token_count for c in chunks)}")

    print_subsection("每个 Chunk 详情")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}] ID: {chunk.chunk_id}")
        print(f"  Token 数: {chunk.token_count}")
        print(f"  默认类型: {chunk.content_type}")
        print(f"  内容预览:")
        content_preview = truncate_text(chunk.content, 150)
        print(f"    {content_preview}")

    # Step 3: Initialize LLM and classify
    print_section("步骤 3: LLM 分类")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    print(f"LLM 配置:")
    print(f"  - Base URL: {base_url}")
    print(f"  - Model: deepseek-chat")

    client = SkillLLMClient(api_key=api_key, base_url=base_url, model="deepseek-chat")
    optimizer = Stage2Optimizer(llm_client=client, batch_size=10)

    print_subsection("分批调用 LLM")

    # Show the batching process
    print(f"将 {len(chunks)} 个 chunks 分为 {len(chunks)//10 + 1} 批处理")
    print(f"每批最多 10 个 chunks，防止上下文溢出")

    # Actually classify
    classified_blocks = optimizer.classify_skill_body(skill)

    print_subsection("LLM 分类结果详情")

    for i, block in enumerate(classified_blocks, 1):
        print(f"\n[Block {i}] {block.chunk_id}")
        print(f"  分类结果: {block.content_type}")
        print(f"  Token 数: {block.token_count}")
        content_preview = truncate_text(block.content, 100)
        print(f"  内容: {content_preview}")

    # Step 4: Show classification summary
    print_section("步骤 4: 分类汇总")

    summary = optimizer.get_classification_summary(classified_blocks)

    print("按类型统计:")
    print(f"\n{'类型':<20} {'块数':<10} {'Token数':<10} {'占比'}")
    print("-" * 60)

    total_tokens = sum(b.token_count for b in classified_blocks)

    for ct in ContentType:
        blocks = optimizer.get_blocks_by_type(classified_blocks, ct)
        tokens = sum(b.token_count for b in blocks)
        if blocks:
            pct = tokens / total_tokens * 100
            print(f"{ct:<20} {len(blocks):<10} {tokens:<10} {pct:.1f}%")

    # Step 5: Show sample content for each type
    print_section("步骤 5: 各类型内容示例")

    for ct in ContentType:
        blocks = optimizer.get_blocks_by_type(classified_blocks, ct)
        if not blocks:
            continue

        print_subsection(f"{ct} 类型内容")

        # Show meaning
        type_meanings = {
            ContentType.CORE_RULE: "→ 始终加载到上下文中，包含可执行指令",
            ContentType.BACKGROUND: "→ 按需加载，包含解释性内容",
            ContentType.EXAMPLE: "→ 按需加载，包含代码示例",
            ContentType.TEMPLATE: "→ 按需加载，包含模板样板",
            ContentType.REDUNDANT: "→ 丢弃，冗余内容"
        }
        print(f"处理策略: {type_meanings[ct]}")
        print(f"包含 {len(blocks)} 个块，共 {sum(b.token_count for b in blocks)} tokens")

        # Show first 2 blocks of each type
        for j, block in enumerate(blocks[:2], 1):
            print(f"\n  示例 {j}: [{block.chunk_id}] ({block.token_count} tokens)")
            # Show more content for examples
            if ct in [ContentType.CORE_RULE, ContentType.EXAMPLE]:
                print(f"  内容:")
                for line in block.content.split('\n')[:5]:
                    print(f"    {line}")
            else:
                preview = truncate_text(block.content, 150)
                print(f"  内容: {preview}")

    # Step 6: Compression result
    print_section("步骤 6: 最终压缩结果")

    compression = optimizer.calculate_compression_potential(classified_blocks)

    print("压缩前:")
    print(f"  原始 Body: {count_tokens(skill.body.original)} tokens")

    print("\n压缩后结构:")
    print(f"  ├─ 核心规则 (始终加载): {compression['always_loaded_tokens']} tokens")
    print(f"  ├─ 按需模块:")
    print(f"  │   ├─ 背景知识: {sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.BACKGROUND))} tokens")
    print(f"  │   ├─ 代码示例: {sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.EXAMPLE))} tokens")
    print(f"  │   └─ 模板样板: {sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.TEMPLATE))} tokens")
    print(f"  └─ 丢弃内容: {compression['discarded_tokens']} tokens")

    print(f"\n压缩效果:")
    original = count_tokens(skill.body.original)
    compressed = compression['always_loaded_tokens']
    print(f"  Token 减少: {original - compressed} tokens")
    print(f"  压缩率: {(1 - compressed/original)*100:.1f}%")
    print(f"  保留核心指令: {compressed/original*100:.1f}%")

    # Step 7: Show what would be loaded vs on-demand
    print_section("步骤 7: Progressive Disclosure 结构")

    print_subsection("阶段 1: 基础加载 (Core Rules)")
    core_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.CORE_RULE)
    print(f"加载 {len(core_blocks)} 个核心规则块，共 {sum(b.token_count for b in core_blocks)} tokens")
    print("\n加载内容:")
    for block in core_blocks[:5]:
        preview = truncate_text(block.content, 80)
        print(f"  • {preview}")

    print_subsection("阶段 2: 按需加载 (Background, Examples, Templates)")
    on_demand_blocks = [
        *optimizer.get_blocks_by_type(classified_blocks, ContentType.BACKGROUND),
        *optimizer.get_blocks_by_type(classified_blocks, ContentType.EXAMPLE),
        *optimizer.get_blocks_by_type(classified_blocks, ContentType.TEMPLATE),
    ]
    print(f"可选 {len(on_demand_blocks)} 个按需模块，共 {sum(b.token_count for b in on_demand_blocks)} tokens")
    print("\n按需模块触发条件:")
    print("  • Background: 当用户需要了解概念或背景时加载")
    print("  • Examples: 当用户需要代码示例时加载")
    print("  • Templates: 当用户需要模板或样板时加载")

    print_subsection("阶段 3: 丢弃 (Redundant)")
    redundant_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.REDUNDANT)
    if redundant_blocks:
        print(f"丢弃 {len(redundant_blocks)} 个冗余块，共 {sum(b.token_count for b in redundant_blocks)} tokens")
        for block in redundant_blocks:
            print(f"  ✗ [{block.chunk_id}]: {truncate_text(block.content, 100)}")
    else:
        print("无需丢弃任何内容")

    # Final summary
    print_section("压缩总结")

    print(f"原始 Skill: {skill.name}")
    print(f"  描述: {skill.description.original[:80]}...")
    print(f"  Body: {original} tokens")
    print(f"\n压缩后 Skill 结构:")
    print(f"  描述: 保持不变 (Stage 1 处理)")
    print(f"  Body 核心部分: {compressed} tokens")
    print(f"  Body 按需部分: {sum(b.token_count for b in on_demand_blocks)} tokens")
    print(f"\n实际效果:")
    print(f"  每次调用节省 {original - compressed} tokens")
    print(f"  按论文估算，每次 API 调用可节省约 ${((original - compressed) / 1000) * 0.01:.4f}")
    print(f"  (假设 GPT-4 输入价格 $0.01/1K tokens)")

    print_section("演示完成")


if __name__ == "__main__":
    main()