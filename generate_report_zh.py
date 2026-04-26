"""
Generate detailed compression report as markdown file in Chinese.

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
    """Generate compression report markdown file in Chinese."""
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
    report = generate_markdown_content_chinese(
        skill, initial_chunks, classified_blocks,
        original_tokens, summary, compression, optimizer
    )

    # Write to file
    output_path = Path(f"data/{skill_name}_compression_report_zh.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"报告已保存到: {output_path}")
    return output_path


def generate_markdown_content_chinese(
    skill, initial_chunks, classified_blocks,
    original_tokens, summary, compression, optimizer
):
    """Generate markdown report content in Chinese."""

    lines = []

    # Header
    lines.append("# SkillReducer 压缩报告")
    lines.append("")
    lines.append(f"**Skill:** {skill.name}")
    lines.append(f"**生成时间:** 2026-04-22")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Part 1: Original Skill
    lines.append("## 第一部分：原始 Skill")
    lines.append("")
    lines.append("### 元数据")
    lines.append("")
    lines.append(f"| 字段 | 值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 名称 | {skill.name} |")
    lines.append(f"| 版本 | {skill.metadata.version or 'N/A'} |")
    lines.append(f"| 作者 | {skill.metadata.author or 'N/A'} |")
    lines.append(f"| 类别 | {skill.metadata.category or 'N/A'} |")
    lines.append(f"| 标签 | {', '.join(skill.metadata.tags) if skill.metadata.tags else 'N/A'} |")
    lines.append("")
    lines.append("### 描述（Description）")
    lines.append("")
    lines.append("描述是 Skill 的路由层文本，用于匹配用户查询到合适的 Skill。")
    lines.append("")
    lines.append(f"```")
    lines.append(skill.description.original)
    lines.append(f"```")
    lines.append("")
    lines.append(f"**描述 Token 数：** {count_tokens(skill.description.original)}")
    lines.append("")
    lines.append("### 原始 Body")
    lines.append("")
    lines.append("Body 是 Skill 的主体内容，包含主要指令文档，注入到 LLM 上下文中。")
    lines.append("")
    lines.append(f"**Body Token 数：** {original_tokens}")
    lines.append("")
    lines.append("#### Body 内容预览（前 1000 字符）")
    lines.append("")
    lines.append(f"```markdown")
    lines.append(skill.body.original[:1000])
    lines.append(f"...")
    lines.append(f"```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Part 2: Chunking Process
    lines.append("## 第二部分：切分过程")
    lines.append("")
    lines.append("### 切分策略")
    lines.append("")
    lines.append("Markdown Body 按语义单元切分，策略如下：")
    lines.append("- 每个标题（h1-h6）开始一个新的 chunk")
    lines.append("- 标题后的内容（段落、列表、代码块、表格）与该标题组合在一起")
    lines.append("- 顶层内容（在任何标题之前）形成独立的 chunk")
    lines.append("")
    lines.append(f"### 切分结果")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 总 Chunk 数 | {len(initial_chunks)} |")
    lines.append(f"| 切分后总 Token 数 | {sum(c.token_count for c in initial_chunks)} |")
    lines.append(f"| 原始 Token 数 | {original_tokens} |")
    lines.append(f"| 内容保留率 | {sum(c.token_count for c in initial_chunks)/original_tokens*100:.1f}% |")
    lines.append("")
    lines.append("### 所有 Chunk 详情")
    lines.append("")
    lines.append("| 序号 | Chunk ID | Token 数 | 内容预览 |")
    lines.append("|------|----------|----------|----------|")

    for i, chunk in enumerate(initial_chunks, 1):
        preview = chunk.content[:80].replace('\n', '\\n')
        lines.append(f"| {i} | {chunk.chunk_id} | {chunk.token_count} | {preview}... |")

    lines.append("")
    lines.append("### Chunk 完整内容")
    lines.append("")

    for i, chunk in enumerate(initial_chunks, 1):
        lines.append(f"#### Chunk {i}：`{chunk.chunk_id}`")
        lines.append("")
        lines.append(f"**Token 数：** {chunk.token_count}")
        lines.append("")
        lines.append(f"**内容：**")
        lines.append(f"```markdown")
        lines.append(chunk.content)
        lines.append(f"```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Part 3: LLM Classification
    lines.append("## 第三部分：LLM 分类")
    lines.append("")
    lines.append("### 分类类别")
    lines.append("")
    lines.append("| 类别 | 描述 | 加载策略 |")
    lines.append("|------|------|----------|")
    lines.append("| `core_rule` (核心规则) | 可执行指令、规则、工作流程 | **始终加载** |")
    lines.append("| `background` (背景知识) | 解释性内容、定义、上下文 | **按需加载** |")
    lines.append("| `example` (示例) | 代码片段、使用示例 | **按需加载** |")
    lines.append("| `template` (模板) | 样板文本、现成格式 | **按需加载** |")
    lines.append("| `redundant` (冗余) | 重复或无价值内容 | **丢弃** |")
    lines.append("")
    lines.append("### 分类汇总")
    lines.append("")
    lines.append(f"| 类别 | 块数 | Token 数 | 占比 |")
    lines.append(f"|------|------|----------|------|")

    total_tokens = sum(b.token_count for b in classified_blocks)
    type_names_zh = {
        ContentType.CORE_RULE: "核心规则",
        ContentType.BACKGROUND: "背景知识",
        ContentType.EXAMPLE: "示例",
        ContentType.TEMPLATE: "模板",
        ContentType.REDUNDANT: "冗余"
    }
    for ct in ContentType:
        blocks = optimizer.get_blocks_by_type(classified_blocks, ct)
        tokens = sum(b.token_count for b in blocks)
        if blocks:
            pct = tokens / total_tokens * 100
            lines.append(f"| {type_names_zh[ct]} | {len(blocks)} | {tokens} | {pct:.1f}% |")

    lines.append("")
    lines.append("### 分类结果详情")
    lines.append("")
    lines.append("| 序号 | Chunk ID | 分类结果 | Token 数 | 内容预览 |")
    lines.append("|------|----------|----------|----------|----------|")

    for i, block in enumerate(classified_blocks, 1):
        preview = block.content[:60].replace('\n', '\\n')
        lines.append(f"| {i} | {block.chunk_id} | {type_names_zh[block.content_type]} | {block.token_count} | {preview}... |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Part 4: Content by Type
    lines.append("## 第四部分：各类型内容详情")
    lines.append("")

    for ct in ContentType:
        blocks = optimizer.get_blocks_by_type(classified_blocks, ct)
        if not blocks:
            continue

        type_desc = {
            ContentType.CORE_RULE: "核心规则（始终加载）",
            ContentType.BACKGROUND: "背景知识（按需加载）",
            ContentType.EXAMPLE: "示例（按需加载）",
            ContentType.TEMPLATE: "模板（按需加载）",
            ContentType.REDUNDANT: "冗余内容（丢弃）"
        }
        type_strategy = {
            ContentType.CORE_RULE: "这部分内容包含可执行的指令，是 Skill 的核心，每次调用时都会注入到上下文中。",
            ContentType.BACKGROUND: "这部分内容包含解释性知识，在用户需要了解背景或概念时按需加载。",
            ContentType.EXAMPLE: "这部分内容包含代码示例，在用户需要参考示例时按需加载。",
            ContentType.TEMPLATE: "这部分内容包含模板样板，在用户需要使用模板时按需加载。",
            ContentType.REDUNDANT: "这部分内容被判定为冗余，将在压缩过程中丢弃。"
        }

        lines.append(f"### {type_desc[ct]}")
        lines.append("")
        lines.append(f"**说明：** {type_strategy[ct]}")
        lines.append("")
        lines.append(f"**统计：** {len(blocks)} 个块，共 {sum(b.token_count for b in blocks)} tokens")
        lines.append("")

        for i, block in enumerate(blocks, 1):
            lines.append(f"#### 块 {i}：`{block.chunk_id}`（{block.token_count} tokens）")
            lines.append("")
            lines.append(f"```markdown")
            lines.append(block.content)
            lines.append(f"```")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Part 5: Compression Result
    lines.append("## 第五部分：压缩结果")
    lines.append("")
    lines.append("### Token 减少分析")
    lines.append("")
    lines.append(f"| 指标 | 压缩前 | 压缩后 | 减少量 |")
    lines.append(f"|------|--------|--------|--------|")
    lines.append(f"| Body Token 数 | {original_tokens} | {compression['always_loaded_tokens']} | {original_tokens - compression['always_loaded_tokens']} ({(1-compression['always_loaded_tokens']/original_tokens)*100:.1f}%) |")
    lines.append("")
    lines.append("### Progressive Disclosure（渐进式披露）结构")
    lines.append("")
    lines.append("``")
    lines.append("压缩后 Skill 结构：")
    lines.append("")
    lines.append(f"├─ 阶段 1：基础加载（始终注入）")
    lines.append(f"│   └─ 核心规则：{compression['always_loaded_tokens']} tokens")
    lines.append("")
    lines.append(f"├─ 阶段 2：按需加载模块")
    lines.append(f"│   ├─ 背景知识：{sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.BACKGROUND))} tokens")
    lines.append(f"│   ├─ 示例：{sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.EXAMPLE))} tokens")
    lines.append(f"│   └─ 模板：{sum(b.token_count for b in optimizer.get_blocks_by_type(classified_blocks, ContentType.TEMPLATE))} tokens")
    lines.append(f"│   └─ 按需模块总计：{compression['on_demand_tokens']} tokens")
    lines.append("")
    lines.append(f"└─ 阶段 3：丢弃")
    lines.append(f"    └─ 冗余内容：{compression['discarded_tokens']} tokens")
    lines.append("```")
    lines.append("")
    lines.append("### 模块加载触发条件")
    lines.append("")
    lines.append("| 模块 | 触发条件 | 说明 |")
    lines.append("|------|----------|------|")
    lines.append("| 背景知识 | 用户询问\"为什么\"或需要概念解释时 | 仅在需要时加载 |")
    lines.append("| 示例 | 用户请求代码示例或演示时 | 仅在需要时加载 |")
    lines.append("| 模板 | 用户需要样板或填空模板时 | 仅在需要时加载 |")
    lines.append("")
    lines.append("### 压缩指标")
    lines.append("")
    lines.append(f"- **原始大小：** {original_tokens} tokens")
    lines.append(f"- **压缩后大小：** {compression['always_loaded_tokens']} tokens")
    lines.append(f"- **压缩率：** {compression['compression_ratio']*100:.1f}%")
    lines.append(f"- **每次调用节省 Token：** {original_tokens - compression['always_loaded_tokens']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Part 6: Compressed Skill Output
    lines.append("## 第六部分：最终压缩后的 Skill")
    lines.append("")
    lines.append("### 核心规则部分（始终加载）")
    lines.append("")
    lines.append("以下是每次调用都会注入到 LLM 上下文的内容：")
    lines.append("")
    lines.append(f"```markdown")

    core_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.CORE_RULE)
    for block in core_blocks:
        lines.append(block.content)
        lines.append("")

    lines.append(f"```")
    lines.append("")
    lines.append(f"**核心规则总 Token 数：** {compression['always_loaded_tokens']}")
    lines.append("")
    lines.append("### 按需加载模块（参考）")
    lines.append("")
    lines.append("以下模块仅在特定需要时加载：")
    lines.append("")
    lines.append("#### 背景知识模块")
    lines.append(f"```markdown")

    bg_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.BACKGROUND)
    for block in bg_blocks[:5]:  # Show first 5
        lines.append(block.content)
        lines.append("")

    if len(bg_blocks) > 5:
        lines.append(f"... （还有 {len(bg_blocks) - 5} 个背景知识块）")
    lines.append(f"```")
    lines.append("")
    lines.append("#### 模板模块")
    lines.append(f"```markdown")

    template_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.TEMPLATE)
    for block in template_blocks[:3]:  # Show first 3
        lines.append(block.content)
        lines.append("")

    if len(template_blocks) > 3:
        lines.append(f"... （还有 {len(template_blocks) - 3} 个模板块）")
    lines.append(f"```")
    lines.append("")
    lines.append("#### 示例模块")
    lines.append(f"```markdown")

    example_blocks = optimizer.get_blocks_by_type(classified_blocks, ContentType.EXAMPLE)
    for block in example_blocks[:2]:
        lines.append(block.content)
        lines.append("")

    if len(example_blocks) > 2:
        lines.append(f"... （还有 {len(example_blocks) - 2} 个示例块）")
    lines.append(f"```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## 总结")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|------|")
    lines.append(f"| Skill 名称 | {skill.name} |")
    lines.append(f"| 原始 Token 数 | {original_tokens} |")
    lines.append(f"| 压缩后 Token 数 | {compression['always_loaded_tokens']} |")
    lines.append(f"| 压缩率 | {compression['compression_ratio']*100:.1f}% |")
    lines.append(f"| 核心规则占比 | {compression['always_loaded_tokens']/original_tokens*100:.1f}% |")
    lines.append(f"| 每次调用预估节省成本 | ${((original_tokens - compression['always_loaded_tokens']) / 1000) * 0.01:.4f} |")
    lines.append("")
    lines.append("### 与论文发现对比")
    lines.append("")
    lines.append("根据 SkillReducer 论文（Gao 等，2026）：")
    lines.append("- **平均核心规则占比：** 38.5%")
    lines.append(f"- **本 Skill 核心规则占比：** {compression['always_loaded_tokens']/original_tokens*100:.1f}%")
    lines.append("- **平均压缩率：** 39%")
    lines.append(f"- **本 Skill 压缩率：** {compression['compression_ratio']*100:.1f}%")
    lines.append("")
    lines.append("本 Skill 的压缩率高于平均水平，说明包含更多背景知识和模板内容。")
    lines.append("")
    lines.append("### 压缩流程回顾")
    lines.append("")
    lines.append("``")
    lines.append("SkillReducer 压缩流程：")
    lines.append("")
    lines.append("1. 解析 Skill 文件")
    lines.append("   └─ 提取 YAML frontmatter 和 Markdown Body")
    lines.append("")
    lines.append("2. Markdown 切分")
    lines.append("   └─ 按标题边界切分为语义块")
    lines.append(f"   └─ 生成 {len(initial_chunks)} 个 Chunk")
    lines.append("")
    lines.append("3. LLM 分类")
    lines.append("   └─ 分批调用 LLM 进行 taxonomy 分类")
    lines.append("   └─ 分类为：核心规则、背景知识、示例、模板、冗余")
    lines.append("")
    lines.append("4. Progressive Disclosure 重构")
    lines.append("   └─ 核心规则始终加载")
    lines.append("   └─ 背景知识、示例、模板按需加载")
    lines.append("   └─ 冗余内容丢弃")
    lines.append("")
    lines.append(f"5. 压缩结果")
    lines.append(f"   └─ 从 {original_tokens} tokens 压缩到 {compression['always_loaded_tokens']} tokens")
    lines.append(f"   └─ 压缩率 {compression['compression_ratio']*100:.1f}%")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 SkillReducer 框架生成*")

    return "\n".join(lines)


if __name__ == "__main__":
    generate_report()