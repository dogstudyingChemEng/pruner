# SkillReducer

## 项目简介

SkillReducer 通过两阶段流水线对 LLM Agent 技能进行压缩优化：

**阶段一：路由层优化 (Routing Layer Optimization)**
- 语义分割：将描述拆分为语义子句
- 模拟 Oracle (O_sim)：生成对抗技能测试路由等价性
- DDMIN 算法：Delta Debugging 找到 1-minimal 子集
- 重写润色：将最小子句集合并为流畅描述

**阶段二：主体重构 (Body Restructuring via Progressive Disclosure)**
- 分类驱动的五类内容分类：
  - `core_rule`：核心规则（可操作指令）— 始终加载
  - `background`：背景说明（解释性内容）— 按需加载模块
  - `example`：示例（代码片段）— 按需加载模块
  - `template`：模板（样板文本）— 按需加载模块
  - `redundant`：冗余内容 — 丢弃
- 各类型压缩处理：核心规则合并、示例去重、模板去重、背景总结
- 跨文件去重：主体与引用文件之间的去重
- 质量门控：忠实性检查 + 任务评估反馈循环

**质量门控 (Quality Gates)**
- Gate 1：忠实性验证 — 检查核心操作概念是否保留
- Gate 2：反馈循环 — 基于任务失败提升相关块为核心规则
- 自动回滚：当 Gate 1 失败时回滚到未压缩版本

### 关键指标

| 指标 | 说明 | 预期效果 |
|------|------|----------|
| 压缩率 | Token 减少百分比 | 描述 ~48%，主体 ~39% |
| 保留率 | score_C / score_A | 86% 通过率 |
| 跨模型迁移 | 不同模型间的保留率 | 0.965 均值 |

## 项目结构

```
skillpruner/
├── src/                        # 核心源代码
│   ├── __init__.py
│   ├── models.py              # 数据模型 (Skill, ContentBlock, ContentType 等)
│   ├── parser.py              # 技能文件解析 (SKILL.md, YAML frontmatter)
│   ├── chunker.py             # Markdown 语义分块
│   ├── llm_client.py          # LLM API 客户端 (支持 OpenAI/DeepSeek/Qwen)
│   ├── stage1_router.py       # 阶段一路由优化器 (NEW)
│   ├── optimizer.py           # 阶段二主体优化器
│   └── quality_gates.py       # 质量门控实现 (NEW)
│
├── tests/                      # 单元测试
│   ├── test_parser.py         # 解析器测试
│   ├── test_chunker.py        # 分块器测试
│   ├── test_optimizer.py      # 阶段二优化器测试
│   ├── test_stage1.py         # 阶段一路由器测试 (NEW)
│   └── test_quality_gates.py  # 质量门控测试 (NEW)
│
├── data/                       # 输入/输出数据目录
│
├── Claude-Skills/              # 245 个生产级技能包 (评估测试数据)
│   ├── engineering/           # 76 个工程技能
│   ├── marketing/             # 38 个营销技能
│   ├── c-level-advisor/       # 26 个 C 级顾问技能
│   ├── hr-operations/         # HR 运营技能
│   ├── finance/               # 财务技能
│   ├── legal/                 # 法律技能
│   └── ...                    # 其他领域
│
├── demo_compression.py         # 压缩演示脚本
├── demo_detailed.py            # 详细过程演示脚本
├── generate_report.py          # 英文报告生成脚本
├── generate_report_zh.py       # 中文报告生成脚本
│
├── requirement.txt             # Python 依赖
├── .env.example               # 环境变量示例
├── CLAUDE.md                  # Claude Code 开发指南
└── README.md                  # 本文件
```

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/skillpruner.git
cd skillpruner
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
pip install -r requirement.txt
```

### 4. 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API 配置：

```env
# OpenAI API (默认)
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 或使用 DeepSeek
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com/v1

# 或使用 Qwen
OPENAI_API_KEY=sk-your-qwen-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 快速开始

### 运行单元测试

```bash
# 运行所有测试 (61 个测试)
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_parser.py -v
pytest tests/test_chunker.py -v
pytest tests/test_optimizer.py -v
pytest tests/test_stage1.py -v
pytest tests/test_quality_gates.py -v

# 带覆盖率报告
pytest tests/ --cov=src
```

### 使用完整压缩流水线

```python
from src.parser import parse_skill_file
from src.llm_client import SkillLLMClient
from src.quality_gates import run_pipeline

# 1. 解析技能文件
skill = parse_skill_file("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")

# 2. 初始化 LLM 客户端
client = SkillLLMClient(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)

# 3. 运行完整压缩流水线
result = run_pipeline(
    skill,
    llm_client=client,
    enable_gate1=True,    # 启用忠实性验证
    enable_gate2=False    # 禁用反馈循环
)

# 4. 查看结果
print(f"阶段一压缩: {result.stage1_compressed}")
print(f"阶段二压缩: {result.stage2_compressed}")
print(f"忠实性通过: {result.faithfulness_passed}")
print(f"回滚执行: {result.rollback_performed}")
print(f"原始 Token: {result.original_tokens}")
print(f"最终 Token: {result.final_tokens}")
print(f"总压缩率: {result.overall_compression_ratio:.2%}")

# 5. 获取压缩后的技能
compressed_skill = result.skill
```

### 分阶段使用

```python
from src.parser import parse_skill_file
from src.llm_client import SkillLLMClient
from src.stage1_router import Stage1Optimizer
from src.optimizer import Stage2Optimizer
from src.quality_gates import QualityGates

# 初始化
client = SkillLLMClient(api_key="your-key", model="gpt-4o-mini")
skill = parse_skill_file("path/to/SKILL.md")

# 阶段一：描述压缩
stage1 = Stage1Optimizer(client)
skill = stage1.compress_skill(skill, use_oracle_validation=True)
print(f"描述压缩: {skill.description.original} -> {skill.description.compressed}")

# 阶段二：主体重构
stage2 = Stage2Optimizer(client)
blocks, refs, metrics = stage2.optimize_skill(
    skill,
    compress_core=True,
    dedup_examples=True,
    dedup_templates=True,
    summarize_background=True,
    dedup_references=True
)
print(f"核心压缩率: {metrics.core_compression_ratio:.2%}")

# Gate 1：忠实性验证
gates = QualityGates(client)
faith_result = gates.run_faithfulness_gate(skill, blocks)
if faith_result.should_rollback:
    print("忠实性验证失败，需要回滚")
    blocks = stage2.classify_skill_body(skill)  # 回滚到原始分类
```

### 运行演示脚本

```bash
# 基本压缩演示
python demo_compression.py

# 详细过程演示
python demo_detailed.py

# 生成英文报告
python generate_report.py

# 生成中文报告
python generate_report_zh.py
```

## 核心模块说明

### models.py - 数据模型

定义核心数据结构：

```python
from src.models import Skill, ContentBlock, ContentType, Description, Body, References

# 内容类型枚举
class ContentType(str, Enum):
    CORE_RULE = "core_rule"      # 核心规则
    BACKGROUND = "background"    # 背景说明
    EXAMPLE = "example"          # 示例
    TEMPLATE = "template"        # 模板
    REDUNDANT = "redundant"      # 冗余内容

# 内容块 (支持 use_enum_values 配置)
class ContentBlock(BaseModel):
    chunk_id: str           # 块 ID
    content: str            # 内容文本
    content_type: ContentType  # 分类类型 (存储为字符串)
    token_count: int        # Token 数量

# 完整技能
class Skill(BaseModel):
    name: str               # 技能名称
    description: Description  # 技能描述 (含压缩版本)
    body: Body              # 技能主体
    references: References  # 引用文件
    metadata: SkillMetadata # 元数据
```

### stage1_router.py - 阶段一路由优化

描述压缩实现：

```python
from src.stage1_router import Stage1Optimizer, CompressionResult

optimizer = Stage1Optimizer(llm_client=client)

# 语义分割
clauses = optimizer.segment_description(description)

# 生成对抗技能
adversarial = optimizer.generate_adversarial_skill(skill)

# 路由测试
result = optimizer.test_routing(
    compressed_description="压缩后的描述",
    target_skill=skill,
    adversarial_skills=adversarial
)

# DDMIN 算法
minimal_clauses = optimizer.ddmin(clauses, test_function)

# 重写润色
polished = optimizer.rewrite_and_polish(minimal_clauses)

# 完整压缩
result = optimizer.compress_description(skill, use_oracle_validation=True)
```

### optimizer.py - 阶段二主体优化

主体重构实现：

```python
from src.optimizer import Stage2Optimizer, CompressionMetrics

optimizer = Stage2Optimizer(llm_client=client)

# 分类技能主体
blocks = optimizer.classify_skill_body(skill)

# 各类型压缩处理
core_blocks = optimizer.compress_core_rules(core_blocks)
example_blocks, removed = optimizer.dedup_examples(example_blocks)
template_blocks, removed = optimizer.dedup_templates(template_blocks)
background_blocks, merged = optimizer.summarize_background(background_blocks)
refs, deduped, discarded = optimizer.dedup_references(body_blocks, references)

# 完整优化流水线
blocks, refs, metrics = optimizer.optimize_skill(skill)
```

### quality_gates.py - 质量门控

质量保证机制：

```python
from src.quality_gates import QualityGates, CompressionPipeline, run_pipeline

gates = QualityGates(llm_client=client)

# Gate 1: 忠实性验证
result = gates.verify_faithfulness(
    original_body=skill.body.original,
    compressed_core_rules="压缩后的核心规则"
)
print(f"通过: {result.passed}")
print(f"缺失概念: {result.missing_concepts}")

# Gate 2: 反馈循环
feedback = gates.feedback_loop(
    blocks=compressed_blocks,
    failed_criteria=["任务失败的原因"]
)
print(f"提升块数: {feedback.promotion_count}")

# 使用完整流水线
pipeline = CompressionPipeline(
    llm_client=client,
    enable_gate1=True,
    enable_gate2=True
)
result = pipeline.run_pipeline(skill)
```

### parser.py - 技能解析

解析 SKILL.md 文件：

```python
from src.parser import parse_skill_file, parse_skill_directory

# 解析单个文件
skill = parse_skill_file("path/to/SKILL.md")

# 解析目录
skill = parse_skill_directory("path/to/skill-dir/")
```

### chunker.py - 语义分块

将 Markdown 主体拆分为语义块：

```python
from src.chunker import chunk_markdown_body, count_tokens

# 分块
blocks = chunk_markdown_body(skill.body.original)

# 统计 Token
token_count = count_tokens(text)
```

### llm_client.py - LLM 客户端

支持 OpenAI 兼容 API：

```python
from src.llm_client import SkillLLMClient

# 初始化
client = SkillLLMClient(
    api_key="your-key",
    base_url="https://api.deepseek.com/v1",  # 可选
    model="deepseek-chat"
)

# 测试连接
if client.test_connection():
    print("连接成功")

# 分类内容块
result = client.classify_content_blocks(
    content_chunks=[
        {"chunk_id": "1", "content": "这是核心规则内容..."},
        {"chunk_id": "2", "content": "这是背景说明..."}
    ],
    skill_context="技能名称: 数据分析\n描述: 帮助用户进行数据分析"
)
```

## 技能包结构

每个技能遵循以下结构：

```
skill-name/
├── SKILL.md              # 主文档 (YAML frontmatter + Markdown 主体)
├── scripts/              # Python CLI 工具 (仅使用标准库)
├── references/           # 专业知识库
└── assets/               # 用户模板
```

SKILL.md 文件示例：

```markdown
---
name: marketing-strategy-pmm
description: |
  Product Marketing Manager skill for developing comprehensive go-to-market
  strategies, product positioning, and competitive analysis.
metadata:
  version: 1.0.0
  category: marketing
  tags:
    - product-marketing
    - go-to-market
    - positioning
---

# Marketing Strategy Skill

## Core Principles

1. Always start with customer research
2. Define clear target segments
...

## Background

Product marketing is the process of bringing a product to market...
```

## 压缩效果示例

以 `marketing-strategy-pmm` 技能为例：

| 指标 | 压缩前 | 压缩后 | 变化 |
|------|--------|--------|------|
| 描述 Token | 45 | 23 | 48.9% |
| 主体 Token | 1,245 | 901 | 27.8% |
| 核心规则 Token | - | 312 | 始终加载 |
| 按需加载 Token | - | 589 | 可延迟 |
| 丢弃 Token | - | 344 | 已移除 |
| **总压缩率** | - | - | **35.5%** |

## API 兼容性

支持所有 OpenAI 兼容的 API：

| 提供商 | base_url | model |
|--------|----------|-------|
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini, gpt-4o |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
| Qwen | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-turbo, qwen-plus |
| 本地模型 | http://localhost:8000/v1 | 根据部署配置 |

## 依赖说明

```
pydantic>=2.0        # 数据验证和模型
pyyaml>=6.0          # YAML 解析
openai>=1.0          # OpenAI API 客户端
tiktoken>=0.5        # Token 计数
markdown-it-py>=3.0  # Markdown 解析
tenacity>=8.0        # 重试机制
python-dotenv>=1.0   # 环境变量管理
pytest>=7.0          # 测试框架
pytest-cov>=4.0      # 测试覆盖率
```

## 开发指南

### 添加新的内容类型

1. 在 `models.py` 中扩展 `ContentType` 枚举
2. 更新 `llm_client.py` 中的分类提示词
3. 更新 `optimizer.py` 中的压缩处理逻辑
4. 更新 `quality_gates.py` 中的反馈循环逻辑

### 扩展 LLM 提供商

客户端已支持所有 OpenAI 兼容 API，只需配置正确的 `base_url` 和 `model` 参数。

### 自定义分块策略

修改 `chunker.py` 中的 `_group_tokens_into_chunks` 函数，调整分块粒度。

### 自定义质量门控阈值

```python
# 修改忠实性阈值 (默认 1.0 要求所有概念保留)
gates = QualityGates(client, faithfulness_threshold=0.8)
```

## 测试覆盖

当前测试覆盖率：

| 模块 | 测试数 | 覆盖内容 |
|------|--------|----------|
| test_parser.py | 7 | 文件解析、元数据、引用文件 |
| test_chunker.py | 10 | 分块、Token 计数、边界情况 |
| test_optimizer.py | 19 | 分类、压缩方法、完整流水线 |
| test_stage1.py | 11 | 分割、Oracle、DDMIN、重写 |
| test_quality_gates.py | 13 | 忠实性、反馈循环、完整流水线 |
| **总计** | **61** | 全模块覆盖 |

## 许可证

MIT License