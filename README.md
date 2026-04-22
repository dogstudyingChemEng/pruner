# SkillReducer

基于论文 "SkillReducer: Optimizing LLM Agent Skills for Token Efficiency" (Gao et al., 2026) 实现的 LLM Agent 技能压缩框架，旨在减少 Token 成本的同时保持功能质量。

## 项目简介

SkillReducer 通过两阶段流水线对 LLM Agent 技能进行压缩优化：

**阶段一：路由层优化 (Routing Layer Optimization)**
- 使用 Delta Debugging 算法压缩技能描述
- DDMIN 算法对语义子句进行快速压缩
- 模拟 Oracle 验证路由等价性

**阶段二：主体重构 (Body Restructuring via Progressive Disclosure)**
- 分类驱动的五类内容分类：
  - `core_rule`：核心规则（可操作指令）— 始终加载
  - `background`：背景说明（解释性内容）— 按需加载
  - `example`：示例（代码片段）— 按需加载
  - `template`：模板（样板文本）— 按需加载
  - `redundant`：冗余内容 — 丢弃
- 跨文件去重（主体与引用文件之间）
- 质量门控：忠实性检查 + 任务评估

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
│   └── optimizer.py           # 阶段二优化器实现
│
├── tests/                      # 单元测试
│   ├── test_parser.py         # 解析器测试
│   ├── test_chunker.py        # 分块器测试
│   └── test_optimizer.py      # 优化器测试
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
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_parser.py -v
pytest tests/test_chunker.py -v
pytest tests/test_optimizer.py -v

# 带覆盖率报告
pytest tests/ --cov=src
```

### 压缩单个技能

```python
from src.parser import parse_skill_file
from src.llm_client import SkillLLMClient
from src.optimizer import Stage2Optimizer
from src.chunker import count_tokens

# 1. 解析技能文件
skill = parse_skill_file("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")

# 2. 初始化 LLM 客户端
client = SkillLLMClient(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)

# 3. 初始化优化器
optimizer = Stage2Optimizer(llm_client=client)

# 4. 分类技能主体内容
blocks = optimizer.classify_skill_body(skill)

# 5. 计算压缩潜力
original_tokens = count_tokens(skill.body.original)
metrics = optimizer.calculate_compression_potential(
    blocks,
    original_body_tokens=original_tokens
)

print(f"原始 Token 数: {metrics['original_tokens']}")
print(f"核心规则 Token 数: {metrics['always_loaded_tokens']}")
print(f"压缩率: {metrics['compression_ratio']:.2%}")
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
from src.models import Skill, ContentBlock, ContentType

# 内容类型枚举
class ContentType(str, Enum):
    CORE_RULE = "core_rule"      # 核心规则
    BACKGROUND = "background"    # 背景说明
    EXAMPLE = "example"          # 示例
    TEMPLATE = "template"        # 模板
    REDUNDANT = "redundant"      # 冗余内容

# 内容块
class ContentBlock(BaseModel):
    chunk_id: str           # 块 ID
    content: str            # 内容文本
    content_type: ContentType  # 分类类型
    token_count: int        # Token 数量

# 完整技能
class Skill(BaseModel):
    name: str               # 技能名称
    description: Description  # 技能描述
    body: Body              # 技能主体
    references: References  # 引用文件
    metadata: SkillMetadata # 元数据
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

### optimizer.py - 优化器

阶段二优化实现：

```python
from src.optimizer import Stage2Optimizer

optimizer = Stage2Optimizer(llm_client=client)

# 分类技能主体
blocks = optimizer.classify_skill_body(skill)

# 获取分类统计
summary = optimizer.get_classification_summary(blocks)
# {ContentType.CORE_RULE: 5, ContentType.BACKGROUND: 3, ...}

# 筛选特定类型
core_blocks = optimizer.get_blocks_by_type(blocks, ContentType.CORE_RULE)

# 计算压缩潜力
metrics = optimizer.calculate_compression_potential(
    blocks,
    original_body_tokens=1000
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
| 原始 Token | 1,245 | - | - |
| 核心规则 Token | - | 312 | 始终加载 |
| 按需加载 Token | - | 589 | 可延迟 |
| 丢弃 Token | - | 344 | 已移除 |
| **压缩率** | - | - | **74.9%** |

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
```

## 开发指南

### 添加新的内容类型

1. 在 `models.py` 中扩展 `ContentType` 枚举
2. 更新 `llm_client.py` 中的分类提示词
3. 更新 `optimizer.py` 中的压缩计算逻辑

### 扩展 LLM 提供商

客户端已支持所有 OpenAI 兼容 API，只需配置正确的 `base_url` 和 `model` 参数。

### 自定义分块策略

修改 `chunker.py` 中的 `_group_tokens_into_chunks` 函数，调整分块粒度。

## 参考文献

Gao, Y., et al. (2026). "SkillReducer: Optimizing LLM Agent Skills for Token Efficiency." *Proceedings of the ACM Web Conference 2026*.

## 许可证

MIT License
