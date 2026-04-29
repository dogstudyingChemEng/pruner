# SkillReducer

复现论文 [*SkillReducer: Optimizing LLM Agent Skills for Token Efficiency*](https://arxiv.org/abs/2603.29919) (Gao et al., 2026) 的两阶段 Skill Debloating 框架，对 LLM Agent 技能进行压缩以降低 Token 成本。

## 核心思路

技能由描述（路由层）和正文（指令层）组成。论文通过实证研究发现 55,315 个公开技能存在系统性冗余：26.4% 缺乏路由描述，60%+ 的正文内容非可操作。SkillReducer 通过两个阶段解决：

| 阶段 | 目标 | 方法 | 论文效果 |
|------|------|------|----------|
| Stage 1 | 路由描述压缩 | DDMIN delta debugging + 对抗 oracle + 真实 CLI 验证 | 48% 压缩 |
| Stage 2 | 正文重构 | 五类分类 + 渐进式披露 (progressive disclosure) | 39% 压缩 |
| Quality Gates | 质量保证 | 忠实性验证 + 任务评估反馈循环 | 86% 通过率 |

## 快速开始

```bash
# 安装依赖
pip install -r requirement.txt

# 配置 API (支持 OpenAI / DeepSeek / Qwen)
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY 和 OPENAI_BASE_URL

# 压缩全部 245 个技能
python batch_compress.py --skip-errors

# 先测试 3 个
python batch_compress.py --limit 3

# 生成每个技能的前后对比
python batch_compress.py --limit 5 --compare

# 运行全部 121 个测试
pytest tests/ -v
```

## 项目结构

```
skillpruner/
├── src/
│   ├── models.py              # Pydantic 数据模型
│   ├── parser.py              # SKILL.md 解析 (YAML frontmatter)
│   ├── chunker.py             # Markdown 语义分块 + tiktoken
│   ├── llm_client.py          # OpenAI 兼容客户端 (tenacity 重试)
│   ├── config.py              # SkillReducerConfig (feature toggles)
│   ├── stage1_router.py       # Stage 1: DDMIN 路由层压缩
│   ├── optimizer.py           # Stage 2: 分类 + 渐进式披露
│   ├── quality_gates.py       # Gate 1 (忠实性) + Gate 2 (反馈循环)
│   ├── hybrid_evaluator.py    # pytest + LLM judge 混合评分
│   └── cli_validator.py       # Claude Code CLI 真实触发验证
│
├── batch_compress.py           # 批量压缩脚本 (支持 --compare)
├── tests/                      # 121 个单元测试
├── Claude-Skills/              # 245 个技能包 (测试数据)
├── requirement.txt
└── .env.example
```

## 压缩流水线详解

### Algorithm 1: Stage 1 — 路由层优化

```
SEGMENT(s.d) → DDMIN(U, O_sim, Q) → 逐句 Paraphrase → POLISH → d_fast
                                                                    ↓
                                          original ← NO ← REALTRIGGER(d_fast, Qval)?
                                                                    ↓ YES
                                                               return d_fast
```

1. **语义分割** — LLM 将描述拆分为独立语义子句
2. **DDMIN** — 完整 Zeller & Hildebrandt (2002) 算法：complement testing + subset testing，找到 1-minimal 子集
3. **Oracle O_sim** — 5 个固定 query，4 个 TF-IDF 真实 distractor + 1 个 LLM 生成的对抗 skill，候选顺序随机化
4. **逐句 Paraphrase** — 尝试缩短每个保留子句，仅当 O_sim 仍通过时保留
5. **POLISH** — LLM 将最小子句集合并为流畅描述
6. **Phase 2 真实验证** — 通过 Claude Code CLI（或模拟 oracle）验证，最多 3 步贪心选择性恢复
7. **Fallback** — 恢复失败则退回原始描述

**描述生成** (缺失或 ≤40 token 的描述)：从正文提取 3 个路由信号（primary_capability, trigger_condition, unique_identifiers），各 20-40 token，经 O_sim 验证后接受。

### Algorithm 2: Stage 2 — 正文重构

```
CLASSIFY(s.b) → 五类分拣 → 类型特定压缩 → 交叉文件去重 → 注释路由元数据
                    ↓
    core_rule         background/example/template         redundant
   (始终加载)           (按需加载, read_file)              (丢弃)
```

1. **分类** — LLM (温度 0) 将正文分为五类，失败 3 次后退回 `core_rule`
2. **交叉验证** — 最多 2 轮，结合邻居标签重新评估每个 block
3. **COMPRESSCORE** — 核心规则合并为简洁要点（压缩后必须更短）
4. **DEDUP** — 示例/模板按概念分组，每组保留最优 1 个，去除注释和样板
5. **SUMMARIZE** — 背景合并为单段，保留所有数字、阈值、API 端点
6. **REMOVEOVERLAP + COMPRESS** — 引用文件与正文去重后，压缩剩余内容，<30 token 的丢弃
7. **GENWHEN / GENTOPICS** — 为每个引用生成 when 触发条件 + 3-5 个主题关键词

### Quality Gates

- **Gate 1: 忠实性验证** — Eq. 3: `C_τ(s.b) ⊆ C_τ(b*) ∪ C_τ(R*)`，按内容类型细粒度回滚
- **Gate 2: 任务评估反馈循环** — 生成 5 个评估任务，失败则提升相关 block 至 core_rule（原始形式，不压缩），最多 2 轮
- **三条件评估 (D/A/C)** — 混合评分：pytest 52.3% + LLM judge 47.7%，Cohen's κ 验证

## 编程使用

```python
from src.parser import parse_skill_file
from src.llm_client import SkillLLMClient
from src.quality_gates import run_pipeline

skill = parse_skill_file("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
client = SkillLLMClient(api_key="sk-xxx", base_url="https://api.deepseek.com/v1", model="deepseek-chat")

result = run_pipeline(skill, client, enable_gate1=True, enable_gate2=False)

print(f"压缩率: {result.overall_compression_ratio:.1%}")
print(f"Gate 1: {'PASS' if result.faithfulness_passed else 'FAIL'}")
print(f"原始: {result.original_tokens:,} → 最终: {result.final_tokens:,} tokens")
```

### 配置预设

```python
from src.config import get_paper_config, get_fast_config

# 论文完整配置 (真实 CLI + 三条件评估 + 混合评分)
config = get_paper_config()

# 快速批量处理 (无 CLI, 无三条件, 无混合评分)
config = get_fast_config()
```

## 批量压缩

```bash
python batch_compress.py [--limit N] [--start-from N] [--skip-errors] [--no-gate1] [--compare]
```

| 参数 | 说明 |
|------|------|
| `--limit N` | 仅处理前 N 个 skill |
| `--start-from N` | 从第 N 个开始 (断点续跑) |
| `--skip-errors` | 遇到错误继续 |
| `--no-gate1` | 跳过忠实性检查 (加速) |
| `--compare` | 生成 `compressed_skills/<name>/` 对比目录 |

### 对比输出 (`--compare`)

```
compressed_skills/
└── marketing-strategy-pmm/
    ├── original.md           # 原始 SKILL.md
    ├── compressed.md         # 压缩后 SKILL.md (仅 core rules)
    ├── report.md             # 详细压缩报告
    └── references/           # 按需加载模块
        ├── on-demand-examples.md
        ├── on-demand-templates.md
        └── on-demand-background.md
```

## API 提供商

支持所有 OpenAI 兼容 API，`batch_compress.py` 自动检测模型：

| 提供商 | Base URL | 自动模型 |
|--------|----------|----------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-v4-flash` |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 其他兼容 | 自定义 | 设置 `LLM_MODEL` 环境变量 |

## 测试

```bash
pytest tests/ -v                    # 全部 121 个测试
pytest tests/test_stage1.py -v      # Stage 1 (DDMIN, oracle, 分割)
pytest tests/test_optimizer.py -v   # Stage 2 (分类, 压缩, 去重)
pytest tests/test_quality_gates.py -v  # Gates (忠实性, 反馈循环, 流水线)
pytest tests/ --cov=src             # 带覆盖率
```

## 依赖

```
pydantic>=2.0        openai>=1.0          tiktoken>=0.5
pyyaml>=6.0          markdown-it-py>=3.0   tenacity>=8.0
python-dotenv>=1.0   pytest>=7.0
```

## 论文引用

```bibtex
@article{gao2026skillreducer,
  title={SkillReducer: Optimizing LLM Agent Skills for Token Efficiency},
  author={Gao, Yudong and Li, Zongjie and Yuan, Yuanyuan and Ji, Zimo and Ma, Pingchuan and Wang, Shuai},
  journal={arXiv preprint arXiv:2603.29919},
  year={2026}
}
```
