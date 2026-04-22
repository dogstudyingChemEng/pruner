# SkillReducer 压缩报告

**Skill:** marketing-strategy-pmm
**生成时间:** 2026-04-22

---

## 第一部分：原始 Skill

### 元数据

| 字段 | 值 |
|------|------|
| 名称 | marketing-strategy-pmm |
| 版本 | 1.0.0 |
| 作者 | borghei |
| 类别 | marketing |
| 标签 | product-marketing, positioning, gtm, competitive-intel, launches |

### 描述（Description）

描述是 Skill 的路由层文本，用于匹配用户查询到合适的 Skill。

```
Product marketing skill for positioning, GTM strategy, competitive intelligence, and product launches. Covers April Dunford positioning, ICP definition, competitive battlecards, launch playbooks, and international market entry.

```

**描述 Token 数：** 40

### 原始 Body

Body 是 Skill 的主体内容，包含主要指令文档，注入到 LLM 上下文中。

**Body Token 数：** 3652

#### Body 内容预览（前 1000 字符）

```markdown
# Marketing Strategy & PMM

Product marketing patterns for positioning, GTM strategy, and competitive intelligence.

---

## Table of Contents

- [ICP Definition Workflow](#icp-definition-workflow)
- [Positioning Development](#positioning-development)
- [Competitive Intelligence](#competitive-intelligence)
- [Product Launch Planning](#product-launch-planning)
- [Sales Enablement](#sales-enablement)
- [International Expansion](#international-expansion)
- [Reference Documentation](#reference-documentation)

---

## ICP Definition Workflow

Define ideal customer profile for targeting:

1. Analyze existing customers (top 20% by LTV)
2. Identify common firmographics (size, industry, revenue)
3. Map technographics (tools, maturity, integrations)
4. Document psychographics (pain level, motivation, risk tolerance)
5. Define 3-5 buyer personas (economic, technical, user)
6. Validate against sales cycle and churn data
7. Score prospects A/B/C/D based on ICP fit
8. **Validation:** A-fit customers
...
```

---

## 第二部分：切分过程

### 切分策略

Markdown Body 按语义单元切分，策略如下：
- 每个标题（h1-h6）开始一个新的 chunk
- 标题后的内容（段落、列表、代码块、表格）与该标题组合在一起
- 顶层内容（在任何标题之前）形成独立的 chunk

### 切分结果

| 指标 | 值 |
|------|------|
| 总 Chunk 数 | 41 |
| 切分后总 Token 数 | 2193 |
| 原始 Token 数 | 3652 |
| 内容保留率 | 60.0% |

### 所有 Chunk 详情

| 序号 | Chunk ID | Token 数 | 内容预览 |
|------|----------|----------|----------|
| 1 | chunk_dcc000e9 | 22 | # Marketing Strategy & PMM\nProduct marketing patterns for positioning, GTM strat... |
| 2 | chunk_ce030789 | 6 | ## Table of Contents\n---... |
| 3 | chunk_b49826bf | 13 | ## ICP Definition Workflow\nDefine ideal customer profile for targeting:... |
| 4 | chunk_4e5cdc6d | 86 | ### Firmographics Template\n| Dimension | Target Range | Rationale |\n|-----------... |
| 5 | chunk_9ddde186 | 36 | ### Buyer Personas\n**Economic Buyer** (signs contract):\n**Technical Buyer** (eva... |
| 6 | chunk_feb86337 | 7 | ### ICP Validation Checklist\n---... |
| 7 | chunk_d020bcd2 | 13 | ## Positioning Development\nDevelop positioning using April Dunford methodology:... |
| 8 | chunk_b16af797 | 47 | ### Positioning Statement Template\n```\nFOR [target customer]\nWHO [statement of n... |
| 9 | chunk_110ac9c3 | 50 | ### Value Proposition Formula\nTemplate: `[Product] helps [Target Customer] [Achi... |
| 10 | chunk_9e8ed1b0 | 105 | ### Messaging Hierarchy\n| Level | Content | Example |\n|-------|---------|-------... |
| 11 | chunk_16639620 | 9 | ## Competitive Intelligence\nBuild competitive knowledge base:... |
| 12 | chunk_ded62cfa | 79 | ### Competitive Tier Structure\n| Tier | Definition | Examples |\n|------|--------... |
| 13 | chunk_a91153df | 154 | ### Battlecard Template\n```\nCOMPETITOR: [Name]\nOVERVIEW: Founded [year], Funding... |
| 14 | chunk_8b748ac7 | 10 | ### Win/Loss Analysis\nTrack monthly:\n---... |
| 15 | chunk_14c40abc | 87 | ## Product Launch Planning\nPlan launches by tier:\n| Tier | Scope | Prep Time | B... |
| 16 | chunk_0e8a27df | 12 | ### Tier 1 Launch Workflow\nExecute major product launch:... |
| 17 | chunk_854b92c3 | 4 | ### Launch Day Checklist... |
| 18 | chunk_f30ba653 | 74 | ### Launch Metrics\n| Metric | Leading (Daily) | Lagging (Weekly) |\n|--------|---... |
| 19 | chunk_2600fd8e | 13 | ## Sales Enablement\nEquip sales team with PMM assets:... |
| 20 | chunk_d2381719 | 77 | ### Sales Deck Structure\n| Slide | Content |\n|-------|---------|\n| 1-2 | Title, ... |
| 21 | chunk_1d32c90e | 74 | ### Demo Flow\n```\n1. Intro (2 min): Who we are, agenda\n2. Discovery (5 min): The... |
| 22 | chunk_c0f088df | 68 | ### Sales-Marketing Handoff\n| Handoff | Frequency | Content |\n|---------|-------... |
| 23 | chunk_0d8922f7 | 9 | ## International Expansion\nEnter new markets systematically:... |
| 24 | chunk_2143cf49 | 113 | ### Market Priority (Series A)\n| Market | Timeline | Budget % | Target ARR |\n|--... |
| 25 | chunk_c1ac19af | 5 | ### Localization Checklist\n---... |
| 26 | chunk_d99610de | 3 | ## Reference Documentation... |
| 27 | chunk_90fe38d3 | 17 | ### Positioning Frameworks\n`references/positioning-frameworks.md` contains:... |
| 28 | chunk_5b5aec55 | 15 | ### Launch Checklists\n`references/launch-checklists.md` contains:... |
| 29 | chunk_2e6c00fb | 15 | ### International GTM\n`references/international-gtm.md` contains:... |
| 30 | chunk_e75d9c2d | 15 | ### Messaging Templates\n`references/messaging-templates.md` contains:\n---... |
| 31 | chunk_322d6ae1 | 101 | ## PMM KPIs\n| Metric | Target | Measurement |\n|--------|--------|-------------|\n... |
| 32 | chunk_724b6b3d | 3 | ## Quick Reference... |
| 33 | chunk_b3483e2d | 59 | ### PMM Monthly Rhythm\n| Week | Focus |\n|------|-------|\n| 1 | Review metrics, u... |
| 34 | chunk_c400d7a8 | 5 | ## Proactive Triggers... |
| 35 | chunk_5955a102 | 95 | ## Output Artifacts\n| When you ask for... | You get... |\n|---------------------|... |
| 36 | chunk_3e83dc31 | 9 | ## Communication\nAll output passes quality verification:... |
| 37 | chunk_e2049321 | 5 | ## Related Skills\n---... |
| 38 | chunk_4aad2c74 | 348 | ## Troubleshooting\n| Symptom | Likely Cause | Resolution |\n|---------|----------... |
| 39 | chunk_c09bff3e | 5 | ## Success Criteria\n---... |
| 40 | chunk_db1c68a3 | 199 | ## Scope & Limitations\n**In Scope:** Product positioning (April Dunford methodol... |
| 41 | chunk_62d2af84 | 126 | ## Scripts\n| Script | Purpose | Usage |\n|--------|---------|-------|\n| `scripts/... |

### Chunk 完整内容

#### Chunk 1：`chunk_dcc000e9`

**Token 数：** 22

**内容：**
```markdown
# Marketing Strategy & PMM
Product marketing patterns for positioning, GTM strategy, and competitive intelligence.
---
```

---

#### Chunk 2：`chunk_ce030789`

**Token 数：** 6

**内容：**
```markdown
## Table of Contents
---
```

---

#### Chunk 3：`chunk_b49826bf`

**Token 数：** 13

**内容：**
```markdown
## ICP Definition Workflow
Define ideal customer profile for targeting:
```

---

#### Chunk 4：`chunk_4e5cdc6d`

**Token 数：** 86

**内容：**
```markdown
### Firmographics Template
| Dimension | Target Range | Rationale |
|-----------|--------------|-----------|
| Employees | 50-5000 | Series A sweet spot |
| Revenue | $5M-$500M | Budget available |
| Industry | SaaS, Tech, Services | Product fit |
| Geography | US, UK, DACH | Market priority |
| Funding | Seed to Growth | Willing to adopt |
```

---

#### Chunk 5：`chunk_9ddde186`

**Token 数：** 36

**内容：**
```markdown
### Buyer Personas
**Economic Buyer** (signs contract):
**Technical Buyer** (evaluates product):
**User/Champion** (advocates internally):
```

---

#### Chunk 6：`chunk_feb86337`

**Token 数：** 7

**内容：**
```markdown
### ICP Validation Checklist
---
```

---

#### Chunk 7：`chunk_d020bcd2`

**Token 数：** 13

**内容：**
```markdown
## Positioning Development
Develop positioning using April Dunford methodology:
```

---

#### Chunk 8：`chunk_b16af797`

**Token 数：** 47

**内容：**
```markdown
### Positioning Statement Template
```
FOR [target customer]
WHO [statement of need]
THE [product] IS A [category]
THAT [key benefit]
UNLIKE [competitive alternative]
OUR PRODUCT [primary differentiation]

```
```

---

#### Chunk 9：`chunk_110ac9c3`

**Token 数：** 50

**内容：**
```markdown
### Value Proposition Formula
Template: `[Product] helps [Target Customer] [Achieve Goal] by [Unique Approach]`
Example: "Acme helps mid-market SaaS teams ship 2x faster by automating project workflows with AI"
```

---

#### Chunk 10：`chunk_9e8ed1b0`

**Token 数：** 105

**内容：**
```markdown
### Messaging Hierarchy
| Level | Content | Example |
|-------|---------|---------|
| Headline | 5-7 words | "Ship faster with AI automation" |
| Subhead | 1 sentence | "Automate workflows so teams focus on what matters" |
| Benefits | 3-4 bullets | Speed, quality, collaboration, cost |
| Features | Supporting evidence | AI automation → 10 hrs/week saved |
| Proof | Social proof | Customer logos, stats, case studies |
---
```

---

#### Chunk 11：`chunk_16639620`

**Token 数：** 9

**内容：**
```markdown
## Competitive Intelligence
Build competitive knowledge base:
```

---

#### Chunk 12：`chunk_ded62cfa`

**Token 数：** 79

**内容：**
```markdown
### Competitive Tier Structure
| Tier | Definition | Examples |
|------|------------|----------|
| 1 | Direct competitor, same category | [Competitor A, B] |
| 2 | Adjacent solution, overlapping use case | [Alt Solution C, D] |
| 3 | Status quo (what they do today) | Spreadsheets, manual, in-house |
```

---

#### Chunk 13：`chunk_a91153df`

**Token 数：** 154

**内容：**
```markdown
### Battlecard Template
```
COMPETITOR: [Name]
OVERVIEW: Founded [year], Funding [stage], Size [employees]

POSITIONING:
- They say: "[Their claim]"
- Reality: [Your assessment]

STRENGTHS:
1. [What they do well]
2. [What they do well]

WEAKNESSES:
1. [Where they fall short]
2. [Where they fall short]

OUR ADVANTAGES:
1. [Your advantage + evidence]
2. [Your advantage + evidence]

WHEN WE WIN:
- [Scenario where you win]

WHEN WE LOSE:
- [Scenario where they win]

TALK TRACK:
Objection: "[Common objection]"
Response: "[Your response]"

```
```

---

#### Chunk 14：`chunk_8b748ac7`

**Token 数：** 10

**内容：**
```markdown
### Win/Loss Analysis
Track monthly:
---
```

---

#### Chunk 15：`chunk_14c40abc`

**Token 数：** 87

**内容：**
```markdown
## Product Launch Planning
Plan launches by tier:
| Tier | Scope | Prep Time | Budget |
|------|-------|-----------|--------|
| 1 | New product, major feature | 6-8 weeks | $50-100k |
| 2 | Significant feature, integration | 3-4 weeks | $10-25k |
| 3 | Small improvement | 1 week | <$5k |
```

---

#### Chunk 16：`chunk_0e8a27df`

**Token 数：** 12

**内容：**
```markdown
### Tier 1 Launch Workflow
Execute major product launch:
```

---

#### Chunk 17：`chunk_854b92c3`

**Token 数：** 4

**内容：**
```markdown
### Launch Day Checklist
```

---

#### Chunk 18：`chunk_f30ba653`

**Token 数：** 74

**内容：**
```markdown
### Launch Metrics
| Metric | Leading (Daily) | Lagging (Weekly) |
|--------|-----------------|------------------|
| Traffic | Landing page visitors | - |
| Engagement | Demo requests, signups | Feature adoption % |
| Pipeline | MQLs generated | SQLs, pipeline $ |
| Revenue | - | Deals closed, revenue |
---
```

---

#### Chunk 19：`chunk_2600fd8e`

**Token 数：** 13

**内容：**
```markdown
## Sales Enablement
Equip sales team with PMM assets:
```

---

#### Chunk 20：`chunk_d2381719`

**Token 数：** 77

**内容：**
```markdown
### Sales Deck Structure
| Slide | Content |
|-------|---------|
| 1-2 | Title, agenda |
| 3-4 | Company intro, problem statement |
| 5-7 | Solution, key benefits, demo |
| 8-10 | Differentiation, case study, pricing |
| 11-12 | Implementation, support, next steps |
```

---

#### Chunk 21：`chunk_1d32c90e`

**Token 数：** 74

**内容：**
```markdown
### Demo Flow
```
1. Intro (2 min): Who we are, agenda
2. Discovery (5 min): Their needs, pain points
3. Demo (20 min): Product focused on their use case
4. Q&A (10 min): Objection handling
5. Next steps (3 min): Trial, POC, proposal

```
```

---

#### Chunk 22：`chunk_c0f088df`

**Token 数：** 68

**内容：**
```markdown
### Sales-Marketing Handoff
| Handoff | Frequency | Content |
|---------|-----------|---------|
| Weekly sync | 30 min | Win/loss, competitive, new assets |
| Monthly enablement | 60 min | Product updates, training |
| Quarterly review | Half-day | Results, strategy, planning |
---
```

---

#### Chunk 23：`chunk_0d8922f7`

**Token 数：** 9

**内容：**
```markdown
## International Expansion
Enter new markets systematically:
```

---

#### Chunk 24：`chunk_2143cf49`

**Token 数：** 113

**内容：**
```markdown
### Market Priority (Series A)
| Market | Timeline | Budget % | Target ARR |
|--------|----------|----------|------------|
| US | Months 1-6 | 50% | $1M |
| UK | Months 4-9 | 20% | $500k |
| DACH | Months 7-12 | 15% | $300k |
| France | Months 10-15 | 10% | $200k |
| Canada | Months 7-12 | 5% | $100k |
```

---

#### Chunk 25：`chunk_c1ac19af`

**Token 数：** 5

**内容：**
```markdown
### Localization Checklist
---
```

---

#### Chunk 26：`chunk_d99610de`

**Token 数：** 3

**内容：**
```markdown
## Reference Documentation
```

---

#### Chunk 27：`chunk_90fe38d3`

**Token 数：** 17

**内容：**
```markdown
### Positioning Frameworks
`references/positioning-frameworks.md` contains:
```

---

#### Chunk 28：`chunk_5b5aec55`

**Token 数：** 15

**内容：**
```markdown
### Launch Checklists
`references/launch-checklists.md` contains:
```

---

#### Chunk 29：`chunk_2e6c00fb`

**Token 数：** 15

**内容：**
```markdown
### International GTM
`references/international-gtm.md` contains:
```

---

#### Chunk 30：`chunk_e75d9c2d`

**Token 数：** 15

**内容：**
```markdown
### Messaging Templates
`references/messaging-templates.md` contains:
---
```

---

#### Chunk 31：`chunk_322d6ae1`

**Token 数：** 101

**内容：**
```markdown
## PMM KPIs
| Metric | Target | Measurement |
|--------|--------|-------------|
| Product adoption | >40% in 90 days | Feature usage after launch |
| Win rate | >30% competitive | Deals won vs. competitors |
| Sales velocity | -20% YoY | Days from SQL to close |
| Deal size | +25% YoY | Average contract value |
| Launch pipeline | 3:1 ROMI | Pipeline $ : marketing spend |
---
```

---

#### Chunk 32：`chunk_724b6b3d`

**Token 数：** 3

**内容：**
```markdown
## Quick Reference
```

---

#### Chunk 33：`chunk_b3483e2d`

**Token 数：** 59

**内容：**
```markdown
### PMM Monthly Rhythm
| Week | Focus |
|------|-------|
| 1 | Review metrics, update battlecards |
| 2 | Create assets, publish content |
| 3 | Support launches, optimize campaigns |
| 4 | Monthly report, plan next month |
```

---

#### Chunk 34：`chunk_c400d7a8`

**Token 数：** 5

**内容：**
```markdown
## Proactive Triggers
```

---

#### Chunk 35：`chunk_5955a102`

**Token 数：** 95

**内容：**
```markdown
## Output Artifacts
| When you ask for... | You get... |
|---------------------|------------|
| "Position my product" | Positioning framework (April Dunford method) with completed output |
| "GTM strategy" | Go-to-market plan with channels, messaging, and timeline |
| "Competitive positioning" | Positioning map with competitive gaps and opportunities |
| "Sales enablement" | Sales deck structure, battlecards, and demo flow |
```

---

#### Chunk 36：`chunk_3e83dc31`

**Token 数：** 9

**内容：**
```markdown
## Communication
All output passes quality verification:
```

---

#### Chunk 37：`chunk_e2049321`

**Token 数：** 5

**内容：**
```markdown
## Related Skills
---
```

---

#### Chunk 38：`chunk_4aad2c74`

**Token 数：** 348

**内容：**
```markdown
## Troubleshooting
| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| Positioning resonates internally but customers do not repeat it | Positioning built from company perspective, not customer language | Rerun April Dunford methodology starting from competitive alternatives, not from product features |
| Win rate against specific competitor below 25% | Battlecard outdated or sales team not using it | Run win_loss_analyzer.py to identify loss patterns; update battlecard monthly; validate 80%+ sales usage |
| GTM motion producing high MQLs but low pipeline conversion | Wrong GTM motion for ACV and buyer type; marketing-led when should be sales-led | Reassess motion using gtm_planner.py; for ACV >$25K, shift to sales-led or hybrid PLG+sales |
| Sales enablement assets gathering dust | Assets created without sales input; format does not match how sales actually works | Co-create assets with sales; survey sales on what they need; track asset usage in deal cycles |
| International expansion burning cash with zero pipeline | Market entered without validating demand (inbound signal, TAM) | Validate 3+ paying customers from market in first 90 days; if not, pause and reassess market priority |
| Competitive intelligence always reactive to lost deals | No proactive monitoring system; battlecards only updated post-loss | Set up monthly competitor monitoring (website, pricing, job postings, G2 reviews); update battlecards proactively |
| Messaging differs across website, sales deck, and ads | No messaging hierarchy documented; each team creates independently | Build messaging hierarchy (headline > subhead > benefits > features > proof) and enforce across all touchpoints |
---
```

---

#### Chunk 39：`chunk_c09bff3e`

**Token 数：** 5

**内容：**
```markdown
## Success Criteria
---
```

---

#### Chunk 40：`chunk_db1c68a3`

**Token 数：** 199

**内容：**
```markdown
## Scope & Limitations
**In Scope:** Product positioning (April Dunford methodology), ICP definition and validation, competitive intelligence and battlecards, GTM strategy and motion selection (PLG, sales-led, marketing-led, community-led), product launch planning, sales enablement, win/loss analysis, international expansion planning, messaging hierarchy, PMM KPIs.
**Out of Scope:** Brand identity and visual design (see brand-guidelines skill), demand generation execution (see marketing-demand-acquisition skill), content creation (see content-creator skill), pricing strategy optimization, sales process design.
**Limitations:** Positioning frameworks require real customer input to be effective — internally generated positioning is unreliable. Win/loss analysis requires honest deal outcome data from sales; incomplete data produces misleading patterns. GTM motion recommendations are based on ACV and buyer type heuristics; edge cases may require hybrid approaches. International expansion timelines assume US-first model and may not apply to non-US companies.
---
```

---

#### Chunk 41：`chunk_62d2af84`

**Token 数：** 126

**内容：**
```markdown
## Scripts
| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/gtm_planner.py` | Generate GTM plans with motion selection, channel strategy, and timeline | `python scripts/gtm_planner.py config.json --demo` |
| `scripts/win_loss_analyzer.py` | Analyze deal outcomes by competitor, segment, and reason | `python scripts/win_loss_analyzer.py deals.json --demo` |
| `scripts/battlecard_generator.py` | Generate competitive battlecards with feature comparison and objection handling | `python scripts/battlecard_generator.py competitor.json --demo` |
```

---

## 第三部分：LLM 分类

### 分类类别

| 类别 | 描述 | 加载策略 |
|------|------|----------|
| `core_rule` (核心规则) | 可执行指令、规则、工作流程 | **始终加载** |
| `background` (背景知识) | 解释性内容、定义、上下文 | **按需加载** |
| `example` (示例) | 代码片段、使用示例 | **按需加载** |
| `template` (模板) | 样板文本、现成格式 | **按需加载** |
| `redundant` (冗余) | 重复或无价值内容 | **丢弃** |

### 分类汇总

| 类别 | 块数 | Token 数 | 占比 |
|------|------|----------|------|
| 核心规则 | 15 | 776 | 35.4% |
| 背景知识 | 15 | 621 | 28.3% |
| 示例 | 2 | 239 | 10.9% |
| 模板 | 8 | 551 | 25.1% |
| 冗余 | 1 | 6 | 0.3% |

### 分类结果详情

| 序号 | Chunk ID | 分类结果 | Token 数 | 内容预览 |
|------|----------|----------|----------|----------|
| 1 | chunk_ed02e664 | 背景知识 | 22 | # Marketing Strategy & PMM\nProduct marketing patterns for po... |
| 2 | chunk_f4f98776 | 冗余 | 6 | ## Table of Contents\n---... |
| 3 | chunk_2bf45614 | 核心规则 | 13 | ## ICP Definition Workflow\nDefine ideal customer profile for... |
| 4 | chunk_fdb81b38 | 模板 | 86 | ### Firmographics Template\n| Dimension | Target Range | Rati... |
| 5 | chunk_50c30332 | 模板 | 36 | ### Buyer Personas\n**Economic Buyer** (signs contract):\n**Te... |
| 6 | chunk_f5e7df3f | 核心规则 | 7 | ### ICP Validation Checklist\n---... |
| 7 | chunk_03bf4f4e | 核心规则 | 13 | ## Positioning Development\nDevelop positioning using April D... |
| 8 | chunk_06bc4d15 | 模板 | 47 | ### Positioning Statement Template\n```\nFOR [target customer]... |
| 9 | chunk_edff49eb | 模板 | 50 | ### Value Proposition Formula\nTemplate: `[Product] helps [Ta... |
| 10 | chunk_59263fc5 | 模板 | 105 | ### Messaging Hierarchy\n| Level | Content | Example |\n|-----... |
| 11 | chunk_4791ad4c | 核心规则 | 9 | ## Competitive Intelligence\nBuild competitive knowledge base... |
| 12 | chunk_c7998bb5 | 背景知识 | 79 | ### Competitive Tier Structure\n| Tier | Definition | Example... |
| 13 | chunk_f771e2e8 | 模板 | 154 | ### Battlecard Template\n```\nCOMPETITOR: [Name]\nOVERVIEW: Fou... |
| 14 | chunk_8920d72a | 核心规则 | 10 | ### Win/Loss Analysis\nTrack monthly:\n---... |
| 15 | chunk_290142be | 背景知识 | 87 | ## Product Launch Planning\nPlan launches by tier:\n| Tier | S... |
| 16 | chunk_9fc58abd | 核心规则 | 12 | ### Tier 1 Launch Workflow\nExecute major product launch:... |
| 17 | chunk_920ab21f | 核心规则 | 4 | ### Launch Day Checklist... |
| 18 | chunk_5ceb50bc | 背景知识 | 74 | ### Launch Metrics\n| Metric | Leading (Daily) | Lagging (Wee... |
| 19 | chunk_6c9081a5 | 核心规则 | 13 | ## Sales Enablement\nEquip sales team with PMM assets:... |
| 20 | chunk_cd06d881 | 背景知识 | 77 | ### Sales Deck Structure\n| Slide | Content |\n|-------|------... |
| 21 | chunk_96dac3ec | 核心规则 | 74 | ### Demo Flow\n```\n1. Intro (2 min): Who we are, agenda\n2. Di... |
| 22 | chunk_de58281e | 模板 | 68 | ### Sales-Marketing Handoff\n| Handoff | Frequency | Content ... |
| 23 | chunk_069efdb1 | 核心规则 | 9 | ## International Expansion\nEnter new markets systematically:... |
| 24 | chunk_3e926d56 | 示例 | 113 | ### Market Priority (Series A)\n| Market | Timeline | Budget ... |
| 25 | chunk_dc761114 | 模板 | 5 | ### Localization Checklist\n---... |
| 26 | chunk_e84d14ea | 背景知识 | 3 | ## Reference Documentation... |
| 27 | chunk_32166ed7 | 背景知识 | 17 | ### Positioning Frameworks\n`references/positioning-framework... |
| 28 | chunk_c443c633 | 背景知识 | 15 | ### Launch Checklists\n`references/launch-checklists.md` cont... |
| 29 | chunk_2cf4cd03 | 背景知识 | 15 | ### International GTM\n`references/international-gtm.md` cont... |
| 30 | chunk_75c81bbb | 背景知识 | 15 | ### Messaging Templates\n`references/messaging-templates.md` ... |
| 31 | chunk_fa0074d4 | 核心规则 | 101 | ## PMM KPIs\n| Metric | Target | Measurement |\n|--------|----... |
| 32 | chunk_f72900c9 | 背景知识 | 3 | ## Quick Reference... |
| 33 | chunk_42a37205 | 核心规则 | 59 | ### PMM Monthly Rhythm\n| Week | Focus |\n|------|-------|\n| 1... |
| 34 | chunk_feb0a0f5 | 背景知识 | 5 | ## Proactive Triggers... |
| 35 | chunk_3b405a08 | 核心规则 | 95 | ## Output Artifacts\n| When you ask for... | You get... |\n|--... |
| 36 | chunk_86fc20ed | 核心规则 | 9 | ## Communication\nAll output passes quality verification:... |
| 37 | chunk_1386d53f | 背景知识 | 5 | ## Related Skills\n---... |
| 38 | chunk_b43a0c48 | 核心规则 | 348 | ## Troubleshooting\n| Symptom | Likely Cause | Resolution |\n|... |
| 39 | chunk_a594b3ab | 背景知识 | 5 | ## Success Criteria\n---... |
| 40 | chunk_e5626c64 | 背景知识 | 199 | ## Scope & Limitations\n**In Scope:** Product positioning (Ap... |
| 41 | chunk_2b540ed6 | 示例 | 126 | ## Scripts\n| Script | Purpose | Usage |\n|--------|---------|... |

---

## 第四部分：各类型内容详情

### 核心规则（始终加载）

**说明：** 这部分内容包含可执行的指令，是 Skill 的核心，每次调用时都会注入到上下文中。

**统计：** 15 个块，共 776 tokens

#### 块 1：`chunk_2bf45614`（13 tokens）

```markdown
## ICP Definition Workflow
Define ideal customer profile for targeting:
```

#### 块 2：`chunk_f5e7df3f`（7 tokens）

```markdown
### ICP Validation Checklist
---
```

#### 块 3：`chunk_03bf4f4e`（13 tokens）

```markdown
## Positioning Development
Develop positioning using April Dunford methodology:
```

#### 块 4：`chunk_4791ad4c`（9 tokens）

```markdown
## Competitive Intelligence
Build competitive knowledge base:
```

#### 块 5：`chunk_8920d72a`（10 tokens）

```markdown
### Win/Loss Analysis
Track monthly:
---
```

#### 块 6：`chunk_9fc58abd`（12 tokens）

```markdown
### Tier 1 Launch Workflow
Execute major product launch:
```

#### 块 7：`chunk_920ab21f`（4 tokens）

```markdown
### Launch Day Checklist
```

#### 块 8：`chunk_6c9081a5`（13 tokens）

```markdown
## Sales Enablement
Equip sales team with PMM assets:
```

#### 块 9：`chunk_96dac3ec`（74 tokens）

```markdown
### Demo Flow
```
1. Intro (2 min): Who we are, agenda
2. Discovery (5 min): Their needs, pain points
3. Demo (20 min): Product focused on their use case
4. Q&A (10 min): Objection handling
5. Next steps (3 min): Trial, POC, proposal

```
```

#### 块 10：`chunk_069efdb1`（9 tokens）

```markdown
## International Expansion
Enter new markets systematically:
```

#### 块 11：`chunk_fa0074d4`（101 tokens）

```markdown
## PMM KPIs
| Metric | Target | Measurement |
|--------|--------|-------------|
| Product adoption | >40% in 90 days | Feature usage after launch |
| Win rate | >30% competitive | Deals won vs. competitors |
| Sales velocity | -20% YoY | Days from SQL to close |
| Deal size | +25% YoY | Average contract value |
| Launch pipeline | 3:1 ROMI | Pipeline $ : marketing spend |
---
```

#### 块 12：`chunk_42a37205`（59 tokens）

```markdown
### PMM Monthly Rhythm
| Week | Focus |
|------|-------|
| 1 | Review metrics, update battlecards |
| 2 | Create assets, publish content |
| 3 | Support launches, optimize campaigns |
| 4 | Monthly report, plan next month |
```

#### 块 13：`chunk_3b405a08`（95 tokens）

```markdown
## Output Artifacts
| When you ask for... | You get... |
|---------------------|------------|
| "Position my product" | Positioning framework (April Dunford method) with completed output |
| "GTM strategy" | Go-to-market plan with channels, messaging, and timeline |
| "Competitive positioning" | Positioning map with competitive gaps and opportunities |
| "Sales enablement" | Sales deck structure, battlecards, and demo flow |
```

#### 块 14：`chunk_86fc20ed`（9 tokens）

```markdown
## Communication
All output passes quality verification:
```

#### 块 15：`chunk_b43a0c48`（348 tokens）

```markdown
## Troubleshooting
| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| Positioning resonates internally but customers do not repeat it | Positioning built from company perspective, not customer language | Rerun April Dunford methodology starting from competitive alternatives, not from product features |
| Win rate against specific competitor below 25% | Battlecard outdated or sales team not using it | Run win_loss_analyzer.py to identify loss patterns; update battlecard monthly; validate 80%+ sales usage |
| GTM motion producing high MQLs but low pipeline conversion | Wrong GTM motion for ACV and buyer type; marketing-led when should be sales-led | Reassess motion using gtm_planner.py; for ACV >$25K, shift to sales-led or hybrid PLG+sales |
| Sales enablement assets gathering dust | Assets created without sales input; format does not match how sales actually works | Co-create assets with sales; survey sales on what they need; track asset usage in deal cycles |
| International expansion burning cash with zero pipeline | Market entered without validating demand (inbound signal, TAM) | Validate 3+ paying customers from market in first 90 days; if not, pause and reassess market priority |
| Competitive intelligence always reactive to lost deals | No proactive monitoring system; battlecards only updated post-loss | Set up monthly competitor monitoring (website, pricing, job postings, G2 reviews); update battlecards proactively |
| Messaging differs across website, sales deck, and ads | No messaging hierarchy documented; each team creates independently | Build messaging hierarchy (headline > subhead > benefits > features > proof) and enforce across all touchpoints |
---
```

---

### 背景知识（按需加载）

**说明：** 这部分内容包含解释性知识，在用户需要了解背景或概念时按需加载。

**统计：** 15 个块，共 621 tokens

#### 块 1：`chunk_ed02e664`（22 tokens）

```markdown
# Marketing Strategy & PMM
Product marketing patterns for positioning, GTM strategy, and competitive intelligence.
---
```

#### 块 2：`chunk_c7998bb5`（79 tokens）

```markdown
### Competitive Tier Structure
| Tier | Definition | Examples |
|------|------------|----------|
| 1 | Direct competitor, same category | [Competitor A, B] |
| 2 | Adjacent solution, overlapping use case | [Alt Solution C, D] |
| 3 | Status quo (what they do today) | Spreadsheets, manual, in-house |
```

#### 块 3：`chunk_290142be`（87 tokens）

```markdown
## Product Launch Planning
Plan launches by tier:
| Tier | Scope | Prep Time | Budget |
|------|-------|-----------|--------|
| 1 | New product, major feature | 6-8 weeks | $50-100k |
| 2 | Significant feature, integration | 3-4 weeks | $10-25k |
| 3 | Small improvement | 1 week | <$5k |
```

#### 块 4：`chunk_5ceb50bc`（74 tokens）

```markdown
### Launch Metrics
| Metric | Leading (Daily) | Lagging (Weekly) |
|--------|-----------------|------------------|
| Traffic | Landing page visitors | - |
| Engagement | Demo requests, signups | Feature adoption % |
| Pipeline | MQLs generated | SQLs, pipeline $ |
| Revenue | - | Deals closed, revenue |
---
```

#### 块 5：`chunk_cd06d881`（77 tokens）

```markdown
### Sales Deck Structure
| Slide | Content |
|-------|---------|
| 1-2 | Title, agenda |
| 3-4 | Company intro, problem statement |
| 5-7 | Solution, key benefits, demo |
| 8-10 | Differentiation, case study, pricing |
| 11-12 | Implementation, support, next steps |
```

#### 块 6：`chunk_e84d14ea`（3 tokens）

```markdown
## Reference Documentation
```

#### 块 7：`chunk_32166ed7`（17 tokens）

```markdown
### Positioning Frameworks
`references/positioning-frameworks.md` contains:
```

#### 块 8：`chunk_c443c633`（15 tokens）

```markdown
### Launch Checklists
`references/launch-checklists.md` contains:
```

#### 块 9：`chunk_2cf4cd03`（15 tokens）

```markdown
### International GTM
`references/international-gtm.md` contains:
```

#### 块 10：`chunk_75c81bbb`（15 tokens）

```markdown
### Messaging Templates
`references/messaging-templates.md` contains:
---
```

#### 块 11：`chunk_f72900c9`（3 tokens）

```markdown
## Quick Reference
```

#### 块 12：`chunk_feb0a0f5`（5 tokens）

```markdown
## Proactive Triggers
```

#### 块 13：`chunk_1386d53f`（5 tokens）

```markdown
## Related Skills
---
```

#### 块 14：`chunk_a594b3ab`（5 tokens）

```markdown
## Success Criteria
---
```

#### 块 15：`chunk_e5626c64`（199 tokens）

```markdown
## Scope & Limitations
**In Scope:** Product positioning (April Dunford methodology), ICP definition and validation, competitive intelligence and battlecards, GTM strategy and motion selection (PLG, sales-led, marketing-led, community-led), product launch planning, sales enablement, win/loss analysis, international expansion planning, messaging hierarchy, PMM KPIs.
**Out of Scope:** Brand identity and visual design (see brand-guidelines skill), demand generation execution (see marketing-demand-acquisition skill), content creation (see content-creator skill), pricing strategy optimization, sales process design.
**Limitations:** Positioning frameworks require real customer input to be effective — internally generated positioning is unreliable. Win/loss analysis requires honest deal outcome data from sales; incomplete data produces misleading patterns. GTM motion recommendations are based on ACV and buyer type heuristics; edge cases may require hybrid approaches. International expansion timelines assume US-first model and may not apply to non-US companies.
---
```

---

### 示例（按需加载）

**说明：** 这部分内容包含代码示例，在用户需要参考示例时按需加载。

**统计：** 2 个块，共 239 tokens

#### 块 1：`chunk_3e926d56`（113 tokens）

```markdown
### Market Priority (Series A)
| Market | Timeline | Budget % | Target ARR |
|--------|----------|----------|------------|
| US | Months 1-6 | 50% | $1M |
| UK | Months 4-9 | 20% | $500k |
| DACH | Months 7-12 | 15% | $300k |
| France | Months 10-15 | 10% | $200k |
| Canada | Months 7-12 | 5% | $100k |
```

#### 块 2：`chunk_2b540ed6`（126 tokens）

```markdown
## Scripts
| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/gtm_planner.py` | Generate GTM plans with motion selection, channel strategy, and timeline | `python scripts/gtm_planner.py config.json --demo` |
| `scripts/win_loss_analyzer.py` | Analyze deal outcomes by competitor, segment, and reason | `python scripts/win_loss_analyzer.py deals.json --demo` |
| `scripts/battlecard_generator.py` | Generate competitive battlecards with feature comparison and objection handling | `python scripts/battlecard_generator.py competitor.json --demo` |
```

---

### 模板（按需加载）

**说明：** 这部分内容包含模板样板，在用户需要使用模板时按需加载。

**统计：** 8 个块，共 551 tokens

#### 块 1：`chunk_fdb81b38`（86 tokens）

```markdown
### Firmographics Template
| Dimension | Target Range | Rationale |
|-----------|--------------|-----------|
| Employees | 50-5000 | Series A sweet spot |
| Revenue | $5M-$500M | Budget available |
| Industry | SaaS, Tech, Services | Product fit |
| Geography | US, UK, DACH | Market priority |
| Funding | Seed to Growth | Willing to adopt |
```

#### 块 2：`chunk_50c30332`（36 tokens）

```markdown
### Buyer Personas
**Economic Buyer** (signs contract):
**Technical Buyer** (evaluates product):
**User/Champion** (advocates internally):
```

#### 块 3：`chunk_06bc4d15`（47 tokens）

```markdown
### Positioning Statement Template
```
FOR [target customer]
WHO [statement of need]
THE [product] IS A [category]
THAT [key benefit]
UNLIKE [competitive alternative]
OUR PRODUCT [primary differentiation]

```
```

#### 块 4：`chunk_edff49eb`（50 tokens）

```markdown
### Value Proposition Formula
Template: `[Product] helps [Target Customer] [Achieve Goal] by [Unique Approach]`
Example: "Acme helps mid-market SaaS teams ship 2x faster by automating project workflows with AI"
```

#### 块 5：`chunk_59263fc5`（105 tokens）

```markdown
### Messaging Hierarchy
| Level | Content | Example |
|-------|---------|---------|
| Headline | 5-7 words | "Ship faster with AI automation" |
| Subhead | 1 sentence | "Automate workflows so teams focus on what matters" |
| Benefits | 3-4 bullets | Speed, quality, collaboration, cost |
| Features | Supporting evidence | AI automation → 10 hrs/week saved |
| Proof | Social proof | Customer logos, stats, case studies |
---
```

#### 块 6：`chunk_f771e2e8`（154 tokens）

```markdown
### Battlecard Template
```
COMPETITOR: [Name]
OVERVIEW: Founded [year], Funding [stage], Size [employees]

POSITIONING:
- They say: "[Their claim]"
- Reality: [Your assessment]

STRENGTHS:
1. [What they do well]
2. [What they do well]

WEAKNESSES:
1. [Where they fall short]
2. [Where they fall short]

OUR ADVANTAGES:
1. [Your advantage + evidence]
2. [Your advantage + evidence]

WHEN WE WIN:
- [Scenario where you win]

WHEN WE LOSE:
- [Scenario where they win]

TALK TRACK:
Objection: "[Common objection]"
Response: "[Your response]"

```
```

#### 块 7：`chunk_de58281e`（68 tokens）

```markdown
### Sales-Marketing Handoff
| Handoff | Frequency | Content |
|---------|-----------|---------|
| Weekly sync | 30 min | Win/loss, competitive, new assets |
| Monthly enablement | 60 min | Product updates, training |
| Quarterly review | Half-day | Results, strategy, planning |
---
```

#### 块 8：`chunk_dc761114`（5 tokens）

```markdown
### Localization Checklist
---
```

---

### 冗余内容（丢弃）

**说明：** 这部分内容被判定为冗余，将在压缩过程中丢弃。

**统计：** 1 个块，共 6 tokens

#### 块 1：`chunk_f4f98776`（6 tokens）

```markdown
## Table of Contents
---
```

---

## 第五部分：压缩结果

### Token 减少分析

| 指标 | 压缩前 | 压缩后 | 减少量 |
|------|--------|--------|--------|
| Body Token 数 | 3652 | 776 | 2876 (78.8%) |

### Progressive Disclosure（渐进式披露）结构

``
压缩后 Skill 结构：

├─ 阶段 1：基础加载（始终注入）
│   └─ 核心规则：776 tokens

├─ 阶段 2：按需加载模块
│   ├─ 背景知识：621 tokens
│   ├─ 示例：239 tokens
│   └─ 模板：551 tokens
│   └─ 按需模块总计：1411 tokens

└─ 阶段 3：丢弃
    └─ 冗余内容：6 tokens
```

### 模块加载触发条件

| 模块 | 触发条件 | 说明 |
|------|----------|------|
| 背景知识 | 用户询问"为什么"或需要概念解释时 | 仅在需要时加载 |
| 示例 | 用户请求代码示例或演示时 | 仅在需要时加载 |
| 模板 | 用户需要样板或填空模板时 | 仅在需要时加载 |

### 压缩指标

- **原始大小：** 3652 tokens
- **压缩后大小：** 776 tokens
- **压缩率：** 64.6%
- **每次调用节省 Token：** 2876

---

## 第六部分：最终压缩后的 Skill

### 核心规则部分（始终加载）

以下是每次调用都会注入到 LLM 上下文的内容：

```markdown
## ICP Definition Workflow
Define ideal customer profile for targeting:

### ICP Validation Checklist
---

## Positioning Development
Develop positioning using April Dunford methodology:

## Competitive Intelligence
Build competitive knowledge base:

### Win/Loss Analysis
Track monthly:
---

### Tier 1 Launch Workflow
Execute major product launch:

### Launch Day Checklist

## Sales Enablement
Equip sales team with PMM assets:

### Demo Flow
```
1. Intro (2 min): Who we are, agenda
2. Discovery (5 min): Their needs, pain points
3. Demo (20 min): Product focused on their use case
4. Q&A (10 min): Objection handling
5. Next steps (3 min): Trial, POC, proposal

```

## International Expansion
Enter new markets systematically:

## PMM KPIs
| Metric | Target | Measurement |
|--------|--------|-------------|
| Product adoption | >40% in 90 days | Feature usage after launch |
| Win rate | >30% competitive | Deals won vs. competitors |
| Sales velocity | -20% YoY | Days from SQL to close |
| Deal size | +25% YoY | Average contract value |
| Launch pipeline | 3:1 ROMI | Pipeline $ : marketing spend |
---

### PMM Monthly Rhythm
| Week | Focus |
|------|-------|
| 1 | Review metrics, update battlecards |
| 2 | Create assets, publish content |
| 3 | Support launches, optimize campaigns |
| 4 | Monthly report, plan next month |

## Output Artifacts
| When you ask for... | You get... |
|---------------------|------------|
| "Position my product" | Positioning framework (April Dunford method) with completed output |
| "GTM strategy" | Go-to-market plan with channels, messaging, and timeline |
| "Competitive positioning" | Positioning map with competitive gaps and opportunities |
| "Sales enablement" | Sales deck structure, battlecards, and demo flow |

## Communication
All output passes quality verification:

## Troubleshooting
| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| Positioning resonates internally but customers do not repeat it | Positioning built from company perspective, not customer language | Rerun April Dunford methodology starting from competitive alternatives, not from product features |
| Win rate against specific competitor below 25% | Battlecard outdated or sales team not using it | Run win_loss_analyzer.py to identify loss patterns; update battlecard monthly; validate 80%+ sales usage |
| GTM motion producing high MQLs but low pipeline conversion | Wrong GTM motion for ACV and buyer type; marketing-led when should be sales-led | Reassess motion using gtm_planner.py; for ACV >$25K, shift to sales-led or hybrid PLG+sales |
| Sales enablement assets gathering dust | Assets created without sales input; format does not match how sales actually works | Co-create assets with sales; survey sales on what they need; track asset usage in deal cycles |
| International expansion burning cash with zero pipeline | Market entered without validating demand (inbound signal, TAM) | Validate 3+ paying customers from market in first 90 days; if not, pause and reassess market priority |
| Competitive intelligence always reactive to lost deals | No proactive monitoring system; battlecards only updated post-loss | Set up monthly competitor monitoring (website, pricing, job postings, G2 reviews); update battlecards proactively |
| Messaging differs across website, sales deck, and ads | No messaging hierarchy documented; each team creates independently | Build messaging hierarchy (headline > subhead > benefits > features > proof) and enforce across all touchpoints |
---

```

**核心规则总 Token 数：** 776

### 按需加载模块（参考）

以下模块仅在特定需要时加载：

#### 背景知识模块
```markdown
# Marketing Strategy & PMM
Product marketing patterns for positioning, GTM strategy, and competitive intelligence.
---

### Competitive Tier Structure
| Tier | Definition | Examples |
|------|------------|----------|
| 1 | Direct competitor, same category | [Competitor A, B] |
| 2 | Adjacent solution, overlapping use case | [Alt Solution C, D] |
| 3 | Status quo (what they do today) | Spreadsheets, manual, in-house |

## Product Launch Planning
Plan launches by tier:
| Tier | Scope | Prep Time | Budget |
|------|-------|-----------|--------|
| 1 | New product, major feature | 6-8 weeks | $50-100k |
| 2 | Significant feature, integration | 3-4 weeks | $10-25k |
| 3 | Small improvement | 1 week | <$5k |

### Launch Metrics
| Metric | Leading (Daily) | Lagging (Weekly) |
|--------|-----------------|------------------|
| Traffic | Landing page visitors | - |
| Engagement | Demo requests, signups | Feature adoption % |
| Pipeline | MQLs generated | SQLs, pipeline $ |
| Revenue | - | Deals closed, revenue |
---

### Sales Deck Structure
| Slide | Content |
|-------|---------|
| 1-2 | Title, agenda |
| 3-4 | Company intro, problem statement |
| 5-7 | Solution, key benefits, demo |
| 8-10 | Differentiation, case study, pricing |
| 11-12 | Implementation, support, next steps |

... （还有 10 个背景知识块）
```

#### 模板模块
```markdown
### Firmographics Template
| Dimension | Target Range | Rationale |
|-----------|--------------|-----------|
| Employees | 50-5000 | Series A sweet spot |
| Revenue | $5M-$500M | Budget available |
| Industry | SaaS, Tech, Services | Product fit |
| Geography | US, UK, DACH | Market priority |
| Funding | Seed to Growth | Willing to adopt |

### Buyer Personas
**Economic Buyer** (signs contract):
**Technical Buyer** (evaluates product):
**User/Champion** (advocates internally):

### Positioning Statement Template
```
FOR [target customer]
WHO [statement of need]
THE [product] IS A [category]
THAT [key benefit]
UNLIKE [competitive alternative]
OUR PRODUCT [primary differentiation]

```

... （还有 5 个模板块）
```

#### 示例模块
```markdown
### Market Priority (Series A)
| Market | Timeline | Budget % | Target ARR |
|--------|----------|----------|------------|
| US | Months 1-6 | 50% | $1M |
| UK | Months 4-9 | 20% | $500k |
| DACH | Months 7-12 | 15% | $300k |
| France | Months 10-15 | 10% | $200k |
| Canada | Months 7-12 | 5% | $100k |

## Scripts
| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/gtm_planner.py` | Generate GTM plans with motion selection, channel strategy, and timeline | `python scripts/gtm_planner.py config.json --demo` |
| `scripts/win_loss_analyzer.py` | Analyze deal outcomes by competitor, segment, and reason | `python scripts/win_loss_analyzer.py deals.json --demo` |
| `scripts/battlecard_generator.py` | Generate competitive battlecards with feature comparison and objection handling | `python scripts/battlecard_generator.py competitor.json --demo` |

```

---

## 总结

| 指标 | 值 |
|------|------|
| Skill 名称 | marketing-strategy-pmm |
| 原始 Token 数 | 3652 |
| 压缩后 Token 数 | 776 |
| 压缩率 | 64.6% |
| 核心规则占比 | 21.2% |
| 每次调用预估节省成本 | $0.0288 |

### 与论文发现对比

根据 SkillReducer 论文（Gao 等，2026）：
- **平均核心规则占比：** 38.5%
- **本 Skill 核心规则占比：** 21.2%
- **平均压缩率：** 39%
- **本 Skill 压缩率：** 64.6%

本 Skill 的压缩率高于平均水平，说明包含更多背景知识和模板内容。

### 压缩流程回顾

``
SkillReducer 压缩流程：

1. 解析 Skill 文件
   └─ 提取 YAML frontmatter 和 Markdown Body

2. Markdown 切分
   └─ 按标题边界切分为语义块
   └─ 生成 41 个 Chunk

3. LLM 分类
   └─ 分批调用 LLM 进行 taxonomy 分类
   └─ 分类为：核心规则、背景知识、示例、模板、冗余

4. Progressive Disclosure 重构
   └─ 核心规则始终加载
   └─ 背景知识、示例、模板按需加载
   └─ 冗余内容丢弃

5. 压缩结果
   └─ 从 3652 tokens 压缩到 776 tokens
   └─ 压缩率 64.6%
```

---

*本报告由 SkillReducer 框架生成*