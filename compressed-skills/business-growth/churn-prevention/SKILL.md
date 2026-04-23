---
name: churn-prevention
description: |
  Develop and execute win-back campaigns and model churn impact to re-engage customers and mitigate attrition.
metadata:
  version: 1.0.0
  category: business-growth
  tags: ["churn", "retention", "cancel-flow", "dunning", "payment-recovery", "win-back"]
---

• Collect monthly churn rate (voluntary/involuntary), cancel flow type, payment processor, ACV, billing cycle, MRR, SaaS model, and exit reason availability.
• Exit survey: 1 required radio question with max 8 options (PRICE, LOW_USAGE, MISSING_FEATURE, COMPETITOR, PROJECT_END, COMPLEXITY, TESTING, OTHER).
• Map each exit reason to one save offer: PRICE→discount, LOW_USAGE→pause, MISSING_FEATURE→roadmap+workaround, COMPETITOR→comparison+discount, PROJECT_END→pause, COMPLEXITY→onboarding, TESTING→no offer, OTHER→general retention.
• Post-cancel emails: immediate confirmation, day 7 reactivation, day 30 update+offer, day 60 final win-back.
• Track exit survey metrics: reason distribution (>30% systemic), save rate by reason (<5% wrong offer), reason trend, top 3 missing features.
• Save offer decision tree: PRICE→discount if ARPU>median else downgrade; LOW_USAGE→pause if login>30d else tips+discount; MISSING_FEATURE→roadmap if planned else discount; COMPETITOR→comparison if known else general; PROJECT_END→pause; COMPLEXITY→onboarding if enterprise else tutorial; TESTING→no offer.
• Smart retry schedule: initial day 0, retry day 3, 7, 12, 18, service action day 21.
• Enable card updater: Stripe (default), Braintree (enable), Paddle (built-in), Recurly (configure).
• Dunning emails: day 0 factual, day 3 loss reminder, day 7 urgency, day 14 final notice, day 21 paused.
• Win-back timing: day 7 (5-10%), day 30 (3-7%), day 60 (2-5%), day 90+ (1-3%).
• Leading churn indicators: login decline, feature drop, support escalation, NPS<7, invoice dispute, champion left, renewal<90d, competitor eval.
• Risk score: sum of signal weights (0-20 low, 21-40 moderate, 41-60 high, 61+ critical).
• Key metrics: save rate (10-15% good, 20%+ excellent), voluntary churn (<3% good, <1.5% excellent), involuntary churn (<1.5% good, <0.5% excellent), payment recovery (25-35% good, 40%+ excellent), win-back (5-10% good, 10%+ excellent), exit survey completion (>70% good, >90% excellent), offer acceptance (15-25% good, 30%+ excellent).
• Red flags: save rate<5%→rebuild mapping; survey<60%→make required; recovery<20%→audit dunning; single reason>40%→escalate; churn>5%→review ICP/product.
• Troubleshooting: low save rate→rebuild mapping; low survey→make required; low recovery→audit dunning; high single reason→escalate; high churn→review ICP; zero win-back→check deliverability; rising involuntary→enable card updater.

### Stage 4: Confirmation
If they decline the save offer or there is no offer to make:
```
┌────────────────────────────────────────┐
│  We're sorry to see you go             │
│                                        │
│  What happens when you cancel:         │
│  - Your data is saved for 90 days     │
│  - Access continues until [date]      │
│  - You can reactivate anytime         │
│                                        │
│  [Yes, Cancel My Account]             │
│  [Wait, I Changed My Mind]            │
│                                        │
│  No pre-checked boxes.                │
│  No confusing language.               │
└────────────────────────────────────────┘
```

### 1. churn_impact_calculator.py
**Purpose:** Calculate the revenue impact of churn reduction at various improvement levels.
```bash
python scripts/churn_impact_calculator.py --mrr 500000 --churn-rate 4.0 --save-rate 20
python scripts/churn_impact_calculator.py --mrr 500000 --churn-rate 4.0 --save-rate 20 --json
```
| Flag | Required | Description |
|------|----------|-------------|
| `--mrr` | Yes | Current monthly recurring revenue in dollars |
| `--churn-rate` | Yes | Current monthly churn rate as percentage (e.g., 4.0 for 4%) |
| `--save-rate` | No | Cancel flow save rate as percentage (default: 15) |
| `--target-churn` | No | Target churn rate as percentage (default: current minus 1) |
| `--json` | No | Output results as JSON |

### 2. dunning_sequence_analyzer.py
**Purpose:** Analyze dunning email sequence effectiveness and recommend retry timing optimizations.
```bash
python scripts/dunning_sequence_analyzer.py dunning_data.json
python scripts/dunning_sequence_analyzer.py dunning_data.json --json
```
| Flag | Required | Description |
|------|----------|-------------|
| `dunning_data.json` | Yes | JSON file with failed payment and retry data |
| `--json` | No | Output results as JSON |
**Input JSON format:**
```json
{
  "failed_payments": [
    {
      "payment_id": "PAY-001",
      "amount": 99.00,
      "failure_reason": "expired_card",
      "retry_attempts": [
        {"day": 0, "recovered": false},
        {"day": 3, "recovered": false},
        {"day": 7, "recovered": true}
      ]
    }
  ]
}
```

### 3. exit_survey_analyzer.py
**Purpose:** Analyze exit survey responses to identify churn patterns, save offer effectiveness, and systemic issues.
```bash
python scripts/exit_survey_analyzer.py survey_data.json
python scripts/exit_survey_analyzer.py survey_data.json --json
```
| Flag | Required | Description |
|------|----------|-------------|
| `survey_data.json` | Yes | JSON file with exit survey response data |
| `--json` | No | Output results as JSON |
| `--period` | No | Analysis period label (default: "current") |

### Win-Back Email Sequence
**Day 7 Email:**
**Day 30 Email:**
**Day 60 Email:**
---

A production-grade SaaS churn reduction framework addresses both voluntary churn (value gap, product-market mismatch, competitor switch, budget cut, project completion, poor experience, never activated) and involuntary churn (expired card 40-50%, insufficient funds 20-30%, bank decline 10-15%, account closed 5-10%, network error 5-10%). The framework includes cancel flow architecture, exit survey design with competitive intelligence, dynamic save offer system with economics (e.g., 30% discount for 3 months saves 15-25%, pause saves 25-40%), dunning sequence engineering (failed payments cause 20-40% of total churn), win-back campaigns, churn health scoring, metrics and benchmarks, and a churn impact calculator. Output artifacts include cancel flow design, exit survey, save offer decision tree, dunning sequence (5 emails), win-back campaign (3 emails at days 7, 30, 60), churn scorecard, and impact model.