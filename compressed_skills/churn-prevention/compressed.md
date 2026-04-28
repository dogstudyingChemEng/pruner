---
name: churn-prevention
description: |
  SaaS churn reduction framework (voluntary & involuntary) via cancel flow optimization, dynamic save offers, exit survey design, dunning, payment recovery, win-backs, and churn impact modeling.
metadata:
  version: 1.0.0
  category: business-growth
  tags: ["churn", "retention", "cancel-flow", "dunning", "payment-recovery", "win-back"]
---

Collect monthly churn (voluntary vs involuntary), cancel flow type, payment processor, ACV, billing cycle, MRR, SaaS model, and exit reasons to determine levers and impact.
Use a single required exit survey with 8 radio-button reasons (PRICE, LOW_USAGE, MISSING_FEATURE, COMPETITOR, PROJECT_END, COMPLEXITY, TESTING, OTHER) and map each to a specific save offer.
For PRICE: offer 30-50% discount 2-3 months; if ARPU < median, offer downgrade instead.
For LOW_USAGE: pause if last login >30 days; else offer usage tips + discount.
For MISSING_FEATURE: share roadmap if planned, else offer discount.
For COMPETITOR: show comparison + retention offer; if unknown, general retention.
For PROJECT_END: always offer pause.
For COMPLEXITY: offer free onboarding for high-value, tutorial for low-value.
For TESTING: no offer, let go gracefully.
For OTHER: general retention offer (contact support).
Send post-cancel emails: immediate (confirmation + reactivation link), day 7 (miss you + CTA), day 30 (product update + offer), day 60 (final win-back).
Track exit reason distribution, save rate by reason, reason trends, and feature gap frequency monthly; escalate if reason >30% or save rate <5%.
Use smart retry schedule for failed payments: retry days 3, 7, 12, 18; service action on day 21.
Enable card updater per processor: Stripe automatic, Braintree enable, Paddle built-in, Recurly configure.
Send dunning emails on days 0, 3, 7, 14, 21 with specific subject, body, and CTA.
Calculate risk score (0-20 low, 21-40 moderate, 41-60 high, 61+ critical) using signal weights.
Monitor red flags: save rate <5%, survey completion <60%, recovery <20%, single reason >40%, churn >5% monthly; escalate or rebuild accordingly.
Estimate churn impact: monthly MRR at risk = total MRR × churn rate; annual saved by 1% reduction = total MRR × 0.01 × 12; annual saved by 20% save rate = (at risk × 0.20) × 12.
Troubleshoot: low save rate → rebuild mapping; low survey completion → make required; low recovery → audit dunning; single reason >40% → escalate; churn >5% → review ICP; win-back zero → check deliverability; involuntary churn rising → enable card updater/retry.

---
## On-Demand Modules

- `references/on-demand-examples.md` — Examples
- `references/on-demand-templates.md` — Templates
- `references/on-demand-background.md` — Background
