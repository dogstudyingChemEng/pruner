# On Demand Examples

This file is loaded on-demand when needed.

---

<!--
WHEN: Load this when user asks about churn reduction impact, dunning email optimization, exit survey analysis, or save offers to prevent churn.
TOPICS: churn prevention, dunning sequences, exit survey, save offers, churn impact
-->

## Example 1

Calculate revenue impact of churn reduction.
Usage: python scripts/churn_impact_calculator.py --mrr 500000 --churn-rate 4.0 --save-rate 20
Args: --mrr (required), --churn-rate (required), --save-rate (default: 15), --target-churn (default: current-1), --json (optional output).

---

## Example 2

Analyze dunning email sequence effectiveness and recommend retry timing optimizations.
Usage: python scripts/dunning_sequence_analyzer.py dunning_data.json
Input JSON: list of failed_payments with payment_id, amount, failure_reason, retry_attempts (each with day, recovered).
Args: --json (optional output).

---

## Example 3

Analyze exit survey responses to identify churn patterns, save offer effectiveness, and systemic issues.
Usage: python scripts/exit_survey_analyzer.py survey_data.json
Args: --json (optional output), --period (default: 'current').