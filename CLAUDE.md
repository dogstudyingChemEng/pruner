# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains the research paper "SkillReducer: Optimizing LLM Agent Skills for Token Efficiency" by Gao et al. (2026). The paper presents a framework for debloating LLM agent skills to reduce token costs while preserving functional quality.

## Key Concepts

### Skills Architecture
Skills consist of four components:
- **Description (s.d)**: Short routing text used to match user queries
- **Body (s.b)**: Main instruction document injected into context
- **References (s.R)**: Optional files loaded alongside body
- **Scripts**: Executable code (out of scope for this work)

### SkillReducer Framework (Two Stages)

**Stage 1: Routing Layer Optimization**
- Delta debugging with simulated oracle for description compression
- Phase 1: Fast compression via DDMIN algorithm on semantic clauses
- Phase 2: Real-environment validation with selective recovery
- Handles missing descriptions via generation from body content
- Goal: 1-minimal descriptions that preserve routing equivalence

**Stage 2: Body Restructuring via Progressive Disclosure**
- Taxonomy-driven classification into five content types:
  - Core rules (actionable instructions) - always loaded
  - Background (explanations) - on-demand module
  - Examples (code snippets) - on-demand module
  - Templates (boilerplate) - on-demand module
  - Redundant content - discarded
- Type-specific compression with faithfulness checks
- Cross-file deduplication between body and references
- Quality gates: Gate 1 (faithfulness), Gate 2 (task-based evaluation with feedback loop)

### Key Metrics
- **Compression ratio**: Token reduction percentage
- **Retention**: score_C / score_A (compressed vs original performance)
- **Pass rate**: Fraction of skills where score_C >= score_A

## Main Findings

- **Description compression**: 48% mean reduction (56.5% for existing, 44.6% for generated)
- **Body compression**: 39% mean reduction
- **Quality preservation**: 86.0% pass rate, 100% on SkillsBench
- **Less-is-more effect**: Compressed skills improve performance by 2.8% (removing distracting content)
- **True compression failures**: Only 4.7% of skills
- **Cross-model transfer**: Mean retention 0.965 across five models from four families
- **Cross-framework transfer**: Retention 0.944 on OpenCode agent framework

## Empirical Study Findings

Analysis of 55,315 public skills revealed:
- 26.4% lack descriptions entirely (routing failure)
- Only 38.5% of body content is actionable core rules
- 40.7% is background, 12.9% examples, 7.6% templates
- Reference-heavy skills can inject tens of thousands of tokens per invocation

## Working with This Repository

This is a research paper directory. There is no code to build, test, or run. The PDF contains the full paper with methodology, evaluation results, and appendices including a running example (marketing-strategy-pmm skill).