"""
Quality Gates for SkillReducer framework.

Implements faithfulness verification and feedback loop mechanisms
from the SkillReducer paper to ensure compression quality.
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from .models import Skill, ContentBlock, ContentType, Description, Body, References
from .stage1_router import Stage1Optimizer
from .optimizer import Stage2Optimizer, CompressionMetrics


def _is_content_type(block: ContentBlock, content_type: ContentType) -> bool:
    """
    Check if a block's content_type matches the given type.

    Handles both enum and string values due to Pydantic's use_enum_values.
    """
    if isinstance(block.content_type, str):
        return block.content_type == content_type.value
    return block.content_type == content_type


class GateResult(str, Enum):
    """Result of a quality gate check."""

    PASSED = "passed"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class FaithfulnessResult:
    """Result of faithfulness verification (Gate 1)."""

    passed: bool
    gate_result: GateResult
    preserved_concepts: list[str]
    missing_concepts: list[str]
    reasoning: str
    should_rollback: bool


@dataclass
class FeedbackLoopResult:
    """Result of feedback loop promotion."""

    promoted_blocks: list[str]  # Block IDs that were promoted
    promotion_count: int
    new_core_tokens: int
    feedback_addressed: list[str]


@dataclass
class PipelineResult:
    """Result of the complete compression pipeline."""

    skill: Skill
    stage1_compressed: bool
    stage2_compressed: bool
    faithfulness_passed: bool
    rollback_performed: bool
    compression_metrics: Optional[CompressionMetrics]
    original_tokens: int
    final_tokens: int
    overall_compression_ratio: float
    errors: list[str] = field(default_factory=list)


class QualityGates:
    """
    Quality gates for SkillReducer compression pipeline.

    Implements Gate 1 (Faithfulness Verification) and Gate 2 (Feedback Loop)
    from the SkillReducer paper.
    """

    def __init__(
        self,
        llm_client,
        faithfulness_threshold: float = 1.0
    ):
        """
        Initialize quality gates.

        Args:
            llm_client: LLM client for verification.
            faithfulness_threshold: Minimum ratio of preserved concepts (0.0-1.0).
                                  Default 1.0 means all concepts must be preserved.
        """
        self.llm_client = llm_client
        self.faithfulness_threshold = faithfulness_threshold

    # ============================================================
    # Gate 1: Faithfulness Verification
    # ============================================================

    def verify_faithfulness(
        self,
        original_body: str,
        compressed_core_rules: str,
        skill_name: str = "unknown"
    ) -> FaithfulnessResult:
        """
        Gate 1: Verify that all core operational concepts are preserved.

        Checks if all 'core operational concepts' from the original body
        are preserved in the compressed core rules. If concepts are missing,
        triggers rollback.

        Args:
            original_body: The original skill body text.
            compressed_core_rules: The compressed core rules.
            skill_name: Skill name for context.

        Returns:
            FaithfulnessResult with verification details.
        """
        if not original_body or not original_body.strip():
            return FaithfulnessResult(
                passed=True,
                gate_result=GateResult.PASSED,
                preserved_concepts=[],
                missing_concepts=[],
                reasoning="Empty original body, verification skipped",
                should_rollback=False
            )

        if not compressed_core_rules or not compressed_core_rules.strip():
            return FaithfulnessResult(
                passed=False,
                gate_result=GateResult.FAILED,
                preserved_concepts=[],
                missing_concepts=["No compressed rules provided"],
                reasoning="Compressed rules are empty",
                should_rollback=True
            )

        system_prompt = """You are a quality assurance expert for technical documentation compression.

Your task is to verify that all CORE OPERATIONAL CONCEPTS from the original document
are preserved in the compressed version.

Core operational concepts include:
- Specific actions or steps that must be performed
- Rules and constraints that must be followed
- Required procedures or workflows
- Key parameters, thresholds, or limits
- Critical conditions or triggers

Do NOT flag as missing:
- Background information or explanations
- Examples or demonstrations
- Optional details or nice-to-have context
- Redundant or repeated information

IMPORTANT: Respond with a JSON object:
{
  "original_concepts": [
    "concept 1 from original",
    "concept 2 from original"
  ],
  "preserved_concepts": [
    "concept that was preserved in compressed"
  ],
  "missing_concepts": [
    "concept that was lost in compression"
  ],
  "faithfulness_score": 0.95,
  "reasoning": "Explanation of the verification result"
}"""

        user_prompt = f"""Skill Name: {skill_name}

ORIGINAL BODY:
{original_body[:4000]}

COMPRESSED CORE RULES:
{compressed_core_rules}

Identify all core operational concepts in the original and check if they are preserved in the compressed version."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        original_concepts = result.get("original_concepts", [])
        preserved_concepts = result.get("preserved_concepts", [])
        missing_concepts = result.get("missing_concepts", [])
        faithfulness_score = result.get("faithfulness_score", 1.0)
        reasoning = result.get("reasoning", "")

        # Determine if passed
        passed = len(missing_concepts) == 0 and faithfulness_score >= self.faithfulness_threshold

        should_rollback = len(missing_concepts) > 0

        return FaithfulnessResult(
            passed=passed,
            gate_result=GateResult.PASSED if passed else GateResult.ROLLBACK,
            preserved_concepts=preserved_concepts,
            missing_concepts=missing_concepts,
            reasoning=reasoning,
            should_rollback=should_rollback
        )

    # ============================================================
    # Gate 2: Feedback Loop (Task-based Evaluation)
    # ============================================================

    def feedback_loop(
        self,
        blocks: list[ContentBlock],
        failed_criteria: list[str],
        skill_context: Optional[str] = None
    ) -> FeedbackLoopResult:
        """
        Gate 2: Promote relevant blocks to core_rule based on task failure feedback.

        When task evaluation fails, this method identifies example/background
        blocks related to the failure criteria and promotes them to core_rule
        type to ensure they are always loaded.

        Args:
            blocks: List of classified content blocks.
            failed_criteria: List of criteria that caused task failure.
            skill_context: Optional context about the skill.

        Returns:
            FeedbackLoopResult with promotion details.
        """
        if not failed_criteria or not blocks:
            return FeedbackLoopResult(
                promoted_blocks=[],
                promotion_count=0,
                new_core_tokens=sum(
                    b.token_count for b in blocks
                    if _is_content_type(b, ContentType.CORE_RULE)
                ),
                feedback_addressed=[]
            )

        # Identify blocks that are relevant to failed criteria
        promotable_types = [ContentType.EXAMPLE, ContentType.BACKGROUND, ContentType.TEMPLATE]
        promotable_type_values = [t.value for t in promotable_types]
        promotable_blocks = [b for b in blocks if (
            b.content_type in promotable_types or
            (isinstance(b.content_type, str) and b.content_type in promotable_type_values)
        )]

        if not promotable_blocks:
            return FeedbackLoopResult(
                promoted_blocks=[],
                promotion_count=0,
                new_core_tokens=sum(
                    b.token_count for b in blocks
                    if _is_content_type(b, ContentType.CORE_RULE)
                ),
                feedback_addressed=[]
            )

        system_prompt = """You are an expert at analyzing content relevance to task failures.

Given a list of failed task criteria and content blocks, identify which blocks
contain information that would help address the failure.

Guidelines:
1. A block is relevant if it contains information directly related to the failed criterion
2. Examples showing the correct approach are relevant
3. Background explaining a concept that was misunderstood is relevant
4. Templates that provide the correct format are relevant

IMPORTANT: Respond with a JSON object:
{
  "relevant_blocks": [
    {
      "block_id": "chunk_xxx",
      "relevant_to": ["criterion that this block addresses"],
      "reason": "Why this block is relevant"
    }
  ],
  "criteria_addressed": ["criterion1", "criterion2"]
}"""

        # Format blocks and criteria
        blocks_text = "\n\n".join([
            f"[{b.chunk_id}] (Type: {b.content_type if isinstance(b.content_type, str) else b.content_type.value})\n{b.content[:500]}"
            for b in promotable_blocks
        ])

        criteria_text = "\n".join([
            f"- {criterion}"
            for criterion in failed_criteria
        ])

        user_prompt = f"""Failed Task Criteria:
{criteria_text}

Available Content Blocks (can be promoted to core_rule):
{blocks_text}

{f'Skill Context: {skill_context}' if skill_context else ''}

Identify which blocks contain information relevant to addressing these failures."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        relevant_blocks = result.get("relevant_blocks", [])
        criteria_addressed = result.get("criteria_addressed", [])

        # Promote relevant blocks
        promoted_ids = []
        for rel_block in relevant_blocks:
            block_id = rel_block.get("block_id", "")
            for block in blocks:
                is_promotable = (
                    block.content_type in promotable_types or
                    (isinstance(block.content_type, str) and block.content_type in promotable_type_values)
                )
                if block.chunk_id == block_id and is_promotable:
                    # Promote to core_rule
                    block.content_type = ContentType.CORE_RULE
                    promoted_ids.append(block_id)
                    break

        # Calculate new core tokens
        new_core_tokens = sum(
            b.token_count for b in blocks
            if _is_content_type(b, ContentType.CORE_RULE)
        )

        return FeedbackLoopResult(
            promoted_blocks=promoted_ids,
            promotion_count=len(promoted_ids),
            new_core_tokens=new_core_tokens,
            feedback_addressed=criteria_addressed
        )

    # ============================================================
    # Utility Methods
    # ============================================================

    def run_faithfulness_gate(
        self,
        skill: Skill,
        compressed_blocks: list[ContentBlock]
    ) -> FaithfulnessResult:
        """
        Run Gate 1 on a skill with compressed blocks.

        Args:
            skill: The original skill.
            compressed_blocks: The compressed content blocks.

        Returns:
            FaithfulnessResult.
        """
        # Extract compressed core rules
        core_rules = "\n".join([
            b.content for b in compressed_blocks
            if _is_content_type(b, ContentType.CORE_RULE)
        ])

        return self.verify_faithfulness(
            original_body=skill.body.original,
            compressed_core_rules=core_rules,
            skill_name=skill.name
        )


class CompressionPipeline:
    """
    Complete compression pipeline combining Stage 1, Stage 2, and Quality Gates.
    """

    def __init__(
        self,
        llm_client,
        stage1_optimizer: Optional[Stage1Optimizer] = None,
        stage2_optimizer: Optional[Stage2Optimizer] = None,
        quality_gates: Optional[QualityGates] = None,
        enable_gate1: bool = True,
        enable_gate2: bool = True
    ):
        """
        Initialize the compression pipeline.

        Args:
            llm_client: LLM client for all operations.
            stage1_optimizer: Optional Stage1Optimizer instance.
            stage2_optimizer: Optional Stage2Optimizer instance.
            quality_gates: Optional QualityGates instance.
            enable_gate1: Whether to enable Gate 1 (faithfulness).
            enable_gate2: Whether to enable Gate 2 (feedback loop).
        """
        self.llm_client = llm_client
        self.stage1_optimizer = stage1_optimizer or Stage1Optimizer(llm_client)
        self.stage2_optimizer = stage2_optimizer or Stage2Optimizer(llm_client)
        self.quality_gates = quality_gates or QualityGates(llm_client)
        self.enable_gate1 = enable_gate1
        self.enable_gate2 = enable_gate2

    def run_pipeline(
        self,
        skill: Skill,
        compress_core: bool = True,
        dedup_examples: bool = True,
        dedup_templates: bool = True,
        summarize_background: bool = True,
        dedup_references: bool = True,
        failed_criteria: Optional[list[str]] = None
    ) -> PipelineResult:
        """
        Run the complete compression pipeline.

        Pipeline steps:
        1. Stage 1: Compress description
        2. Stage 2: Classify and compress body
        3. Gate 1: Verify faithfulness
        4. Gate 2: Apply feedback loop (if failed_criteria provided)
        5. Finalize results

        Args:
            skill: The skill to compress.
            compress_core: Whether to compress core rules.
            dedup_examples: Whether to deduplicate examples.
            dedup_templates: Whether to deduplicate templates.
            summarize_background: Whether to summarize background.
            dedup_references: Whether to deduplicate references.
            failed_criteria: Optional list of failed task criteria for Gate 2.

        Returns:
            PipelineResult with complete compression results.
        """
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")

        errors = []
        original_tokens = len(encoding.encode(skill.body.original))
        if skill.description.original:
            original_tokens += len(encoding.encode(skill.description.original))
        for ref_content in skill.references.files.values():
            original_tokens += len(encoding.encode(ref_content))

        stage1_compressed = False
        stage2_compressed = False
        faithfulness_passed = True
        rollback_performed = False
        compression_metrics = None
        final_blocks = []
        updated_references = skill.references

        try:
            # ============================================================
            # Stage 1: Description Compression
            # ============================================================
            try:
                skill = self.stage1_optimizer.compress_skill(skill)
                stage1_compressed = skill.description.compressed is not None
            except Exception as e:
                errors.append(f"Stage 1 failed: {str(e)}")

            # ============================================================
            # Stage 2: Body Classification and Compression
            # ============================================================
            try:
                final_blocks, updated_references, compression_metrics = self.stage2_optimizer.optimize_skill(
                    skill,
                    compress_core=compress_core,
                    dedup_examples=dedup_examples,
                    dedup_templates=dedup_templates,
                    summarize_background=summarize_background,
                    dedup_references=dedup_references
                )
                stage2_compressed = True
            except Exception as e:
                errors.append(f"Stage 2 failed: {str(e)}")
                # Use original blocks as fallback
                final_blocks = self.stage2_optimizer.classify_skill_body(skill)

            # ============================================================
            # Gate 1: Faithfulness Verification
            # ============================================================
            if self.enable_gate1 and stage2_compressed and final_blocks:
                faithfulness_result = self.quality_gates.run_faithfulness_gate(
                    skill, final_blocks
                )

                if faithfulness_result.should_rollback:
                    faithfulness_passed = False
                    rollback_performed = True

                    # Rollback: re-classify without compression
                    final_blocks = self.stage2_optimizer.classify_skill_body(skill)

                    errors.append(
                        f"Gate 1 failed - missing concepts: {faithfulness_result.missing_concepts}"
                    )

            # ============================================================
            # Gate 2: Feedback Loop (if failed criteria provided)
            # ============================================================
            if self.enable_gate2 and failed_criteria and final_blocks:
                feedback_result = self.quality_gates.feedback_loop(
                    final_blocks,
                    failed_criteria,
                    skill_context=f"{skill.name}: {skill.description.original}"
                )

                if feedback_result.promotion_count > 0:
                    errors.append(
                        f"Gate 2 promoted {feedback_result.promotion_count} blocks to core_rule"
                    )

        except Exception as e:
            errors.append(f"Pipeline error: {str(e)}")

        # Calculate final tokens
        final_tokens = sum(b.token_count for b in final_blocks)
        if skill.description.compressed:
            final_tokens += len(encoding.encode(skill.description.compressed))
        else:
            final_tokens += len(encoding.encode(skill.description.original))
        final_tokens += updated_references.total_token_count

        overall_ratio = 0.0
        if original_tokens > 0:
            overall_ratio = 1.0 - (final_tokens / original_tokens)

        # Update skill with results
        skill.body.content_blocks = final_blocks
        skill.references = updated_references

        return PipelineResult(
            skill=skill,
            stage1_compressed=stage1_compressed,
            stage2_compressed=stage2_compressed,
            faithfulness_passed=faithfulness_passed,
            rollback_performed=rollback_performed,
            compression_metrics=compression_metrics,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            overall_compression_ratio=round(overall_ratio, 4),
            errors=errors
        )


# ============================================================
# Convenience Functions
# ============================================================

def run_pipeline(
    skill: Skill,
    llm_client,
    enable_gate1: bool = True,
    enable_gate2: bool = True,
    failed_criteria: Optional[list[str]] = None
) -> PipelineResult:
    """
    Convenience function to run the complete compression pipeline.

    Args:
        skill: The skill to compress.
        llm_client: LLM client for all operations.
        enable_gate1: Whether to enable Gate 1.
        enable_gate2: Whether to enable Gate 2.
        failed_criteria: Optional failed criteria for Gate 2.

    Returns:
        PipelineResult with complete compression results.
    """
    pipeline = CompressionPipeline(
        llm_client=llm_client,
        enable_gate1=enable_gate1,
        enable_gate2=enable_gate2
    )
    return pipeline.run_pipeline(skill, failed_criteria=failed_criteria)
