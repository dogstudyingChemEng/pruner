"""
Quality Gates for SkillReducer framework.

Implements faithfulness verification and feedback loop mechanisms
from the SkillReducer paper to ensure compression quality.

Key improvements aligned with paper:
- Gate 1: Per-content-type fine-grained rollback
- Gate 2: Fully automated evaluation loop with task generation
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from .models import Skill, ContentBlock, ContentType, Description, Body, References, OnDemandModules
from .stage1_router import Stage1Optimizer
from .optimizer import Stage2Optimizer, CompressionMetrics
from .chunker import count_tokens


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
    rollback_types: list[str] = field(default_factory=list)  # Content types that need rollback


@dataclass
class FeedbackLoopResult:
    """Result of feedback loop promotion."""

    promoted_blocks: list[str]  # Block IDs that were promoted
    promotion_count: int
    new_core_tokens: int
    feedback_addressed: list[str]


@dataclass
class EvaluationTask:
    """A single evaluation task for Gate 2."""

    task_id: str
    query: str
    expected_outcome: str


@dataclass
class TaskResult:
    """Result of evaluating a single task."""

    task_id: str
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class AutomatedLoopResult:
    """Result of the automated Gate 2 feedback loop."""

    loop_iteration: int
    all_tasks_passed: bool
    task_results: list[TaskResult]
    promoted_blocks: list[str]
    final_blocks: list[ContentBlock]


@dataclass
class PipelineResult:
    """Result of the complete compression pipeline."""

    skill: Skill
    stage1_compressed: bool
    stage2_compressed: bool
    faithfulness_passed: bool
    rollback_performed: bool
    original_tokens: int
    final_tokens: int
    overall_compression_ratio: float
    rollback_types: list[str] = field(default_factory=list)
    compression_metrics: Optional[CompressionMetrics] = None
    errors: list[str] = field(default_factory=list)
    gate2_result: Optional[AutomatedLoopResult] = None


class QualityGates:
    """
    Quality gates for SkillReducer compression pipeline.

    Implements Gate 1 (Faithfulness Verification) and Gate 2 (Feedback Loop)
    from the SkillReducer paper.

    Gate 1 improvements:
    - Per-content-type fine-grained rollback
    - Only rollback the types that have missing concepts

    Gate 2 improvements:
    - Automated task generation (5 tasks)
    - Automated evaluation and promotion loop
    - Max 2 iterations
    """

    def __init__(
        self,
        llm_client,
        faithfulness_threshold: float = 1.0,
        max_loop_iterations: int = 2
    ):
        """
        Initialize quality gates.

        Args:
            llm_client: LLM client for verification.
            faithfulness_threshold: Minimum ratio of preserved concepts (0.0-1.0).
                                  Default 1.0 means all concepts must be preserved.
            max_loop_iterations: Maximum iterations for Gate 2 feedback loop (default 2).
        """
        self.llm_client = llm_client
        self.faithfulness_threshold = faithfulness_threshold
        self.max_loop_iterations = max_loop_iterations

    # ============================================================
    # Gate 1: Faithfulness Verification with Fine-grained Rollback
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
        identifies which content types need rollback.

        Args:
            original_body: The original skill body text.
            compressed_core_rules: The compressed core rules.
            skill_name: Skill name for context.

        Returns:
            FaithfulnessResult with verification details and rollback types.
        """
        if not original_body or not original_body.strip():
            return FaithfulnessResult(
                passed=True,
                gate_result=GateResult.PASSED,
                preserved_concepts=[],
                missing_concepts=[],
                reasoning="Empty original body, verification skipped",
                should_rollback=False,
                rollback_types=[]
            )

        if not compressed_core_rules or not compressed_core_rules.strip():
            return FaithfulnessResult(
                passed=False,
                gate_result=GateResult.FAILED,
                preserved_concepts=[],
                missing_concepts=["No compressed rules provided"],
                reasoning="Compressed rules are empty",
                should_rollback=True,
                rollback_types=["core_rule"]
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

For each missing concept, identify which content TYPE it belongs to:
- core_rule: Actionable instructions, rules, procedures
- example: Code examples, usage demonstrations
- template: Boilerplate, ready-to-use templates
- background: Explanations, contextual knowledge

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
    {
      "concept": "the missing concept",
      "likely_type": "core_rule|example|template|background"
    }
  ],
  "faithfulness_score": 0.95,
  "reasoning": "Explanation of the verification result"
}"""

        user_prompt = f"""Skill Name: {skill_name}

ORIGINAL BODY:
{original_body[:4000]}

COMPRESSED CORE RULES:
{compressed_core_rules}

Identify all core operational concepts in the original and check if they are preserved in the compressed version. For missing concepts, identify their likely content type."""

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
        missing_concepts_raw = result.get("missing_concepts", [])
        faithfulness_score = result.get("faithfulness_score", 1.0)
        reasoning = result.get("reasoning", "")

        # Extract missing concepts and their types
        missing_concepts = []
        rollback_types = set()

        for mc in missing_concepts_raw:
            if isinstance(mc, dict):
                missing_concepts.append(mc.get("concept", str(mc)))
                likely_type = mc.get("likely_type", "core_rule")
                rollback_types.add(likely_type)
            else:
                missing_concepts.append(str(mc))
                rollback_types.add("core_rule")

        # Determine if passed
        passed = len(missing_concepts) == 0 and faithfulness_score >= self.faithfulness_threshold

        should_rollback = len(missing_concepts) > 0

        return FaithfulnessResult(
            passed=passed,
            gate_result=GateResult.PASSED if passed else GateResult.ROLLBACK,
            preserved_concepts=preserved_concepts,
            missing_concepts=missing_concepts,
            reasoning=reasoning,
            should_rollback=should_rollback,
            rollback_types=list(rollback_types)
        )

    def fine_grained_rollback(
        self,
        compressed_blocks: list[ContentBlock],
        original_blocks: list[ContentBlock],
        rollback_types: list[str]
    ) -> list[ContentBlock]:
        """
        Perform fine-grained rollback by content type.

        Uses set concatenation logic: keep compressed blocks NOT in rollback_types,
        add original blocks that ARE in rollback_types.

        Args:
            compressed_blocks: The compressed blocks.
            original_blocks: The original classified blocks before compression.
            rollback_types: List of content types to rollback.

        Returns:
            List of blocks with selective rollback applied.
        """
        if not rollback_types:
            return compressed_blocks

        rollback_type_set = set(rollback_types)

        # Set concatenation: keep non-rollback compressed + add rollback originals
        result_blocks = []

        # Add compressed blocks that are NOT in rollback_types
        for block in compressed_blocks:
            block_type = block.content_type if isinstance(block.content_type, str) else block.content_type.value
            if block_type not in rollback_type_set:
                result_blocks.append(block)

        # Add original blocks that ARE in rollback_types
        for block in original_blocks:
            block_type = block.content_type if isinstance(block.content_type, str) else block.content_type.value
            if block_type in rollback_type_set:
                result_blocks.append(ContentBlock(
                    chunk_id=block.chunk_id,
                    content=block.content,
                    content_type=block.content_type,
                    token_count=block.token_count
                ))

        return result_blocks

    # ============================================================
    # Gate 2: Automated Feedback Loop
    # ============================================================

    def generate_evaluation_tasks(
        self,
        skill: Skill,
        num_tasks: int = 5
    ) -> list[EvaluationTask]:
        """
        Generate evaluation tasks for Gate 2.

        Creates diverse tasks that test the skill's capabilities.

        Args:
            skill: The skill to generate tasks for.
            num_tasks: Number of tasks to generate (default 5).

        Returns:
            List of EvaluationTask objects.
        """
        system_prompt = """You are an expert at creating evaluation tasks for AI skills.

Your task is to generate diverse, realistic tasks that test a skill's capabilities.

Guidelines:
1. Tasks should be specific and testable
2. Cover different aspects of the skill
3. Some tasks should be simple, some complex
4. Include clear expected outcomes
5. Tasks should be realistic user requests

IMPORTANT: Respond with a JSON object:
{
  "tasks": [
    {
      "task_id": "task_1",
      "query": "The user's request",
      "expected_outcome": "What successful completion looks like"
    }
  ]
}"""

        user_prompt = f"""Skill Name: {skill.name}
Skill Description: {skill.description.original}

Skill Body Summary (first 2000 chars):
{skill.body.original[:2000]}

Generate {num_tasks} diverse evaluation tasks for this skill."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        tasks = []
        for t in result.get("tasks", [])[:num_tasks]:
            tasks.append(EvaluationTask(
                task_id=t.get("task_id", f"task_{len(tasks)}"),
                query=t.get("query", ""),
                expected_outcome=t.get("expected_outcome", "")
            ))

        return tasks

    def evaluate_task(
        self,
        task: EvaluationTask,
        skill: Skill,
        core_rules: str
    ) -> TaskResult:
        """
        Evaluate a single task against the compressed skill.

        Simulates whether the skill with compressed core rules can
        successfully complete the task.

        Args:
            task: The evaluation task.
            skill: The skill being evaluated.
            core_rules: The compressed core rules.

        Returns:
            TaskResult with pass/fail and failure reasons.
        """
        system_prompt = """You are an expert at evaluating AI skill performance.

Your task is to determine if a skill's core rules contain enough information
to successfully complete a given task.

Guidelines:
1. Check if the core rules provide necessary instructions for the task
2. Identify any missing information or capabilities
3. Be strict - the skill should be able to complete the task without additional context

IMPORTANT: Respond with a JSON object:
{
  "passed": true/false,
  "failure_reasons": [
    "Reason 1 why the skill cannot complete this task",
    "Reason 2..."
  ],
  "reasoning": "Explanation of the evaluation"
}"""

        user_prompt = f"""Skill: {skill.name}

Task Query: {task.query}

Expected Outcome: {task.expected_outcome}

Core Rules Available:
{core_rules}

Can this skill successfully complete the task with only these core rules?"""

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

        return TaskResult(
            task_id=task.task_id,
            passed=result.get("passed", False),
            failure_reasons=result.get("failure_reasons", [])
        )

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

    def run_automated_feedback_loop(
        self,
        skill: Skill,
        initial_blocks: list[ContentBlock],
        stage2_optimizer: Stage2Optimizer,
        max_iterations: int = 2
    ) -> AutomatedLoopResult:
        """
        Run the fully automated Gate 2 feedback loop.

        Steps:
        1. Generate 5 evaluation tasks
        2. Evaluate all tasks
        3. If any fail, promote relevant blocks
        4. Re-compress core rules
        5. Re-evaluate
        6. Repeat up to max_iterations times

        Args:
            skill: The skill being evaluated.
            initial_blocks: The initial compressed blocks.
            stage2_optimizer: Stage2 optimizer for re-compression.
            max_iterations: Maximum loop iterations (default 2).

        Returns:
            AutomatedLoopResult with final blocks and status.
        """
        current_blocks = list(initial_blocks)
        all_promoted = []

        for iteration in range(1, max_iterations + 1):
            # Generate evaluation tasks
            tasks = self.generate_evaluation_tasks(skill, num_tasks=5)

            # Get current core rules
            core_rules = "\n".join([
                b.content for b in current_blocks
                if _is_content_type(b, ContentType.CORE_RULE)
            ])

            # Evaluate all tasks
            task_results = []
            all_failure_reasons = []

            for task in tasks:
                result = self.evaluate_task(task, skill, core_rules)
                task_results.append(result)
                if not result.passed:
                    all_failure_reasons.extend(result.failure_reasons)

            # Check if all passed
            all_passed = all(r.passed for r in task_results)

            if all_passed:
                return AutomatedLoopResult(
                    loop_iteration=iteration,
                    all_tasks_passed=True,
                    task_results=task_results,
                    promoted_blocks=all_promoted,
                    final_blocks=current_blocks
                )

            # Not all passed - do promotion
            if all_failure_reasons:
                feedback_result = self.feedback_loop(
                    current_blocks,
                    all_failure_reasons,
                    skill_context=f"{skill.name}: {skill.description.original}"
                )

                if feedback_result.promotion_count > 0:
                    all_promoted.extend(feedback_result.promoted_blocks)

                    # Re-compress core rules with newly promoted blocks
                    core_blocks = [b for b in current_blocks if _is_content_type(b, ContentType.CORE_RULE)]
                    if core_blocks:
                        compressed_core = stage2_optimizer.compress_core_rules(core_blocks)

                        # Update blocks: remove old core, add new compressed
                        non_core = [b for b in current_blocks if not _is_content_type(b, ContentType.CORE_RULE)]
                        current_blocks = compressed_core + non_core

            # If no promotions happened, we can't improve further
            if not all_failure_reasons or (feedback_result and feedback_result.promotion_count == 0):
                break

        # Final evaluation
        tasks = self.generate_evaluation_tasks(skill, num_tasks=5)
        core_rules = "\n".join([
            b.content for b in current_blocks
            if _is_content_type(b, ContentType.CORE_RULE)
        ])
        task_results = [self.evaluate_task(t, skill, core_rules) for t in tasks]
        all_passed = all(r.passed for r in task_results)

        return AutomatedLoopResult(
            loop_iteration=max_iterations,
            all_tasks_passed=all_passed,
            task_results=task_results,
            promoted_blocks=all_promoted,
            final_blocks=current_blocks
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
        3. Gate 1: Verify faithfulness (with per-type rollback)
        4. Gate 2: Automated feedback loop (if enabled)
        5. Finalize results

        Args:
            skill: The skill to compress.
            compress_core: Whether to compress core rules.
            dedup_examples: Whether to deduplicate examples.
            dedup_templates: Whether to deduplicate templates.
            summarize_background: Whether to summarize background.
            dedup_references: Whether to deduplicate references.
            failed_criteria: Optional list of failed task criteria for Gate 2 (deprecated - auto-generated now).

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
        rollback_types = []
        compression_metrics = None
        final_blocks = []
        updated_references = skill.references
        gate2_result = None
        original_blocks = []  # Store original classification for fine-grained rollback

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
            on_demand_modules = None
            try:
                # First, classify without compression to store originals
                original_blocks = self.stage2_optimizer.classify_skill_body(skill)

                # New return structure: (core_blocks, on_demand_modules, references, metrics)
                core_blocks, on_demand_modules, updated_references, compression_metrics = self.stage2_optimizer.optimize_skill(
                    skill,
                    compress_core=compress_core,
                    dedup_examples=dedup_examples,
                    dedup_templates=dedup_templates,
                    summarize_background=summarize_background,
                    dedup_references=dedup_references
                )
                final_blocks = core_blocks  # Only core blocks go into body
                stage2_compressed = True
            except Exception as e:
                errors.append(f"Stage 2 failed: {str(e)}")
                # Use original blocks as fallback
                final_blocks = self.stage2_optimizer.classify_skill_body(skill)
                on_demand_modules = OnDemandModules()
                updated_references = skill.references

            # ============================================================
            # Gate 1: Faithfulness Verification (Fine-grained Rollback)
            # ============================================================
            if self.enable_gate1 and stage2_compressed:
                faithfulness_result = self.quality_gates.run_faithfulness_gate(
                    skill, final_blocks
                )

                if faithfulness_result.should_rollback:
                    faithfulness_passed = False
                    rollback_performed = True
                    rollback_types = faithfulness_result.rollback_types

                    # Fine-grained rollback by content type
                    rollback_type_set = set(rollback_types)

                    # 1. Rollback core_rule: replace final_blocks with original's core_rule
                    if "core_rule" in rollback_type_set:
                        original_core_blocks = [
                            b for b in original_blocks
                            if (b.content_type if isinstance(b.content_type, str) else b.content_type.value) == "core_rule"
                        ]
                        # Keep non-core_rule compressed blocks, add original core_rule blocks
                        non_core_compressed = [
                            b for b in final_blocks
                            if (b.content_type if isinstance(b.content_type, str) else b.content_type.value) != "core_rule"
                        ]
                        final_blocks = non_core_compressed + [
                            ContentBlock(
                                chunk_id=b.chunk_id,
                                content=b.content,
                                content_type=b.content_type,
                                token_count=b.token_count
                            ) for b in original_core_blocks
                        ]

                    # 2. Rollback on-demand modules: examples, templates, backgrounds
                    # Each type is rolled back independently
                    if on_demand_modules is None:
                        on_demand_modules = OnDemandModules()

                    # Rollback examples
                    if "example" in rollback_type_set:
                        original_examples = [
                            b for b in original_blocks
                            if (b.content_type if isinstance(b.content_type, str) else b.content_type.value) == "example"
                        ]
                        on_demand_modules.examples = [
                            ContentBlock(
                                chunk_id=b.chunk_id,
                                content=b.content,
                                content_type=b.content_type,
                                token_count=b.token_count
                            ) for b in original_examples
                        ]

                    # Rollback templates
                    if "template" in rollback_type_set:
                        original_templates = [
                            b for b in original_blocks
                            if (b.content_type if isinstance(b.content_type, str) else b.content_type.value) == "template"
                        ]
                        on_demand_modules.templates = [
                            ContentBlock(
                                chunk_id=b.chunk_id,
                                content=b.content,
                                content_type=b.content_type,
                                token_count=b.token_count
                            ) for b in original_templates
                        ]

                    # Rollback backgrounds
                    if "background" in rollback_type_set:
                        original_backgrounds = [
                            b for b in original_blocks
                            if (b.content_type if isinstance(b.content_type, str) else b.content_type.value) == "background"
                        ]
                        on_demand_modules.backgrounds = [
                            ContentBlock(
                                chunk_id=b.chunk_id,
                                content=b.content,
                                content_type=b.content_type,
                                token_count=b.token_count
                            ) for b in original_backgrounds
                        ]

                    # 3. Regenerate routing metadata and update references for rolled-back modules
                    types_need_routing_update = [t for t in rollback_types if t in ("example", "template", "background")]
                    if types_need_routing_update and on_demand_modules:
                        routing_metadata = self.stage2_optimizer.generate_all_routing_metadata(
                            on_demand_modules=on_demand_modules,
                            skill_name=skill.name,
                            skill_description=skill.description.original
                        )
                        # Update reference files with rolled-back on-demand modules
                        on_demand_files = on_demand_modules.to_reference_files(routing_metadata)
                        # Remove old on-demand files from references, add new ones
                        existing_non_on_demand = {
                            k: v for k, v in updated_references.files.items()
                            if not k.startswith("on-demand-")
                        }
                        all_reference_files = {**existing_non_on_demand, **on_demand_files}
                        updated_references = References(
                            files=all_reference_files,
                            total_token_count=sum(
                                count_tokens(c) for c in all_reference_files.values()
                            )
                        )

                    errors.append(
                        f"Gate 1 failed - missing concepts: {faithfulness_result.missing_concepts}, rolled back types: {rollback_types}"
                    )

            # ============================================================
            # Gate 2: Automated Feedback Loop
            # ============================================================
            if self.enable_gate2 and final_blocks:
                gate2_result = self.quality_gates.run_automated_feedback_loop(
                    skill=skill,
                    initial_blocks=final_blocks,
                    stage2_optimizer=self.stage2_optimizer,
                    max_iterations=2
                )

                if gate2_result.promoted_blocks:
                    final_blocks = gate2_result.final_blocks
                    errors.append(
                        f"Gate 2 promoted {len(gate2_result.promoted_blocks)} blocks across {gate2_result.loop_iteration} iteration(s)"
                    )

                if not gate2_result.all_tasks_passed:
                    errors.append(
                        f"Gate 2: {sum(1 for r in gate2_result.task_results if not r.passed)}/{len(gate2_result.task_results)} tasks failed after {gate2_result.loop_iteration} iteration(s)"
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
            rollback_types=rollback_types,
            compression_metrics=compression_metrics,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            overall_compression_ratio=round(overall_ratio, 4),
            errors=errors,
            gate2_result=gate2_result
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
        failed_criteria: Optional failed criteria for Gate 2 (deprecated - auto-generated now).

    Returns:
        PipelineResult with complete compression results.
    """
    pipeline = CompressionPipeline(
        llm_client=llm_client,
        enable_gate1=enable_gate1,
        enable_gate2=enable_gate2
    )
    return pipeline.run_pipeline(skill, failed_criteria=failed_criteria)
