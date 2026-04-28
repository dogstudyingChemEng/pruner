"""
Quality Gates for SkillReducer framework.

Implements faithfulness verification and feedback loop mechanisms
from the SkillReducer paper to ensure compression quality.

Key improvements aligned with paper:
- Gate 1: Per-content-type fine-grained rollback
- Gate 2: Fully automated evaluation loop with task generation
- Gate 2: Three-condition evaluation (D/A/C) with retention calculation
- Gate 2: read_file tool simulation for progressive disclosure
- Gate 2: Hybrid scoring (pytest + LLM judge)
"""

import json
import uuid
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from .models import (
    Skill, ContentBlock, ContentType, Description, Body, References,
    OnDemandModules, RoutingMetadata,
    EvaluationCondition, ConditionScore, RetentionResult
)
from .stage1_router import Stage1Optimizer
from .optimizer import Stage2Optimizer, CompressionMetrics
from .chunker import count_tokens
from .hybrid_evaluator import HybridEvaluator, PytestExecutor, LLMJudge


def _is_content_type(block: ContentBlock, content_type: ContentType) -> bool:
    """
    Check if a block's content_type matches the given type.

    Handles both enum and string values due to Pydantic's use_enum_values.
    """
    if isinstance(block.content_type, str):
        return block.content_type == content_type.value
    return block.content_type == content_type


# ============================================================
# read_file Tool Simulation
# ============================================================


class ToolCallDecision(BaseModel):
    """Decision to call read_file tool for on-demand module."""

    module_name: str = Field(description="Module filename to load")
    reason: str = Field(description="Why this module is needed")
    relevance_score: float = Field(
        default=0.0,
        description="Relevance match score (0.0-1.0)"
    )


class ReadFileToolCall(BaseModel):
    """A simulated read_file tool call execution."""

    call_id: str = Field(description="Unique call identifier")
    module_name: str = Field(description="Module filename loaded")
    module_content: str = Field(description="Module content returned")
    relevance_match: float = Field(default=0.0)


class ReadFileToolSimulator:
    """
    Simulates read_file tool for on-demand module loading.

    Per SkillReducer paper, Condition C evaluation uses read_file tool:
    - Agent decides which references to load based on when/topics metadata
    - Maximum 6 tool calls per task
    - Returns module content when called

    This simulator matches query semantics to module routing metadata
    to simulate the agent's decision process.
    """

    def __init__(
        self,
        on_demand_modules: OnDemandModules,
        routing_metadata: dict[str, RoutingMetadata],
        max_calls: int = 6
    ):
        """
        Initialize read_file tool simulator.

        Args:
            on_demand_modules: On-demand modules available for loading.
            routing_metadata: Routing metadata for each module (when/topics).
            max_calls: Maximum tool calls allowed per task.
        """
        self.modules = on_demand_modules
        self.routing = routing_metadata
        self.max_calls = max_calls
        self.call_history: list[ReadFileToolCall] = []

    def suggest_tool_calls(
        self,
        llm_client,
        query: str,
        current_context: str
    ) -> list[ToolCallDecision]:
        """
        LLM suggests which modules to load based on query and context.

        Args:
            llm_client: LLM client for decision making.
            query: Task query to match against modules.
            current_context: Current context (core rules).

        Returns:
            List of ToolCallDecision for modules to load.
        """
        # Build module descriptions for LLM
        module_descriptions = self._build_module_descriptions()

        system_prompt = """You are an AI agent deciding which reference modules to load.

Given a task query and current context (core rules), decide which on-demand
modules would be helpful to complete the task.

For each module, consider:
1. Does the query topic match the module's topics?
2. Does the task type match the module's WHEN condition?
3. Is the module likely to provide needed information not in core rules?

Respond with a JSON object:
{
  "decisions": [
    {
      "module_name": "on-demand-examples.md",
      "reason": "Query asks for code example",
      "relevance_score": 0.8
    }
  ]
}

Only suggest modules that are actually needed. Maximum 6 modules."""

        user_prompt = f"""Task Query: {query}

Current Context (Core Rules):
{current_context[:1500]}

Available Modules:
{module_descriptions}

Decide which modules to load (maximum {self.max_calls})."""

        try:
            response = llm_client.client.chat.completions.create(
                model=llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            decisions = []

            for d in result.get("decisions", [])[:self.max_calls]:
                decisions.append(ToolCallDecision(
                    module_name=d.get("module_name", ""),
                    reason=d.get("reason", ""),
                    relevance_score=d.get("relevance_score", 0.0)
                ))

            return decisions

        except Exception:
            # Fallback: keyword matching
            return self._keyword_match_tool_calls(query)

    def _build_module_descriptions(self) -> str:
        """Build descriptions of available modules for LLM prompt."""
        lines = []

        if self.modules.examples:
            meta = self.routing.get("on-demand-examples.md", RoutingMetadata())
            content_preview = self.modules.examples[0].content[:200] if self.modules.examples else ""
            lines.append(f"- on-demand-examples.md")
            lines.append(f"  WHEN: {meta.when}")
            lines.append(f"  TOPICS: {', '.join(meta.topics)}")
            lines.append(f"  Preview: {content_preview}")

        if self.modules.templates:
            meta = self.routing.get("on-demand-templates.md", RoutingMetadata())
            content_preview = self.modules.templates[0].content[:200] if self.modules.templates else ""
            lines.append(f"- on-demand-templates.md")
            lines.append(f"  WHEN: {meta.when}")
            lines.append(f"  TOPICS: {', '.join(meta.topics)}")
            lines.append(f"  Preview: {content_preview}")

        if self.modules.background:
            meta = self.routing.get("on-demand-background.md", RoutingMetadata())
            content_preview = self.modules.background[0].content[:200] if self.modules.background else ""
            lines.append(f"- on-demand-background.md")
            lines.append(f"  WHEN: {meta.when}")
            lines.append(f"  TOPICS: {', '.join(meta.topics)}")
            lines.append(f"  Preview: {content_preview}")

        return "\n".join(lines)

    def _keyword_match_tool_calls(self, query: str) -> list[ToolCallDecision]:
        """Fallback keyword matching for tool call decisions."""
        decisions = []
        query_lower = query.lower()

        # Check each module's topics against query
        for module_name, meta in self.routing.items():
            for topic in meta.topics:
                if topic.lower() in query_lower:
                    decisions.append(ToolCallDecision(
                        module_name=module_name,
                        reason=f"Query mentions topic '{topic}'",
                        relevance_score=0.6
                    ))
                    break

        return decisions[:self.max_calls]

    def execute_tool_calls(
        self,
        decisions: list[ToolCallDecision]
    ) -> list[ReadFileToolCall]:
        """
        Execute simulated read_file calls for selected modules.

        Args:
            decisions: List of ToolCallDecision to execute.

        Returns:
            List of ReadFileToolCall with module contents.
        """
        calls = []
        self.call_history = []

        for decision in decisions:
            module_content = self._get_module_content(decision.module_name)
            if module_content:
                call = ReadFileToolCall(
                    call_id=str(uuid.uuid4())[:8],
                    module_name=decision.module_name,
                    module_content=module_content,
                    relevance_match=decision.relevance_score
                )
                calls.append(call)
                self.call_history.append(call)

        return calls

    def _get_module_content(self, module_name: str) -> str:
        """Get content for a module by name."""
        if module_name == "on-demand-examples.md" and self.modules.examples:
            return "\n\n---\n\n".join([b.content for b in self.modules.examples])

        if module_name == "on-demand-templates.md" and self.modules.templates:
            return "\n\n---\n\n".join([b.content for b in self.modules.templates])

        if module_name == "on-demand-background.md" and self.modules.background:
            return "\n\n---\n\n".join([b.content for b in self.modules.background])

        return ""

    def build_augmented_context(
        self,
        base_context: str,
        tool_calls: list[ReadFileToolCall]
    ) -> str:
        """
        Build augmented context with loaded module contents.

        Args:
            base_context: Core rules context.
            tool_calls: Executed tool calls with module contents.

        Returns:
            Combined context string.
        """
        parts = [base_context]

        for call in tool_calls:
            parts.append(f"\n\n--- Loaded Reference: {call.module_name} ---\n\n")
            parts.append(call.module_content)

        return "\n".join(parts)


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
    """A single evaluation task for Gate 2.

    Per SkillReducer paper Section IV-B:
    - Core-only tasks: Answerable from core rules alone
    - Needs-reference tasks: Require at least one reference module
    - Mix of code execution tasks (52.3%) and rubric tasks (47.7%)
    """

    task_id: str
    query: str
    expected_outcome: str
    required_references: list[str] = field(default_factory=list)  # Specific modules needed
    is_code_task: bool = False  # True for pytest-verifiable tasks
    needs_reference: bool = False  # True if task requires reference modules


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
    - Three-condition evaluation (D/A/C) with retention calculation
    - read_file tool simulation for progressive disclosure
    - Hybrid scoring (pytest + LLM judge)
    """

    def __init__(
        self,
        llm_client,
        faithfulness_threshold: float = 1.0,
        max_loop_iterations: int = 2,
        use_three_conditions: bool = False,
        use_hybrid_scoring: bool = False,
        pytest_weight: float = 0.523,
        llm_judge_weight: float = 0.477,
        enable_read_file_tool: bool = True,
        max_tool_calls: int = 6,
        hybrid_evaluator: Optional[HybridEvaluator] = None
    ):
        """
        Initialize quality gates.

        Args:
            llm_client: LLM client for verification.
            faithfulness_threshold: Minimum ratio of preserved concepts (0.0-1.0).
                                  Default 1.0 means all concepts must be preserved.
            max_loop_iterations: Maximum iterations for Gate 2 feedback loop (default 2).
            use_three_conditions: Enable three-condition (D/A/C) evaluation.
            use_hybrid_scoring: Enable hybrid scoring (pytest + LLM judge).
            pytest_weight: Weight for pytest score (default 0.523 per paper).
            llm_judge_weight: Weight for LLM judge score (default 0.477 per paper).
            enable_read_file_tool: Enable read_file tool simulation for Condition C.
            max_tool_calls: Maximum read_file calls per task (default 6 per paper).
            hybrid_evaluator: Optional HybridEvaluator instance (created if not provided).
        """
        self.llm_client = llm_client
        self.faithfulness_threshold = faithfulness_threshold
        self.max_loop_iterations = max_loop_iterations
        self.use_three_conditions = use_three_conditions
        self.use_hybrid_scoring = use_hybrid_scoring
        self.pytest_weight = pytest_weight
        self.llm_judge_weight = llm_judge_weight
        self.enable_read_file_tool = enable_read_file_tool
        self.max_tool_calls = max_tool_calls

        # Initialize hybrid evaluator for three-condition evaluation
        if hybrid_evaluator:
            self.hybrid_evaluator = hybrid_evaluator
        elif use_hybrid_scoring:
            # Create hybrid evaluator with pytest executor and LLM judge
            pytest_executor = PytestExecutor()
            llm_judge = LLMJudge(llm_client)
            self.hybrid_evaluator = HybridEvaluator(
                pytest_executor=pytest_executor,
                llm_judge=llm_judge
            )
        else:
            self.hybrid_evaluator = None

    # ============================================================
    # Gate 1: Faithfulness Verification with Fine-grained Rollback
    # ============================================================

    def verify_faithfulness(
        self,
        original_body: str,
        compressed_core_rules: str,
        skill_name: str = "unknown",
        reference_modules: Optional[dict[str, str]] = None
    ) -> FaithfulnessResult:
        """
        Gate 1: Verify that all core operational concepts are preserved.

        Per paper Eq. 3: ∀τ : C_τ(s.b) ⊆ C_τ(b*) ∪ ∪_{r∈R*} C_τ(r)

        Checks if all 'core operational concepts' from the original body
        are preserved EITHER in the compressed core rules OR in reference
        modules (examples, templates, background). Concepts that appear in
        reference modules are NOT missing — they were correctly moved by
        progressive disclosure.

        If concepts are missing from both, identifies which content types
        need rollback.

        Args:
            original_body: The original skill body text.
            compressed_core_rules: The compressed core rules.
            skill_name: Skill name for context.
            reference_modules: Optional dict of reference module filenames
                to their content (e.g., {"on-demand-examples.md": "..."}).
                Per paper Eq. 3, concepts in these modules count as preserved.

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
are preserved somewhere in the output. Per the SkillReducer paper (Eq. 3), concepts
can be preserved in EITHER the compressed core rules OR the reference modules.

Reference modules (examples.md, templates.md, background.md) contain content that
was intentionally moved out of the core by progressive disclosure. If a concept from
the original body appears in a reference module, it is PRESERVED (not missing).

Core operational concepts include:
- Specific actions or steps that must be performed
- Rules and constraints that must be followed
- Required procedures or workflows
- Key parameters, thresholds, or limits
- Critical conditions or triggers

Do NOT flag as missing:
- Concepts that appear in the reference modules (they were moved, not lost)
- Background information or explanations (correctly in background.md)
- Examples or demonstrations (correctly in examples.md)
- Templates or boilerplate (correctly in templates.md)
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
    "concept that was preserved (in core OR in a reference module)"
  ],
  "missing_concepts": [
    {
      "concept": "the missing concept",
      "likely_type": "core_rule|example|template|background"
    }
  ],
  "faithfulness_score": 0.95,
  "reasoning": "Explanation of the verification result, noting which concepts were found in reference modules"
}"""

        # Build reference modules section for the prompt
        reference_section = ""
        if reference_modules:
            ref_parts = []
            for ref_name, ref_content in reference_modules.items():
                if ref_content and ref_content.strip():
                    ref_parts.append(f"--- {ref_name} ---\n{ref_content[:2000]}")
            if ref_parts:
                reference_section = "\n\nREFERENCE MODULES (content moved here by progressive disclosure — concepts found here are PRESERVED, not missing):\n\n" + "\n\n".join(ref_parts)

        user_prompt = f"""Skill Name: {skill_name}

ORIGINAL BODY:
{original_body[:8000]}

COMPRESSED CORE RULES:
{compressed_core_rules}{reference_section}

Identify all core operational concepts in the original and check if they are preserved in the compressed core rules OR in the reference modules. A concept found in a reference module is PRESERVED — do NOT flag it as missing."""

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
        num_tasks: int = 5,
        ensure_task_mix: bool = True
    ) -> list[EvaluationTask]:
        """
        Generate evaluation tasks for Gate 2.

        Per SkillReducer paper Section IV-B:
        - 5 diverse tasks per skill
        - Mix of "core-only" (answerable from b* alone) and "needs-reference" tasks
        - Mix of code execution tasks (52.3%) and rubric tasks (47.7%)

        Args:
            skill: The skill to generate tasks for.
            num_tasks: Number of tasks to generate (default 5).
            ensure_task_mix: Ensure proper mix of task types per paper.

        Returns:
            List of EvaluationTask objects with type annotations.
        """
        system_prompt = """You are an expert at creating evaluation tasks for AI skills.

Your task is to generate diverse, realistic tasks that test a skill's capabilities.

Per SkillReducer methodology, generate tasks with TWO distinctions:

1. TASK CONTENT TYPE:
   - Core-only: Can be completed using only core rules (no references needed)
   - Needs-reference: Requires loading at least one reference module (examples/templates/background)

2. TASK VERIFICATION TYPE:
   - Code task: Output can be verified by pytest (executable code, assertions)
   - Rubric task: Output must be judged by LLM against rubric criteria

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
      "expected_outcome": "What successful completion looks like",
      "is_code_task": true/false,
      "needs_reference": true/false,
      "required_references": ["list of specific reference modules needed, if any"]
    }
  ]
}

Generate a mix: ~50% core-only, ~50% needs-reference. ~52% code tasks, ~48% rubric tasks."""

        user_prompt = f"""Skill Name: {skill.name}
Skill Description: {skill.description.original}

Skill Body Summary (first 2000 chars):
{skill.body.original[:2000]}

Reference Files: {list(skill.references.files.keys()) if skill.references.files else 'none'}

Generate {num_tasks} diverse evaluation tasks for this skill. Include proper task type annotations."""

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
                expected_outcome=t.get("expected_outcome", ""),
                is_code_task=t.get("is_code_task", False),
                needs_reference=t.get("needs_reference", False),
                required_references=t.get("required_references", [])
            ))

        # If ensure_task_mix, validate and adjust
        if ensure_task_mix and tasks:
            # Ensure at least 1 core-only, 1 needs-reference
            core_only_count = sum(1 for t in tasks if not t.needs_reference)
            needs_ref_count = sum(1 for t in tasks if t.needs_reference)

            if core_only_count == 0:
                # Mark first task as core-only
                tasks[0].needs_reference = False
                tasks[0].required_references = []
            if needs_ref_count == 0:
                # Mark last task as needs-reference
                tasks[-1].needs_reference = True
                tasks[-1].required_references = ["on-demand-examples.md"]

            # Ensure ~52% code tasks
            code_count = sum(1 for t in tasks if t.is_code_task)
            target_code = round(num_tasks * 0.523)
            if code_count < target_code:
                # Convert some rubric tasks to code tasks
                for t in tasks:
                    if not t.is_code_task and code_count < target_code:
                        t.is_code_task = True
                        code_count += 1

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
        skill_context: Optional[str] = None,
        original_blocks: Optional[list[ContentBlock]] = None
    ) -> FeedbackLoopResult:
        """
        Gate 2: Promote relevant blocks to core_rule based on task failure feedback.

        When task evaluation fails, this method identifies example/background
        blocks related to the failure criteria and promotes them to core_rule
        type to ensure they are always loaded.

        Per paper: promoted items are appended to the core in their original
        form without further compression.

        Args:
            blocks: List of classified content blocks (possibly compressed).
            failed_criteria: List of criteria that caused task failure.
            skill_context: Optional context about the skill.
            original_blocks: Original uncompressed blocks for restoring
                promoted content to its original form.

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
        # Build lookup of original blocks by chunk_id for restoring uncompressed content
        original_by_id = {}
        if original_blocks:
            original_by_id = {b.chunk_id: b for b in original_blocks}

        for rel_block in relevant_blocks:
            block_id = rel_block.get("block_id", "")
            for block in blocks:
                is_promotable = (
                    block.content_type in promotable_types or
                    (isinstance(block.content_type, str) and block.content_type in promotable_type_values)
                )
                if block.chunk_id == block_id and is_promotable:
                    # Promote to core_rule, restoring original content if available
                    block.content_type = ContentType.CORE_RULE
                    if block_id in original_by_id:
                        orig = original_by_id[block_id]
                        block.content = orig.content
                        block.token_count = orig.token_count
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
        max_iterations: int = 2,
        original_blocks: Optional[list[ContentBlock]] = None
    ) -> AutomatedLoopResult:
        """
        Run the fully automated Gate 2 feedback loop.

        Steps:
        1. Generate 5 evaluation tasks
        2. Evaluate all tasks
        3. If any fail, promote relevant blocks (in original form, per paper)
        4. Re-compress only existing core rules (not newly promoted blocks)
        5. Re-evaluate
        6. Repeat up to max_iterations times

        Per paper: promoted items are appended to the core in their original
        form without further compression. Only non-promoted core items retain
        their compressed form.

        Args:
            skill: The skill being evaluated.
            initial_blocks: The initial compressed blocks.
            stage2_optimizer: Stage2 optimizer for re-compression.
            max_iterations: Maximum loop iterations (default 2).
            original_blocks: Original uncompressed blocks for restoring
                promoted content to its original form.

        Returns:
            AutomatedLoopResult with final blocks and status.
        """
        current_blocks = list(initial_blocks)
        all_promoted = []
        newly_promoted_ids: set = set()  # Track blocks promoted in this loop

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
                prev_promoted_count = len(all_promoted)
                feedback_result = self.feedback_loop(
                    current_blocks,
                    all_failure_reasons,
                    skill_context=f"{skill.name}: {skill.description.original}",
                    original_blocks=original_blocks
                )

                if feedback_result.promotion_count > 0:
                    # Track newly promoted block IDs from this iteration
                    new_promotions = feedback_result.promoted_blocks[prev_promoted_count:] if len(all_promoted) < len(feedback_result.promoted_blocks) else feedback_result.promoted_blocks
                    newly_promoted_ids.update(feedback_result.promoted_blocks)
                    all_promoted = feedback_result.promoted_blocks

                    # Separate existing core blocks (re-compress) from
                    # newly promoted blocks (keep original, per paper)
                    existing_core = [
                        b for b in current_blocks
                        if _is_content_type(b, ContentType.CORE_RULE)
                        and b.chunk_id not in newly_promoted_ids
                    ]
                    promoted_core = [
                        b for b in current_blocks
                        if _is_content_type(b, ContentType.CORE_RULE)
                        and b.chunk_id in newly_promoted_ids
                    ]

                    if existing_core:
                        compressed_existing = stage2_optimizer.compress_core_rules(existing_core)
                    else:
                        compressed_existing = []

                    # Keep promoted blocks in original form (no re-compression)
                    # Combine: compressed existing core + unmodified promoted blocks + non-core
                    non_core = [
                        b for b in current_blocks
                        if not _is_content_type(b, ContentType.CORE_RULE)
                    ]
                    current_blocks = compressed_existing + promoted_core + non_core

            # If no promotions happened, we can't improve further
            if not all_failure_reasons or feedback_result.promotion_count == 0:
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
        compressed_blocks: list[ContentBlock],
        reference_modules: Optional[dict[str, str]] = None
    ) -> FaithfulnessResult:
        """
        Run Gate 1 on a skill with compressed blocks.

        Per paper Eq. 3, reference modules are included in the verification:
        concepts moved to on-demand modules count as preserved.

        Args:
            skill: The original skill.
            compressed_blocks: The compressed content blocks.
            reference_modules: Optional dict of reference file contents
                (e.g., {"on-demand-examples.md": "...", ...}).

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
            skill_name=skill.name,
            reference_modules=reference_modules
        )

    # ============================================================
    # Three-Condition Evaluation (D/A/C)
    # ============================================================

    def evaluate_condition_D(
        self,
        task: EvaluationTask,
        skill_name: str,
        is_code_task: bool = False
    ) -> ConditionScore:
        """
        Evaluate task without any skill (Condition D - baseline).

        Agent receives only the query, no skill context.
        Tests if agent can complete task from general knowledge.

        Args:
            task: Evaluation task with query and expected outcome.
            skill_name: Skill name for context.
            is_code_task: Whether this is a code execution task.

        Returns:
            ConditionScore for condition D.
        """
        # Use hybrid scoring if enabled (mostly for consistency)
        if self.use_hybrid_scoring and self.hybrid_evaluator and is_code_task:
            # For baseline D, pytest would fail without skill context
            pytest_score = 0.0

            llm_judge_score = self._llm_judge_eval_baseline(task)

            weighted_score = self.hybrid_evaluator.calculate_weighted_score(
                pytest_score=pytest_score,
                judge_score=llm_judge_score,
                is_code_task=True
            )

            return ConditionScore(
                condition=EvaluationCondition.D,
                pytest_score=pytest_score,
                llm_judge_score=llm_judge_score,
                weighted_score=weighted_score,
                task_id=task.task_id,
                details={"baseline": True}
            )

        # Fallback: simple LLM evaluation
        system_prompt = """You are an AI assistant completing a task without any specialized skill context.

You have general knowledge but no specific skill instructions for this task.
Complete the task based on your general capabilities.

IMPORTANT: Respond with a JSON object:
{
  "output": "Your response to complete the task",
  "confidence": 0.0-1.0,
  "reasoning": "How you approached this task"
}"""

        user_prompt = f"""Task Query: {task.query}
Expected Outcome: {task.expected_outcome}

Complete this task using only your general knowledge (no skill context provided)."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        score = result.get("confidence", 0.0)

        return ConditionScore(
            condition=EvaluationCondition.D,
            pytest_score=0.0,
            llm_judge_score=score,
            weighted_score=score,
            task_id=task.task_id,
            details={"output": result.get("output", ""), "reasoning": result.get("reasoning", "")}
        )

    def _llm_judge_eval_baseline(self, task: EvaluationTask) -> float:
        """Evaluate baseline task without skill context."""
        system_prompt = """You are an LLM judge evaluating task completion without skill context.

Rate how well the task could be completed using only general knowledge.

IMPORTANT: Respond with a JSON object:
{
  "score": 0.0-1.0,
  "reasoning": "Explanation"
}"""

        user_prompt = f"""Task Query: {task.query}
Expected Outcome: {task.expected_outcome}

Rate how well this task can be completed with no skill context."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("score", 0.0)

    def evaluate_condition_A(
        self,
        task: EvaluationTask,
        skill: Skill,
        include_references: bool = True,
        is_code_task: bool = False
    ) -> ConditionScore:
        """
        Evaluate task with original uncompressed skill (Condition A - baseline).

        Agent receives full original skill body + references.
        Establishes upper bound for expected performance.

        Args:
            task: Evaluation task with query and expected outcome.
            skill: Original skill with full body and references.
            include_references: Whether to include reference files.
            is_code_task: Whether this is a code execution task.

        Returns:
            ConditionScore for condition A.
        """
        # Build full context from original skill
        context_parts = [skill.body.original]

        if include_references and skill.references.files:
            for filename, content in skill.references.files.items():
                context_parts.append(f"\n\n--- Reference: {filename} ---\n\n{content}")

        full_context = "\n".join(context_parts)[:4000]  # Limit for prompt

        # Use hybrid scoring if enabled
        if self.use_hybrid_scoring and self.hybrid_evaluator:
            if is_code_task:
                # Code task: pytest + LLM judge
                pytest_result = self.hybrid_evaluator.pytest.run_pytest(
                    self.hybrid_evaluator.pytest.generate_test_file(
                        task_id=task.task_id,
                        expected_outcome=task.expected_outcome,
                        skill_context=full_context[:2000]
                    )
                )
                pytest_score = pytest_result.score if pytest_result.passed else 0.0

                llm_judge_score = self._llm_judge_eval(task, full_context, skill.name)

                weighted_score = self.hybrid_evaluator.calculate_weighted_score(
                    pytest_score=pytest_score,
                    judge_score=llm_judge_score,
                    is_code_task=True
                )

                return ConditionScore(
                    condition=EvaluationCondition.A,
                    pytest_score=pytest_score,
                    llm_judge_score=llm_judge_score,
                    weighted_score=weighted_score,
                    task_id=task.task_id,
                    loaded_references=list(skill.references.files.keys()) if include_references else [],
                    details={"pytest_passed": pytest_result.passed}
                )
            else:
                # Rubric task: LLM judge only
                llm_judge_score = self._llm_judge_eval(task, full_context, skill.name)

                return ConditionScore(
                    condition=EvaluationCondition.A,
                    pytest_score=0.0,
                    llm_judge_score=llm_judge_score,
                    weighted_score=llm_judge_score,
                    task_id=task.task_id,
                    loaded_references=list(skill.references.files.keys()) if include_references else []
                )

        # Fallback: simple LLM evaluation
        system_prompt = """You are an AI assistant completing a task with full skill context.

You have access to the complete original skill instructions and all references.
Use this information to complete the task.

IMPORTANT: Respond with a JSON object:
{
  "output": "Your response to complete the task",
  "confidence": 0.0-1.0,
  "used_references": ["list of references used"],
  "reasoning": "How you used the skill context"
}"""

        user_prompt = f"""Task Query: {task.query}
Expected Outcome: {task.expected_outcome}

Complete this task using the full skill context provided.

Skill Context:
{full_context}"""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        score = result.get("confidence", 0.0)

        return ConditionScore(
            condition=EvaluationCondition.A,
            pytest_score=0.0,
            llm_judge_score=score,
            weighted_score=score,
            task_id=task.task_id,
            loaded_references=result.get("used_references", []),
            details={"output": result.get("output", ""), "reasoning": result.get("reasoning", "")}
        )

    def evaluate_condition_C(
        self,
        task: EvaluationTask,
        compressed_skill: Skill,
        on_demand_modules: OnDemandModules,
        routing_metadata: dict[str, RoutingMetadata],
        enable_read_file: bool = True,
        is_code_task: bool = False
    ) -> ConditionScore:
        """
        Evaluate task with compressed skill + read_file tool (Condition C).

        Agent receives only core rules, can load on-demand modules
        via simulated read_file tool calls.

        Per SkillReducer paper Section IV-B:
        - Code execution tasks: pytest (52.3%) + LLM judge (47.7%)
        - Rubric tasks: LLM judge only
        - Cohen's kappa validation for evaluator agreement

        Args:
            task: Evaluation task with query and expected outcome.
            compressed_skill: Compressed skill with core rules only.
            on_demand_modules: On-demand modules available for loading.
            routing_metadata: Routing metadata for module matching.
            enable_read_file: Whether to simulate read_file tool calls.
            is_code_task: Whether this is a code execution task (requires pytest).

        Returns:
            ConditionScore for condition C with loaded references tracked.
        """
        # Build base context (core rules only)
        core_rules = "\n".join([
            b.content for b in compressed_skill.body.content_blocks
            if _is_content_type(b, ContentType.CORE_RULE)
        ])

        loaded_references = []

        # Simulate read_file tool calls if enabled
        if enable_read_file and on_demand_modules and self.enable_read_file_tool:
            simulator = ReadFileToolSimulator(
                on_demand_modules=on_demand_modules,
                routing_metadata=routing_metadata,
                max_calls=self.max_tool_calls
            )

            # Get tool call suggestions
            decisions = simulator.suggest_tool_calls(
                llm_client=self.llm_client,
                query=task.query,
                current_context=core_rules
            )

            # Execute tool calls
            tool_calls = simulator.execute_tool_calls(decisions)

            # Build augmented context
            augmented_context = simulator.build_augmented_context(core_rules, tool_calls)
            loaded_references = [tc.module_name for tc in tool_calls]
        else:
            augmented_context = core_rules

        # Use hybrid scoring if enabled
        if self.use_hybrid_scoring and self.hybrid_evaluator:
            if is_code_task:
                # Code task: pytest + LLM judge
                # Generate pytest test file and run
                pytest_result = self.hybrid_evaluator.pytest.run_pytest(
                    self.hybrid_evaluator.pytest.generate_test_file(
                        task_id=task.task_id,
                        expected_outcome=task.expected_outcome,
                        skill_context=augmented_context[:2000]
                    )
                )
                pytest_score = pytest_result.score if pytest_result.passed else 0.0

                # LLM judge evaluation
                llm_judge_score = self._llm_judge_eval(
                    task, augmented_context, compressed_skill.name
                )

                # Calculate weighted score per paper (pytest 52.3%, judge 47.7%)
                weighted_score = self.hybrid_evaluator.calculate_weighted_score(
                    pytest_score=pytest_score,
                    judge_score=llm_judge_score,
                    is_code_task=True
                )

                return ConditionScore(
                    condition=EvaluationCondition.C,
                    pytest_score=pytest_score,
                    llm_judge_score=llm_judge_score,
                    weighted_score=weighted_score,
                    task_id=task.task_id,
                    loaded_references=loaded_references,
                    details={
                        "pytest_passed": pytest_result.passed,
                        "pytest_error": pytest_result.error
                    }
                )
            else:
                # Rubric task: LLM judge only
                llm_judge_score = self._llm_judge_eval(
                    task, augmented_context, compressed_skill.name
                )

                return ConditionScore(
                    condition=EvaluationCondition.C,
                    pytest_score=0.0,
                    llm_judge_score=llm_judge_score,
                    weighted_score=llm_judge_score,  # No pytest weight for rubric tasks
                    task_id=task.task_id,
                    loaded_references=loaded_references
                )

        # Fallback: simple LLM evaluation (original implementation)
        system_prompt = """You are an AI assistant completing a task with compressed skill context.

You have access to the core skill rules. Additional reference modules are available
via the read_file tool if needed.

IMPORTANT: Respond with a JSON object:
{
  "output": "Your response to complete the task",
  "confidence": 0.0-1.0,
  "reasoning": "How you completed the task with available context"
}"""

        user_prompt = f"""Task Query: {task.query}
Expected Outcome: {task.expected_outcome}

Complete this task using the skill context provided.

Available Context (Core Rules + Loaded References):
{augmented_context[:4000]}"""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        score = result.get("confidence", 0.0)

        return ConditionScore(
            condition=EvaluationCondition.C,
            pytest_score=0.0,
            llm_judge_score=score,
            weighted_score=score,
            task_id=task.task_id,
            loaded_references=loaded_references,
            details={"output": result.get("output", ""), "reasoning": result.get("reasoning", "")}
        )

    def _llm_judge_eval(
        self,
        task: EvaluationTask,
        context: str,
        skill_name: str
    ) -> float:
        """
        LLM judge evaluation using rubric.

        Args:
            task: Evaluation task with rubric criteria.
            context: Skill context for evaluation.
            skill_name: Skill name for context.

        Returns:
            LLM judge score (0.0-1.0).
        """
        # Use LLMJudge's evaluate_with_rubric if available
        if self.hybrid_evaluator and self.hybrid_evaluator.llm_judge:
            result = self.hybrid_evaluator.llm_judge.evaluate_with_rubric(
                task_id=task.task_id,
                query=task.query,
                actual_output=context,  # Use context as output for evaluation
                expected_outcome=task.expected_outcome,
                skill_context=f"Skill: {skill_name}"
            )
            return self.hybrid_evaluator.llm_judge.calculate_judge_score(result.rubric_scores)

        # Fallback: simple confidence evaluation
        system_prompt = """You are an LLM judge evaluating task completion quality.

Evaluate the response against the expected outcome using the rubric:
- Correctness: Does the response address the task correctly? (weight 0.4)
- Completeness: Does the response cover all required aspects? (weight 0.3)
- Quality: Is the response well-structured and clear? (weight 0.3)

IMPORTANT: Respond with a JSON object:
{
  "correctness": 0.0-1.0,
  "completeness": 0.0-1.0,
  "quality": 0.0-1.0,
  "overall_score": 0.0-1.0
}"""

        user_prompt = f"""Task Query: {task.query}
Expected Outcome: {task.expected_outcome}

Skill Context:
{context[:2000]}

Evaluate the quality of this skill context for completing the task."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("overall_score", 0.0)

    def calculate_retention(
        self,
        scores_D: list[ConditionScore],
        scores_A: list[ConditionScore],
        scores_C: list[ConditionScore],
        threshold: float = 1.0
    ) -> list[RetentionResult]:
        """
        Calculate retention for each task: score_C / score_A.

        Per SkillReducer paper, retention measures how well compressed skill
        maintains performance compared to original.

        Args:
            scores_D: List of Condition D scores (baseline).
            scores_A: List of Condition A scores (original skill).
            scores_C: List of Condition C scores (compressed skill).
            threshold: Minimum retention threshold (default 1.0 per paper).

        Returns:
            List of RetentionResult for each task.
        """
        results = []

        for d, a, c in zip(scores_D, scores_A, scores_C):
            # Handle edge cases
            if a.weighted_score == 0:
                # Original skill failed - retention defaults to 1.0 per paper
                retention = 1.0
            else:
                retention = c.weighted_score / a.weighted_score

            # Calculate improvement over baseline
            if d.weighted_score > 0:
                improvement = (a.weighted_score - d.weighted_score) / d.weighted_score
            else:
                improvement = a.weighted_score

            results.append(RetentionResult(
                score_D=d.weighted_score,
                score_A=a.weighted_score,
                score_C=c.weighted_score,
                retention=min(retention, 1.0),  # Cap at 1.0
                passed=retention >= threshold,
                improvement_over_baseline=improvement
            ))

        return results

    def run_three_condition_evaluation(
        self,
        skill: Skill,
        compressed_skill: Skill,
        on_demand_modules: OnDemandModules,
        routing_metadata: dict[str, RoutingMetadata],
        num_tasks: int = 5,
        threshold: float = 1.0,
        calculate_kappa: bool = True
    ) -> tuple[list[RetentionResult], list[EvaluationTask], bool, Optional[float]]:
        """
        Full three-condition evaluation pipeline.

        Generates tasks, evaluates D/A/C for each, calculates retention.

        Per SkillReducer paper Section IV-B:
        - 5 tasks per skill
        - Mix of core-only and needs-reference tasks
        - Code tasks: pytest 52.3% + LLM judge 47.7%
        - Rubric tasks: LLM judge only
        - Cohen's kappa validation for evaluator agreement (κ≥0.8 threshold)

        Args:
            skill: Original skill for condition A.
            compressed_skill: Compressed skill for condition C.
            on_demand_modules: On-demand modules for condition C.
            routing_metadata: Routing metadata for read_file simulation.
            num_tasks: Number of evaluation tasks (default 5 per paper).
            threshold: Retention threshold (default 0.86).
            calculate_kappa: Whether to calculate Cohen's kappa for evaluator agreement.

        Returns:
            Tuple of (retention_results, tasks, overall_passed, cohens_kappa).
        """
        # Generate evaluation tasks with proper type mix
        tasks = self.generate_evaluation_tasks(skill, num_tasks, ensure_task_mix=True)

        # Evaluate all three conditions for each task
        scores_D = []
        scores_A = []
        scores_C = []

        for task in tasks:
            # Condition D: No skill
            score_d = self.evaluate_condition_D(
                task, skill.name, is_code_task=task.is_code_task
            )
            scores_D.append(score_d)

            # Condition A: Original skill
            score_a = self.evaluate_condition_A(
                task, skill,
                include_references=True,
                is_code_task=task.is_code_task
            )
            scores_A.append(score_a)

            # Condition C: Compressed skill with read_file
            score_c = self.evaluate_condition_C(
                task, compressed_skill, on_demand_modules, routing_metadata,
                is_code_task=task.is_code_task
            )
            scores_C.append(score_c)

        # Calculate retention
        retention_results = self.calculate_retention(scores_D, scores_A, scores_C, threshold)

        # Overall pass: all tasks must pass retention threshold
        overall_passed = all(r.passed for r in retention_results)

        # Cohen's kappa validation if hybrid scoring enabled
        cohens_kappa = None
        if calculate_kappa and self.use_hybrid_scoring and self.hybrid_evaluator:
            # Calculate kappa between pytest and LLM judge scores
            pytest_scores = [s.pytest_score for s in scores_C if s.pytest_score > 0]
            judge_scores = [s.llm_judge_score for s in scores_C if s.pytest_score > 0]

            if len(pytest_scores) >= 2:
                kappa, kappa_passed = self.hybrid_evaluator.calculate_cohens_kappa(
                    pytest_scores, judge_scores
                )
                cohens_kappa = kappa

                # Per paper Section VII: κ≥0.8 required for evaluator agreement
                if kappa < 0.8:
                    # Log warning about evaluator disagreement
                    pass

        return retention_results, tasks, overall_passed, cohens_kappa


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
        enable_gate2: bool = True,
        use_real_cli: bool = False,
        cli_path: str = "claude",
        cli_timeout: int = 60
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
            use_real_cli: Use real Claude Code CLI for Phase 2 validation.
            cli_path: Path to Claude Code CLI executable.
            cli_timeout: Timeout in seconds for CLI subprocess calls.
        """
        self.llm_client = llm_client
        self.stage1_optimizer = stage1_optimizer or Stage1Optimizer(
            llm_client,
            use_real_cli=use_real_cli,
            cli_path=cli_path,
            cli_timeout=cli_timeout
        )
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
                # Per paper Eq. 3: include reference modules so concepts
                # moved to on-demand modules by progressive disclosure
                # are correctly counted as preserved, not missing.
                ref_modules_for_gate1 = None
                if on_demand_modules is not None:
                    ref_modules_for_gate1 = {}
                    if on_demand_modules.examples:
                        ref_modules_for_gate1["on-demand-examples.md"] = "\n".join(
                            b.content for b in on_demand_modules.examples
                        )
                    if on_demand_modules.templates:
                        ref_modules_for_gate1["on-demand-templates.md"] = "\n".join(
                            b.content for b in on_demand_modules.templates
                        )
                    if on_demand_modules.background:
                        ref_modules_for_gate1["on-demand-background.md"] = "\n".join(
                            b.content for b in on_demand_modules.background
                        )
                    if not ref_modules_for_gate1:
                        ref_modules_for_gate1 = None

                faithfulness_result = self.quality_gates.run_faithfulness_gate(
                    skill, final_blocks,
                    reference_modules=ref_modules_for_gate1
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
                        on_demand_modules.background = [
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
                    max_iterations=2,
                    original_blocks=original_blocks
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
    failed_criteria: Optional[list[str]] = None,
    use_real_cli: bool = False,
    cli_path: str = "claude",
    cli_timeout: int = 60
) -> PipelineResult:
    """
    Convenience function to run the complete compression pipeline.

    Args:
        skill: The skill to compress.
        llm_client: LLM client for all operations.
        enable_gate1: Whether to enable Gate 1.
        enable_gate2: Whether to enable Gate 2.
        failed_criteria: Optional failed criteria for Gate 2 (deprecated - auto-generated now).
        use_real_cli: Use real Claude Code CLI for Phase 2 validation.
        cli_path: Path to Claude Code CLI executable.
        cli_timeout: Timeout in seconds for CLI subprocess calls.

    Returns:
        PipelineResult with complete compression results.
    """
    pipeline = CompressionPipeline(
        llm_client=llm_client,
        enable_gate1=enable_gate1,
        enable_gate2=enable_gate2,
        use_real_cli=use_real_cli,
        cli_path=cli_path,
        cli_timeout=cli_timeout
    )
    return pipeline.run_pipeline(skill, failed_criteria=failed_criteria)
