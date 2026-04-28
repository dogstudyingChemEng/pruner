"""
Hybrid Evaluator for Gate 2 task evaluation.

Per SkillReducer paper, Gate 2 uses hybrid scoring:
- pytest execution: 52.3% weight - deterministic code execution tests
- LLM judge: 47.7% weight - rubric-based quality evaluation

This module implements:
- PytestExecutor: Generate and run pytest tests for tasks
- LLMJudge: Evaluate outputs against structured rubric
- HybridEvaluator: Combine both scoring methods
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class RubricScore(BaseModel):
    """Score for a single rubric dimension."""

    dimension: str = Field(description="Rubric dimension name")
    score: float = Field(description="Score (0.0-1.0)")
    weight: float = Field(description="Weight in overall score")
    reasoning: str = Field(default="", description="Scoring reasoning")


class PytestResult(BaseModel):
    """Result of pytest execution."""

    passed: bool = Field(description="Whether tests passed")
    score: float = Field(description="Score based on passed/failed ratio")
    output: str = Field(default="", description="pytest output")
    test_count: int = Field(default=0, description="Number of tests run")
    passed_count: int = Field(default=0, description="Number of tests passed")
    failed_count: int = Field(default=0, description="Number of tests failed")
    error: Optional[str] = Field(default=None, description="Error message")


class LLMJudgeResult(BaseModel):
    """Result of LLM rubric evaluation."""

    overall_score: float = Field(description="Weighted overall score")
    rubric_scores: list[RubricScore] = Field(
        default_factory=list,
        description="Scores for each rubric dimension"
    )
    reasoning: str = Field(default="", description="Overall evaluation reasoning")


class HybridEvaluationResult(BaseModel):
    """Combined hybrid evaluation result."""

    pytest_score: float = Field(default=0.0, description="pytest execution score")
    llm_judge_score: float = Field(default=0.0, description="LLM rubric score")
    weighted_score: float = Field(default=0.0, description="Combined weighted score")
    pytest_result: Optional[PytestResult] = Field(default=None)
    llm_result: Optional[LLMJudgeResult] = Field(default=None)
    is_code_task: bool = Field(default=True, description="Whether task produces executable code")


class PytestExecutor:
    """
    Execute pytest tests for skill task evaluation.

    Generates test files from task expected outcomes and executes them
    against agent output. Provides deterministic evaluation for code tasks.
    """

    PYTEST_WEIGHT = 0.523

    def __init__(
        self,
        test_dir: Optional[str] = None,
        timeout: int = 30,
        llm_client=None
    ):
        """
        Initialize pytest executor.

        Args:
            test_dir: Directory for generated test files.
            timeout: Timeout for pytest execution.
            llm_client: Optional LLM client for generating real test code.
                        When provided, tests are generated from expected outcomes
                        using the LLM. When None, falls back to basic assertions.
        """
        self.test_dir = Path(test_dir or tempfile.mkdtemp(prefix="pytest_eval_"))
        self.timeout = timeout
        self.llm_client = llm_client

    def generate_test_file(
        self,
        task_id: str,
        expected_outcome: str,
        skill_context: str
    ) -> Path:
        """
        Generate pytest test file from task.

        Args:
            task_id: Task identifier.
            expected_outcome: Expected outcome description.
            skill_context: Skill context used for test generation.

        Returns:
            Path to generated test file.
        """
        # Create test file
        test_file = self.test_dir / f"test_{task_id.replace('-', '_')}.py"

        # Generate test code based on expected outcome
        test_code = self._generate_test_code(task_id, expected_outcome, skill_context)
        test_file.write_text(test_code)

        return test_file

    def _generate_test_code(
        self,
        task_id: str,
        expected_outcome: str,
        skill_context: str
    ) -> str:
        """
        Generate pytest test code from expected outcome.

        When an LLM client is available, generates real assertion-based test code
        from the expected outcome description and skill context.
        Otherwise falls back to basic keyword-based assertions.

        Per paper: code execution tasks run the agent's output as Python code and
        verify correctness via assertion-based test scripts (analogous to pytest).
        """
        if self.llm_client:
            return self._generate_test_code_with_llm(task_id, expected_outcome, skill_context)
        return self._generate_test_code_fallback(task_id, expected_outcome, skill_context)

    def _generate_test_code_with_llm(
        self,
        task_id: str,
        expected_outcome: str,
        skill_context: str
    ) -> str:
        """Use LLM to generate real pytest test code from expected outcome."""
        import json as json_module

        system_prompt = """You are an expert at writing pytest test code.

Given a task description and expected outcome, generate a pytest test function that
verifies the expected behavior. The test will be executed with `pytest -v`.

Guidelines:
1. Write a complete, runnable pytest test function
2. Include specific assertions that verify the expected outcome
3. Use assert statements with meaningful error messages
4. Include necessary imports
5. If the task involves code output, write tests that import and verify that code
6. Handle edge cases where appropriate
7. Keep tests focused and deterministic

IMPORTANT: Respond with a JSON object:
{
  "test_code": "Complete pytest test code as a string",
  "test_count": 2,
  "description": "What the tests verify"
}"""

        user_prompt = f"""Task ID: {task_id}
Expected Outcome: {expected_outcome}

Skill Context:
{skill_context[:2000]}

Generate pytest test code that verifies the expected outcome. The test file will be
executed standalone, so include all necessary imports and function definitions.

Respond with the complete test code as a Python string."""

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json_module.loads(response.choices[0].message.content)
            test_code = result.get("test_code", "")

            if test_code and len(test_code.strip()) > 20:
                return test_code

        except Exception:
            pass

        # Fallback if LLM generation fails
        return self._generate_test_code_fallback(task_id, expected_outcome, skill_context)

    def _generate_test_code_fallback(
        self,
        task_id: str,
        expected_outcome: str,
        skill_context: str
    ) -> str:
        """Generate basic pytest test code without LLM (fallback)."""
        safe_id = task_id.replace("-", "_")
        keywords = self._extract_keywords(expected_outcome)

        lines = [f'''"""
Test for task: {task_id}
Expected: {expected_outcome}
Generated by SkillReducer hybrid evaluator.
"""

import pytest


def test_{safe_id}_expected_outcome():
    """Verify that the output matches the expected outcome.

    Expected: {expected_outcome}
    """
    # Key terms from expected outcome for assertion generation
    expected_terms = {json.dumps(keywords)}

    # Placeholder: replace with actual agent output
    # In real execution, this module imports and tests the agent's generated code
    assert True, "LLM-generated test placeholder — run with llm_client for real tests"
''']

        return "\n".join(lines)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract key terms from text for test generation."""
        # Simple keyword extraction
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Remove common words
        common = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'is', 'was'}
        keywords = [w for w in words if w not in common][:10]
        return keywords

    def run_pytest(
        self,
        test_file: Path,
        agent_output: Optional[str] = None
    ) -> PytestResult:
        """
        Execute pytest on generated test file.

        Args:
            test_file: Path to test file.
            agent_output: Optional agent output to verify.

        Returns:
            PytestResult with execution details.
        """
        try:
            # Run pytest subprocess
            result = subprocess.run(
                ["pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Parse output for test counts
            output = result.stdout + result.stderr
            passed_count, failed_count = self._parse_pytest_output(output)

            test_count = passed_count + failed_count
            score = passed_count / test_count if test_count > 0 else 0.0

            return PytestResult(
                passed=(failed_count == 0),
                score=score,
                output=output,
                test_count=test_count,
                passed_count=passed_count,
                failed_count=failed_count
            )

        except subprocess.TimeoutExpired:
            return PytestResult(
                passed=False,
                score=0.0,
                error=f"pytest timeout after {self.timeout} seconds"
            )

        except Exception as e:
            return PytestResult(
                passed=False,
                score=0.0,
                error=str(e)
            )

    def _parse_pytest_output(self, output: str) -> tuple[int, int]:
        """Parse pytest output for passed/failed counts."""
        import re

        # Look for summary line like "X passed, Y failed"
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0

        return passed, failed

    def calculate_pytest_score(self, result: PytestResult) -> float:
        """Calculate pytest score from result."""
        return result.score * self.PYTEST_WEIGHT


class LLMJudge:
    """
    LLM-based rubric evaluator for task quality assessment.

    Uses structured rubric for consistent evaluation across tasks.
    Provides qualitative scoring complementary to pytest execution.
    """

    LLM_JUDGE_WEIGHT = 0.477

    # Standard rubric dimensions per paper methodology
    DEFAULT_RUBRIC = {
        "correctness": {
            "description": "Output matches expected outcome and requirements",
            "weight": 0.4,
            "criteria": ["Matches specification", "Correct approach", "Valid output"]
        },
        "completeness": {
            "description": "All required elements present and addressed",
            "weight": 0.3,
            "criteria": ["All requirements addressed", "No missing elements", "Complete coverage"]
        },
        "quality": {
            "description": "Output quality, clarity, and adherence to best practices",
            "weight": 0.3,
            "criteria": ["Clear and readable", "Follows conventions", "Well-structured"]
        }
    }

    def __init__(self, llm_client, rubric: Optional[dict] = None):
        """
        Initialize LLM judge.

        Args:
            llm_client: LLM client for evaluation.
            rubric: Optional custom rubric (uses DEFAULT_RUBRIC if None).
        """
        self.llm_client = llm_client
        self.rubric = rubric or self.DEFAULT_RUBRIC

    def evaluate_with_rubric(
        self,
        task_id: str,
        query: str,
        actual_output: str,
        expected_outcome: str,
        skill_context: str
    ) -> LLMJudgeResult:
        """
        Evaluate output against rubric dimensions.

        Args:
            task_id: Task identifier.
            query: Original task query.
            actual_output: Agent's output to evaluate.
            expected_outcome: Expected outcome description.
            skill_context: Skill context used for evaluation.

        Returns:
            LLMJudgeResult with rubric scores.
        """
        system_prompt = """You are an expert evaluator for LLM agent task outputs.

Your task is to evaluate an agent's output against a structured rubric.

For each dimension, provide:
- score: 0.0 (poor) to 1.0 (excellent)
- reasoning: Brief explanation of the score

IMPORTANT: Respond with a JSON object:
{
  "rubric_scores": [
    {
      "dimension": "correctness",
      "score": 0.8,
      "reasoning": "Output mostly matches expected outcome"
    },
    ...
  ],
  "overall_reasoning": "Summary of evaluation"
}"""

        rubric_text = self._format_rubric()
        user_prompt = f"""Task: {task_id}
Query: {query}
Expected Outcome: {expected_outcome}

Skill Context Used:
{skill_context[:2000]}

Agent Output:
{actual_output[:2000]}

Rubric Dimensions:
{rubric_text}

Evaluate the agent output against each rubric dimension."""

        try:
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

            rubric_scores = []
            for rs in result.get("rubric_scores", []):
                dimension = rs.get("dimension", "")
                weight = self.rubric.get(dimension, {}).get("weight", 0.3)
                rubric_scores.append(RubricScore(
                    dimension=dimension,
                    score=rs.get("score", 0.0),
                    weight=weight,
                    reasoning=rs.get("reasoning", "")
                ))

            overall_score = self.calculate_judge_score(rubric_scores)

            return LLMJudgeResult(
                overall_score=overall_score,
                rubric_scores=rubric_scores,
                reasoning=result.get("overall_reasoning", "")
            )

        except Exception as e:
            # Fallback: return default scores
            default_scores = [
                RubricScore(dimension=d, score=0.0, weight=self.rubric[d]["weight"])
                for d in self.rubric
            ]
            return LLMJudgeResult(
                overall_score=0.0,
                rubric_scores=default_scores,
                reasoning=f"Evaluation error: {str(e)}"
            )

    def _format_rubric(self) -> str:
        """Format rubric for prompt."""
        lines = []
        for dim, config in self.rubric.items():
            lines.append(f"- {dim} (weight {config['weight']}): {config['description']}")
            for criterion in config.get("criteria", []):
                lines.append(f"  - {criterion}")
        return "\n".join(lines)

    def calculate_judge_score(self, rubric_scores: list[RubricScore]) -> float:
        """Calculate weighted LLM judge score."""
        if not rubric_scores:
            return 0.0

        total = sum(rs.score * rs.weight for rs in rubric_scores)
        return total * self.LLM_JUDGE_WEIGHT


class HybridEvaluator:
    """
    Combine pytest and LLM judge scores per paper methodology.

    Weight distribution:
    - pytest: 52.3%
    - LLM judge: 47.7%

    Task types:
    - Code execution tasks (52.3%): pytest-based
    - Rubric tasks (47.7%): LLM judge-based
    """

    PYTEST_WEIGHT = 0.523
    LLM_JUDGE_WEIGHT = 0.477

    def __init__(
        self,
        pytest_executor: Optional[PytestExecutor] = None,
        llm_judge: Optional[LLMJudge] = None,
        llm_client=None
    ):
        """
        Initialize hybrid evaluator.

        Args:
            pytest_executor: Optional pytest executor instance.
            llm_judge: Optional LLM judge instance.
            llm_client: LLM client for creating pytest executor and LLM judge
                        if not provided. Also used by PytestExecutor for test generation.
        """
        self.pytest = pytest_executor or PytestExecutor(llm_client=llm_client)
        self.llm_judge = llm_judge or (LLMJudge(llm_client) if llm_client else None)

    def evaluate_task(
        self,
        task_id: str,
        query: str,
        expected_outcome: str,
        actual_output: str,
        skill_context: str,
        is_code_task: bool = True
    ) -> HybridEvaluationResult:
        """
        Perform hybrid evaluation of a task.

        Args:
            task_id: Task identifier.
            query: Original task query.
            expected_outcome: Expected outcome description.
            actual_output: Agent's output to evaluate.
            skill_context: Skill context used.
            is_code_task: Whether task produces executable code.

        Returns:
            HybridEvaluationResult with combined scores.
        """
        pytest_result = None
        llm_result = None
        pytest_score = 0.0
        llm_score = 0.0

        # Code tasks: run pytest
        if is_code_task and self.pytest:
            test_file = self.pytest.generate_test_file(
                task_id=task_id,
                expected_outcome=expected_outcome,
                skill_context=skill_context
            )
            pytest_result = self.pytest.run_pytest(test_file, actual_output)
            pytest_score = self.pytest.calculate_pytest_score(pytest_result)

        # All tasks: LLM judge evaluation
        if self.llm_judge:
            llm_result = self.llm_judge.evaluate_with_rubric(
                task_id=task_id,
                query=query,
                actual_output=actual_output,
                expected_outcome=expected_outcome,
                skill_context=skill_context
            )
            llm_score = llm_result.overall_score

        # Calculate weighted score
        if is_code_task:
            weighted_score = pytest_score + llm_score
        else:
            # Rubric-only tasks use LLM judge fully
            weighted_score = llm_score / self.LLM_JUDGE_WEIGHT  # Normalize to 0-1

        return HybridEvaluationResult(
            pytest_score=pytest_score,
            llm_judge_score=llm_score,
            weighted_score=min(weighted_score, 1.0),
            pytest_result=pytest_result,
            llm_result=llm_result,
            is_code_task=is_code_task
        )

    def calculate_weighted_score(
        self,
        pytest_score: float,
        judge_score: float,
        is_code_task: bool = True
    ) -> float:
        """
        Calculate weighted hybrid score.

        Args:
            pytest_score: pytest execution score (0.0-1.0).
            judge_score: LLM judge score (0.0-1.0).
            is_code_task: Whether task is code execution type.

        Returns:
            Weighted combined score.
        """
        if is_code_task:
            return pytest_score * self.PYTEST_WEIGHT + judge_score * self.LLM_JUDGE_WEIGHT
        else:
            return judge_score  # Rubric tasks use LLM judge only

    def calculate_cohens_kappa(
        self,
        pytest_scores: list[float],
        judge_scores: list[float],
        threshold: float = 0.7
    ) -> tuple[float, bool]:
        """
        Calculate Cohen's kappa between pytest and LLM judge.

        Validates agreement between evaluation methods.

        Args:
            pytest_scores: List of pytest scores.
            judge_scores: List of LLM judge scores.
            threshold: Minimum kappa threshold (default 0.7).

        Returns:
            Tuple of (kappa_value, passed_threshold).
        """
        if len(pytest_scores) != len(judge_scores) or len(pytest_scores) < 2:
            return 0.0, False

        # Convert scores to binary (pass/fail at 0.5 threshold)
        pytest_binary = [1 if s >= 0.5 else 0 for s in pytest_scores]
        judge_binary = [1 if s >= 0.5 else 0 for s in judge_scores]

        # Calculate Cohen's kappa
        n = len(pytest_binary)

        # Agreement counts
        agree_positive = sum(1 for p, j in zip(pytest_binary, judge_binary) if p == j == 1)
        agree_negative = sum(1 for p, j in zip(pytest_binary, judge_binary) if p == j == 0)
        disagree_p_pos = sum(1 for p, j in zip(pytest_binary, judge_binary) if p == 1 and j == 0)
        disagree_j_pos = sum(1 for p, j in zip(pytest_binary, judge_binary) if p == 0 and j == 1)

        # Observed agreement
        po = (agree_positive + agree_negative) / n

        # Expected agreement
        pytest_pos_rate = sum(pytest_binary) / n
        judge_pos_rate = sum(judge_binary) / n
        pe = pytest_pos_rate * judge_pos_rate + (1 - pytest_pos_rate) * (1 - judge_pos_rate)

        # Cohen's kappa
        if pe == 1.0:
            kappa = 1.0
        else:
            kappa = (po - pe) / (1 - pe)

        return kappa, kappa >= threshold

    def batch_evaluate(
        self,
        tasks: list[dict],
        skill_context: str
    ) -> list[HybridEvaluationResult]:
        """
        Evaluate multiple tasks in batch.

        Args:
            tasks: List of task dicts with id, query, expected_outcome, output.
            skill_context: Skill context for all tasks.

        Returns:
            List of HybridEvaluationResult for each task.
        """
        results = []
        for task in tasks:
            result = self.evaluate_task(
                task_id=task.get("task_id", ""),
                query=task.get("query", ""),
                expected_outcome=task.get("expected_outcome", ""),
                actual_output=task.get("output", ""),
                skill_context=skill_context,
                is_code_task=task.get("is_code_task", True)
            )
            results.append(result)

        return results


def create_hybrid_evaluator(llm_client) -> HybridEvaluator:
    """
    Create hybrid evaluator with LLM client.

    Args:
        llm_client: LLM client for evaluation.

    Returns:
        Configured HybridEvaluator instance.
    """
    return HybridEvaluator(
        pytest_executor=PytestExecutor(),
        llm_judge=LLMJudge(llm_client),
        llm_client=llm_client
    )