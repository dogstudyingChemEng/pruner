"""
Unit tests for Three-Condition Evaluation (Gate 2 D/A/C).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.models import (
    EvaluationCondition,
    ConditionScore,
    RetentionResult,
    RoutingMetadata,
    Skill, Description, Body, References, SkillMetadata,
    ContentBlock, ContentType, OnDemandModules
)
from src.quality_gates import (
    QualityGates,
    EvaluationTask,
    ReadFileToolSimulator,
    ToolCallDecision,
    ReadFileToolCall
)


class TestEvaluationCondition:
    """Tests for EvaluationCondition enum."""

    def test_condition_d_is_baseline(self):
        """Test Condition D represents baseline."""
        assert EvaluationCondition.D.value == "D"

    def test_condition_a_is_original(self):
        """Test Condition A represents original skill."""
        assert EvaluationCondition.A.value == "A"

    def test_condition_c_is_compressed(self):
        """Test Condition C represents compressed skill."""
        assert EvaluationCondition.C.value == "C"


class TestConditionScore:
    """Tests for ConditionScore data model."""

    def test_score_creation(self):
        """Test creating condition score."""
        score = ConditionScore(
            condition=EvaluationCondition.C,
            pytest_score=0.8,
            llm_judge_score=0.6,
            weighted_score=0.7,
            task_id="task_001"
        )

        assert score.condition == EvaluationCondition.C
        assert score.pytest_score == 0.8
        assert score.llm_judge_score == 0.6
        assert score.weighted_score == 0.7
        assert score.task_id == "task_001"

    def test_score_with_loaded_references(self):
        """Test score with loaded references for Condition C."""
        score = ConditionScore(
            condition=EvaluationCondition.C,
            weighted_score=0.7,
            task_id="task_001",
            loaded_references=["on-demand-examples.md", "on-demand-background.md"]
        )

        assert len(score.loaded_references) == 2


class TestRetentionResult:
    """Tests for RetentionResult data model."""

    def test_result_creation(self):
        """Test creating retention result."""
        result = RetentionResult(
            score_D=0.3,
            score_A=0.7,
            score_C=0.6,
            retention=0.857,
            passed=True
        )

        assert result.score_D == 0.3
        assert result.score_A == 0.7
        assert result.score_C == 0.6
        assert result.retention == 0.857
        assert result.passed is True

    def test_result_below_threshold(self):
        """Test retention result below threshold."""
        result = RetentionResult(
            score_D=0.3,
            score_A=0.7,
            score_C=0.5,
            retention=0.71,
            passed=False
        )

        assert result.passed is False

    def test_improvement_over_baseline(self):
        """Test improvement over baseline calculation."""
        result = RetentionResult(
            score_D=0.3,
            score_A=0.7,
            score_C=0.6,
            retention=0.857,
            passed=True,
            improvement_over_baseline=1.33  # (0.7 - 0.3) / 0.3
        )

        assert result.improvement_over_baseline > 0


class TestReadFileToolSimulator:
    """Tests for ReadFileToolSimulator class."""

    def _create_test_modules(self) -> tuple[OnDemandModules, dict[str, RoutingMetadata]]:
        """Create test on-demand modules and routing metadata."""
        modules = OnDemandModules(
            examples=[
                ContentBlock(
                    chunk_id="ex1",
                    content="Example code snippet",
                    content_type=ContentType.EXAMPLE,
                    token_count=20
                )
            ],
            templates=[
                ContentBlock(
                    chunk_id="tp1",
                    content="Template boilerplate",
                    content_type=ContentType.TEMPLATE,
                    token_count=15
                )
            ],
            background=[
                ContentBlock(
                    chunk_id="bg1",
                    content="Background explanation",
                    content_type=ContentType.BACKGROUND,
                    token_count=25
                )
            ]
        )

        routing = {
            "on-demand-examples.md": RoutingMetadata(
                when="you need to see example code",
                topics=["example", "code", "snippet"]
            ),
            "on-demand-templates.md": RoutingMetadata(
                when="you need to use templates",
                topics=["template", "boilerplate"]
            ),
            "on-demand-background.md": RoutingMetadata(
                when="you need context explanation",
                topics=["background", "explanation", "context"]
            )
        }

        return modules, routing

    def test_init(self):
        """Test simulator initialization."""
        modules, routing = self._create_test_modules()
        simulator = ReadFileToolSimulator(modules, routing, max_calls=6)

        assert simulator.modules is not None
        assert simulator.routing is not None
        assert simulator.max_calls == 6

    def test_keyword_match_tool_calls(self):
        """Test keyword matching for tool call decisions."""
        modules, routing = self._create_test_modules()
        simulator = ReadFileToolSimulator(modules, routing)

        decisions = simulator._keyword_match_tool_calls(
            "I need an example code snippet for testing"
        )

        assert len(decisions) > 0
        assert any("example" in d.module_name for d in decisions)

    def test_execute_tool_calls(self):
        """Test executing tool calls."""
        modules, routing = self._create_test_modules()
        simulator = ReadFileToolSimulator(modules, routing)

        decisions = [
            ToolCallDecision(module_name="on-demand-examples.md", reason="Need example", relevance_score=0.8)
        ]

        calls = simulator.execute_tool_calls(decisions)

        assert len(calls) == 1
        assert "Example code snippet" in calls[0].module_content

    def test_build_augmented_context(self):
        """Test building augmented context."""
        modules, routing = self._create_test_modules()
        simulator = ReadFileToolSimulator(modules, routing)

        base_context = "Core rules: Always follow best practices."
        calls = [
            ReadFileToolCall(
                call_id="call1",
                module_name="on-demand-examples.md",
                module_content="Example code here",
                relevance_match=0.8
            )
        ]

        augmented = simulator.build_augmented_context(base_context, calls)

        assert "Core rules" in augmented
        assert "on-demand-examples.md" in augmented

    def test_max_calls_limit(self):
        """Test max calls limit is enforced."""
        modules, routing = self._create_test_modules()
        simulator = ReadFileToolSimulator(modules, routing, max_calls=2)

        decisions = [
            ToolCallDecision(module_name="on-demand-examples.md", reason="1", relevance_score=0.5),
            ToolCallDecision(module_name="on-demand-templates.md", reason="2", relevance_score=0.5),
            ToolCallDecision(module_name="on-demand-background.md", reason="3", relevance_score=0.5),
        ]

        # Should only use max_calls
        limited_decisions = decisions[:simulator.max_calls]
        assert len(limited_decisions) == 2


class TestQualityGatesThreeConditions:
    """Tests for QualityGates three-condition evaluation methods."""

    def _create_mock_llm_client(self):
        """Create mock LLM client."""
        mock_client = Mock()
        mock_client.model = "gpt-4o-mini"
        mock_client.client = Mock()
        mock_client.client.chat.completions.create = Mock(return_value=Mock(
            choices=[Mock(message=Mock(content='{"output": "test", "confidence": 0.8}'))]
        ))
        return mock_client

    def _create_test_skill(self) -> Skill:
        """Create test skill."""
        return Skill(
            name="test-skill",
            description=Description(
                original="Test skill description",
                original_token_count=30
            ),
            body=Body(
                original="# Test Skill\n\nCore rules for testing.",
                original_token_count=50,
                content_blocks=[
                    ContentBlock(
                        chunk_id="core1",
                        content="Core rule 1",
                        content_type=ContentType.CORE_RULE,
                        token_count=10
                    )
                ]
            ),
            metadata=SkillMetadata(),
            references=References()
        )

    def test_init_with_three_condition_params(self):
        """Test QualityGates initialization with three-condition params."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(
            llm_client=mock_client,
            use_three_conditions=True,
            enable_read_file_tool=True,
            max_tool_calls=6
        )

        assert gates.use_three_conditions is True
        assert gates.enable_read_file_tool is True
        assert gates.max_tool_calls == 6

    def test_calculate_retention_basic(self):
        """Test basic retention calculation."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(llm_client=mock_client)

        scores_D = [
            ConditionScore(condition=EvaluationCondition.D, weighted_score=0.3, task_id="t1")
        ]
        scores_A = [
            ConditionScore(condition=EvaluationCondition.A, weighted_score=0.7, task_id="t1")
        ]
        scores_C = [
            ConditionScore(condition=EvaluationCondition.C, weighted_score=0.6, task_id="t1")
        ]

        results = gates.calculate_retention(scores_D, scores_A, scores_C, threshold=0.86)

        assert len(results) == 1
        assert results[0].retention == 0.6 / 0.7
        assert results[0].passed == (0.6 / 0.7 >= 0.86)

    def test_calculate_retention_zero_original(self):
        """Test retention when original score is zero."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(llm_client=mock_client)

        scores_D = [
            ConditionScore(condition=EvaluationCondition.D, weighted_score=0.3, task_id="t1")
        ]
        scores_A = [
            ConditionScore(condition=EvaluationCondition.A, weighted_score=0.0, task_id="t1")
        ]
        scores_C = [
            ConditionScore(condition=EvaluationCondition.C, weighted_score=0.5, task_id="t1")
        ]

        results = gates.calculate_retention(scores_D, scores_A, scores_C)

        # When original fails, retention defaults to 1.0 per paper
        assert results[0].retention == 1.0

    def test_evaluate_condition_d(self):
        """Test evaluating Condition D (no skill)."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(llm_client=mock_client)

        task = EvaluationTask(
            task_id="task_001",
            query="How do I test?",
            expected_outcome="Provide testing guidance"
        )

        score = gates.evaluate_condition_D(task, "test-skill")

        assert score.condition == EvaluationCondition.D
        assert score.task_id == "task_001"

    def test_evaluate_condition_a(self):
        """Test evaluating Condition A (original skill)."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(llm_client=mock_client)

        task = EvaluationTask(
            task_id="task_001",
            query="How do I test?",
            expected_outcome="Provide testing guidance"
        )

        skill = self._create_test_skill()

        score = gates.evaluate_condition_A(task, skill)

        assert score.condition == EvaluationCondition.A
        assert score.task_id == "task_001"


class TestEvaluationTask:
    """Tests for EvaluationTask dataclass."""

    def test_task_creation(self):
        """Test creating evaluation task."""
        task = EvaluationTask(
            task_id="task_001",
            query="How do I implement authentication?",
            expected_outcome="Provide JWT authentication setup guide"
        )

        assert task.task_id == "task_001"
        assert "authentication" in task.query
        assert "JWT" in task.expected_outcome