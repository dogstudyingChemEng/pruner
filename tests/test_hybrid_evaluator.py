"""
Unit tests for Hybrid Evaluator (Gate 2 Hybrid Scoring).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.hybrid_evaluator import (
    RubricScore,
    PytestResult,
    LLMJudgeResult,
    HybridEvaluationResult,
    PytestExecutor,
    LLMJudge,
    HybridEvaluator
)


class TestPytestExecutor:
    """Tests for PytestExecutor class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        executor = PytestExecutor()

        assert executor.timeout == 30
        assert executor.test_dir is not None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        executor = PytestExecutor(test_dir="/tmp/tests", timeout=60)

        assert executor.timeout == 60
        assert executor.test_dir == Path("/tmp/tests")

    def test_generate_test_file(self):
        """Test generating test file."""
        executor = PytestExecutor()

        test_file = executor.generate_test_file(
            task_id="task_001",
            expected_outcome="Output should contain 'hello'",
            skill_context="This is a greeting skill"
        )

        assert test_file.exists()
        assert "test_task_001" in test_file.name

    def test_extract_keywords(self):
        """Test keyword extraction from text."""
        executor = PytestExecutor()
        keywords = executor._extract_keywords("This is a test for hello world greeting")

        assert "test" in keywords
        assert "hello" in keywords
        assert "world" in keywords

    @patch('src.hybrid_evaluator.subprocess.run')
    def test_run_pytest_success(self, mock_run):
        """Test running pytest successfully."""
        mock_run.return_value = MagicMock(
            stdout="test_task_001.py::test_task_001 PASSED\n2 passed",
            stderr="",
            returncode=0
        )

        executor = PytestExecutor()
        result = executor.run_pytest(Path("/tmp/test.py"))

        assert isinstance(result, PytestResult)

    @patch('src.hybrid_evaluator.subprocess.run')
    def test_run_pytest_timeout(self, mock_run):
        """Test pytest timeout handling."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", timeout=30)

        executor = PytestExecutor(timeout=30)
        result = executor.run_pytest(Path("/tmp/test.py"))

        assert result.passed is False
        assert "timeout" in result.error.lower()

    def test_parse_pytest_output(self):
        """Test parsing pytest output."""
        executor = PytestExecutor()
        output = "test_1 PASSED\ntest_2 FAILED\n2 passed, 1 failed"

        passed, failed = executor._parse_pytest_output(output)

        assert passed == 2
        assert failed == 1


class TestLLMJudge:
    """Tests for LLMJudge class."""

    def test_init_default_rubric(self):
        """Test initialization with default rubric."""
        mock_client = Mock()
        judge = LLMJudge(mock_client)

        assert judge.rubric == judge.DEFAULT_RUBRIC
        assert "correctness" in judge.rubric
        assert "completeness" in judge.rubric
        assert "quality" in judge.rubric

    def test_format_rubric(self):
        """Test rubric formatting."""
        mock_client = Mock()
        judge = LLMJudge(mock_client)

        formatted = judge._format_rubric()

        assert "correctness" in formatted
        assert "weight" in formatted

    def test_calculate_judge_score(self):
        """Test calculating weighted judge score."""
        mock_client = Mock()
        judge = LLMJudge(mock_client)

        rubric_scores = [
            RubricScore(dimension="correctness", score=0.8, weight=0.4),
            RubricScore(dimension="completeness", score=0.7, weight=0.3),
            RubricScore(dimension="quality", score=0.9, weight=0.3)
        ]

        score = judge.calculate_judge_score(rubric_scores)

        # Should be weighted and multiplied by LLM_JUDGE_WEIGHT (0.477)
        expected_weighted = 0.8 * 0.4 + 0.7 * 0.3 + 0.9 * 0.3
        expected = expected_weighted * 0.477

        assert abs(score - expected) < 0.01


class TestHybridEvaluator:
    """Tests for HybridEvaluator class."""

    def test_init_with_clients(self):
        """Test initialization with executor and judge."""
        executor = PytestExecutor()
        mock_client = Mock()
        judge = LLMJudge(mock_client)

        evaluator = HybridEvaluator(
            pytest_executor=executor,
            llm_judge=judge
        )

        assert evaluator.pytest is not None
        assert evaluator.llm_judge is not None

    def test_pytest_weight(self):
        """Test pytest weight constant."""
        assert HybridEvaluator.PYTEST_WEIGHT == 0.523

    def test_llm_judge_weight(self):
        """Test LLM judge weight constant."""
        assert HybridEvaluator.LLM_JUDGE_WEIGHT == 0.477

    def test_calculate_weighted_score_code_task(self):
        """Test weighted score calculation for code task."""
        evaluator = HybridEvaluator()

        score = evaluator.calculate_weighted_score(
            pytest_score=0.8,
            judge_score=0.6,
            is_code_task=True
        )

        expected = 0.8 * 0.523 + 0.6 * 0.477
        assert abs(score - expected) < 0.01

    def test_calculate_weighted_score_rubric_task(self):
        """Test weighted score calculation for rubric-only task."""
        evaluator = HybridEvaluator()

        score = evaluator.calculate_weighted_score(
            pytest_score=0.0,
            judge_score=0.7,
            is_code_task=False
        )

        assert score == 0.7  # Only LLM judge for rubric tasks

    def test_calculate_cohens_kappa_perfect_agreement(self):
        """Test Cohen's kappa with perfect agreement."""
        evaluator = HybridEvaluator()

        pytest_scores = [0.8, 0.8, 0.6, 0.6]
        judge_scores = [0.8, 0.8, 0.6, 0.6]

        kappa, passed = evaluator.calculate_cohens_kappa(pytest_scores, judge_scores)

        assert kappa == 1.0
        assert passed is True

    def test_calculate_cohens_kappa_no_agreement(self):
        """Test Cohen's kappa with no agreement."""
        evaluator = HybridEvaluator()

        pytest_scores = [0.8, 0.8]
        judge_scores = [0.3, 0.3]

        kappa, passed = evaluator.calculate_cohens_kappa(pytest_scores, judge_scores)

        assert kappa == 0.0
        assert passed is False


class TestHybridEvaluationResult:
    """Tests for HybridEvaluationResult data model."""

    def test_result_creation(self):
        """Test creating hybrid evaluation result."""
        result = HybridEvaluationResult(
            pytest_score=0.8,
            llm_judge_score=0.6,
            weighted_score=0.7,
            is_code_task=True
        )

        assert result.pytest_score == 0.8
        assert result.llm_judge_score == 0.6
        assert result.weighted_score == 0.7
        assert result.is_code_task is True

    def test_result_with_details(self):
        """Test result with detailed pytest and LLM results."""
        pytest_result = PytestResult(passed=True, score=0.8)
        llm_result = LLMJudgeResult(overall_score=0.6)

        result = HybridEvaluationResult(
            pytest_score=0.8,
            llm_judge_score=0.6,
            weighted_score=0.7,
            pytest_result=pytest_result,
            llm_result=llm_result
        )

        assert result.pytest_result is not None
        assert result.llm_result is not None