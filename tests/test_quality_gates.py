"""
Tests for quality gates implementation.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.models import ContentBlock, ContentType, Skill, Description, Body, References, SkillMetadata
from src.quality_gates import (
    QualityGates,
    CompressionPipeline,
    run_pipeline,
    FaithfulnessResult,
    FeedbackLoopResult,
    PipelineResult,
    GateResult
)


class TestQualityGates:
    """Tests for QualityGates class."""

    def _create_mock_llm_client(self):
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.model = "gpt-4o-mini"
        mock.client = MagicMock()
        return mock

    def _mock_response(self, content: str):
        """Create a mock response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        return mock_response

    def test_verify_faithfulness_passed(self):
        """Test faithfulness verification when all concepts are preserved."""
        mock_client = self._create_mock_llm_client()

        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "original_concepts": [
                    "Analyze market trends",
                    "Create positioning statement",
                    "Define target audience"
                ],
                "preserved_concepts": [
                    "Analyze market trends",
                    "Create positioning statement",
                    "Define target audience"
                ],
                "missing_concepts": [],
                "faithfulness_score": 1.0,
                "reasoning": "All core concepts are preserved in compressed rules"
            })
        )

        gates = QualityGates(llm_client=mock_client)

        result = gates.verify_faithfulness(
            original_body="To succeed in marketing, you must analyze market trends, create a positioning statement, and define your target audience.",
            compressed_core_rules="• Analyze market trends\n• Create positioning statement\n• Define target audience",
            skill_name="marketing-skill"
        )

        assert result.passed is True
        assert result.gate_result == GateResult.PASSED
        assert len(result.missing_concepts) == 0
        assert result.should_rollback is False

    def test_verify_faithfulness_failed(self):
        """Test faithfulness verification when concepts are missing."""
        mock_client = self._create_mock_llm_client()

        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "original_concepts": [
                    "Analyze market trends",
                    "Create positioning statement",
                    "Define target audience"
                ],
                "preserved_concepts": [
                    "Analyze market trends"
                ],
                "missing_concepts": [
                    "Create positioning statement",
                    "Define target audience"
                ],
                "faithfulness_score": 0.33,
                "reasoning": "Missing critical concepts about positioning and audience definition"
            })
        )

        gates = QualityGates(llm_client=mock_client)

        result = gates.verify_faithfulness(
            original_body="To succeed in marketing, you must analyze market trends, create a positioning statement, and define your target audience.",
            compressed_core_rules="• Analyze market trends",
            skill_name="marketing-skill"
        )

        assert result.passed is False
        assert result.gate_result == GateResult.ROLLBACK
        assert len(result.missing_concepts) == 2
        assert result.should_rollback is True

    def test_verify_faithfulness_empty_original(self):
        """Test faithfulness with empty original body."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(llm_client=mock_client)

        result = gates.verify_faithfulness(
            original_body="",
            compressed_core_rules="Some rules",
            skill_name="test-skill"
        )

        assert result.passed is True
        assert result.should_rollback is False

    def test_verify_faithfulness_empty_compressed(self):
        """Test faithfulness with empty compressed rules."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(llm_client=mock_client)

        result = gates.verify_faithfulness(
            original_body="Original content here",
            compressed_core_rules="",
            skill_name="test-skill"
        )

        assert result.passed is False
        assert result.should_rollback is True

    def test_feedback_loop_promotes_blocks(self):
        """Test feedback loop promotes relevant blocks to core_rule."""
        mock_client = self._create_mock_llm_client()

        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "relevant_blocks": [
                    {
                        "block_id": "chunk_2",
                        "relevant_to": ["API authentication failed"],
                        "reason": "This example shows correct authentication"
                    }
                ],
                "criteria_addressed": ["API authentication failed"]
            })
        )

        gates = QualityGates(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="chunk_1",
                content="Core rule content",
                content_type=ContentType.CORE_RULE,
                token_count=20
            ),
            ContentBlock(
                chunk_id="chunk_2",
                content="Example showing API authentication",
                content_type=ContentType.EXAMPLE,
                token_count=30
            ),
            ContentBlock(
                chunk_id="chunk_3",
                content="Background information",
                content_type=ContentType.BACKGROUND,
                token_count=25
            )
        ]

        result = gates.feedback_loop(
            blocks,
            failed_criteria=["API authentication failed"],
            skill_context="API integration skill"
        )

        assert result.promotion_count == 1
        assert "chunk_2" in result.promoted_blocks
        assert "API authentication failed" in result.feedback_addressed

        # Check that the block was promoted
        promoted_block = next(b for b in blocks if b.chunk_id == "chunk_2")
        assert promoted_block.content_type == ContentType.CORE_RULE

    def test_feedback_loop_no_relevant_blocks(self):
        """Test feedback loop when no blocks are relevant."""
        mock_client = self._create_mock_llm_client()

        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "relevant_blocks": [],
                "criteria_addressed": []
            })
        )

        gates = QualityGates(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="chunk_1",
                content="Core rule",
                content_type=ContentType.CORE_RULE,
                token_count=20
            )
        ]

        result = gates.feedback_loop(
            blocks,
            failed_criteria=["Unrelated failure"],
            skill_context="Test skill"
        )

        assert result.promotion_count == 0
        assert len(result.promoted_blocks) == 0

    def test_feedback_loop_empty_criteria(self):
        """Test feedback loop with empty failed criteria."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="chunk_1",
                content="Content",
                content_type=ContentType.EXAMPLE,
                token_count=10
            )
        ]

        result = gates.feedback_loop(blocks, failed_criteria=[])

        assert result.promotion_count == 0

    def test_feedback_loop_empty_blocks(self):
        """Test feedback loop with no blocks."""
        mock_client = self._create_mock_llm_client()
        gates = QualityGates(llm_client=mock_client)

        result = gates.feedback_loop([], failed_criteria=["Some failure"])

        assert result.promotion_count == 0


class TestCompressionPipeline:
    """Tests for CompressionPipeline class."""

    def _create_mock_llm_client(self):
        """Create a mock LLM client with full pipeline mocking."""
        mock = MagicMock()
        mock.model = "gpt-4o-mini"
        mock.client = MagicMock()
        return mock

    def _create_test_skill(self) -> Skill:
        """Create a test skill."""
        return Skill(
            name="test-skill",
            description=Description(
                original="A test skill for demonstration purposes with multiple capabilities.",
                original_token_count=15
            ),
            body=Body(
                original="# Core Rules\n\nRule 1: Do this.\nRule 2: Do that.\n\n## Example\n\n```python\ncode()\n```",
                original_token_count=50
            ),
            references=References(),
            metadata=SkillMetadata(category="test")
        )

    def test_run_pipeline_success(self):
        """Test successful pipeline execution."""
        mock_client = self._create_mock_llm_client()

        # Mock Stage 1 responses
        segment_response = json.dumps({
            "clauses": [
                {"clause_id": "1", "content": "A test skill"},
                {"clause_id": "2", "content": "for demonstration purposes"},
                {"clause_id": "3", "content": "with multiple capabilities"}
            ]
        })
        adversarial_response = json.dumps({"adversarial_skills": []})
        query_response = json.dumps({"query": "test query"})
        routing_response = json.dumps({"selected_skill": "test-skill", "reasoning": "test"})
        rewrite_response = json.dumps({"description": "Test skill for demonstration."})

        # Mock Stage 2 responses
        classify_response = json.dumps({
            "classifications": [
                {"chunk_id": "chunk_1", "content_type": "core_rule", "reasoning": "test"}
            ]
        })
        compress_response = json.dumps({
            "compressed_rules": ["• Rule 1: Do this", "• Rule 2: Do that"],
            "rules_merged": 2
        })
        faithfulness_response = json.dumps({
            "original_concepts": ["Rule 1", "Rule 2"],
            "preserved_concepts": ["Rule 1", "Rule 2"],
            "missing_concepts": [],
            "faithfulness_score": 1.0,
            "reasoning": "All preserved"
        })

        responses = iter([
            segment_response,
            adversarial_response,
            query_response,
            routing_response,
            rewrite_response,
            classify_response,
            compress_response,
            faithfulness_response
        ])

        def mock_create(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = next(responses)
            return mock_resp

        mock_client.client.chat.completions.create = mock_create

        # Mock classify_content_blocks for Stage 2
        from src.llm_client import BlockClassificationResult, BlockClassification
        mock_client.classify_content_blocks.return_value = BlockClassificationResult(
            classifications=[
                BlockClassification(chunk_id="chunk_1", content_type=ContentType.CORE_RULE)
            ]
        )

        pipeline = CompressionPipeline(
            llm_client=mock_client,
            enable_gate1=True,
            enable_gate2=False
        )

        skill = self._create_test_skill()
        result = pipeline.run_pipeline(skill)

        assert isinstance(result, PipelineResult)
        assert result.skill is not None
        assert isinstance(result.overall_compression_ratio, float)

    def test_run_pipeline_with_feedback_loop(self):
        """Test pipeline with Gate 2 feedback loop."""
        mock_client = self._create_mock_llm_client()

        # Simplified mocking - skip most pipeline steps
        segment_response = json.dumps({"clauses": [{"clause_id": "1", "content": "Test"}]})
        adversarial_response = json.dumps({"adversarial_skills": []})
        query_response = json.dumps({"query": "test"})
        routing_response = json.dumps({"selected_skill": "test-skill", "reasoning": "test"})
        rewrite_response = json.dumps({"description": "Test skill."})
        classify_response = json.dumps({"classifications": []})
        faithfulness_response = json.dumps({
            "original_concepts": [],
            "preserved_concepts": [],
            "missing_concepts": [],
            "faithfulness_score": 1.0,
            "reasoning": "OK"
        })
        feedback_response = json.dumps({
            "relevant_blocks": [],
            "criteria_addressed": []
        })

        responses = iter([
            segment_response,
            adversarial_response,
            query_response,
            routing_response,
            rewrite_response,
            classify_response,
            faithfulness_response,
            feedback_response
        ])

        def mock_create(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = next(responses)
            return mock_resp

        mock_client.client.chat.completions.create = mock_create

        from src.llm_client import BlockClassificationResult
        mock_client.classify_content_blocks.return_value = BlockClassificationResult(
            classifications=[]
        )

        pipeline = CompressionPipeline(
            llm_client=mock_client,
            enable_gate1=True,
            enable_gate2=True
        )

        skill = self._create_test_skill()
        result = pipeline.run_pipeline(
            skill,
            failed_criteria=["Task failed due to missing information"]
        )

        assert isinstance(result, PipelineResult)
        # Gate 2 should have been invoked

    def test_run_pipeline_with_rollback(self):
        """Test pipeline rollback when Gate 1 fails."""
        mock_client = self._create_mock_llm_client()

        # Track which calls we're responding to
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]

            # Stage 1 uses 2 calls when:
            # - segment returns 1 clause
            # - adversarial is empty
            # - single clause skips rewrite LLM call
            if call_count[0] <= 2:
                if call_count[0] == 1:
                    # segment: returns 1 clause
                    mock_resp.choices[0].message.content = json.dumps({"clauses": [{"clause_id": "1", "content": "Test"}]})
                elif call_count[0] == 2:
                    # adversarial: returns empty
                    mock_resp.choices[0].message.content = json.dumps({"adversarial_skills": []})
            elif call_count[0] == 3:
                # Stage 2 compress core
                mock_resp.choices[0].message.content = json.dumps({
                    "compressed_rules": ["• Rule 1 only"],
                    "rules_merged": 1
                })
            else:
                # Gate 1 faithfulness - return failure (call 4+)
                mock_resp.choices[0].message.content = json.dumps({
                    "original_concepts": ["Rule 1", "Rule 2"],
                    "preserved_concepts": ["Rule 1"],
                    "missing_concepts": ["Rule 2"],
                    "faithfulness_score": 0.5,
                    "reasoning": "Missing Rule 2"
                })
            return mock_resp

        mock_client.client.chat.completions.create = mock_create

        from src.llm_client import BlockClassificationResult, BlockClassification

        # Use side_effect to dynamically return classifications matching input chunk IDs
        def mock_classify(*args, **kwargs):
            content_chunks = kwargs.get('content_chunks', args[0] if args else [])
            classifications = [
                BlockClassification(chunk_id=chunk['chunk_id'], content_type=ContentType.CORE_RULE)
                for chunk in content_chunks
            ]
            return BlockClassificationResult(classifications=classifications)

        mock_client.classify_content_blocks = mock_classify

        pipeline = CompressionPipeline(
            llm_client=mock_client,
            enable_gate1=True,
            enable_gate2=False
        )

        skill = self._create_test_skill()
        result = pipeline.run_pipeline(
            skill,
            compress_core=True,
            dedup_examples=False,
            dedup_templates=False,
            summarize_background=False,
            dedup_references=False
        )

        assert isinstance(result, PipelineResult)
        # Rollback should have occurred due to missing concepts
        assert result.rollback_performed is True
        assert result.faithfulness_passed is False

    def test_run_pipeline_gates_disabled(self):
        """Test pipeline with quality gates disabled."""
        mock_client = self._create_mock_llm_client()

        segment_response = json.dumps({"clauses": [{"clause_id": "1", "content": "Test"}]})
        adversarial_response = json.dumps({"adversarial_skills": []})
        query_response = json.dumps({"query": "test"})
        routing_response = json.dumps({"selected_skill": "test-skill", "reasoning": "test"})
        rewrite_response = json.dumps({"description": "Test skill."})
        classify_response = json.dumps({"classifications": []})

        responses = iter([
            segment_response,
            adversarial_response,
            query_response,
            routing_response,
            rewrite_response,
            classify_response
        ])

        def mock_create(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = next(responses)
            return mock_resp

        mock_client.client.chat.completions.create = mock_create

        from src.llm_client import BlockClassificationResult
        mock_client.classify_content_blocks.return_value = BlockClassificationResult(
            classifications=[]
        )

        pipeline = CompressionPipeline(
            llm_client=mock_client,
            enable_gate1=False,
            enable_gate2=False
        )

        skill = self._create_test_skill()
        result = pipeline.run_pipeline(skill)

        assert isinstance(result, PipelineResult)
        # Faithfulness should pass by default when gate is disabled
        assert result.faithfulness_passed is True


class TestConvenienceFunction:
    """Tests for the run_pipeline convenience function."""

    def _create_mock_llm_client(self):
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.model = "gpt-4o-mini"
        mock.client = MagicMock()
        return mock

    def test_run_pipeline_function(self):
        """Test the convenience run_pipeline function."""
        mock_client = self._create_mock_llm_client()

        # Minimal mocking to avoid errors
        mock_client.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"clauses": []}'))]
        )
        mock_client.classify_content_blocks.return_value = MagicMock(classifications=[])

        skill = Skill(
            name="test",
            description=Description(original="Test", original_token_count=2),
            body=Body(original="Test body", original_token_count=5),
            references=References(),
            metadata=SkillMetadata()
        )

        result = run_pipeline(
            skill,
            llm_client=mock_client,
            enable_gate1=False,
            enable_gate2=False
        )

        assert isinstance(result, PipelineResult)
        assert result.skill.name == "test"
