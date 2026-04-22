"""
Unit tests for Stage 1 Router Optimizer.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.stage1_router import (
    Stage1Optimizer,
    SemanticClause,
    SegmentationResult,
    RoutingTestResult,
    CompressionResult
)
from src.models import Skill, Description, Body, SkillMetadata
from src.llm_client import SkillLLMClient


class TestStage1Optimizer:
    """Tests for Stage1Optimizer class."""

    def _create_mock_llm_client(self):
        """Create a mock LLM client for testing."""
        mock_client = Mock(spec=SkillLLMClient)
        mock_client.model = "gpt-4o-mini"
        mock_client.client = Mock()
        return mock_client

    def _create_test_skill(self) -> Skill:
        """Create a test skill for testing."""
        return Skill(
            name="marketing-strategy-pmm",
            description=Description(
                original="Product Marketing Manager skill for developing comprehensive "
                        "go-to-market strategies, product positioning, competitive analysis, "
                        "and launch planning for enterprise SaaS products.",
                original_token_count=0
            ),
            body=Body(
                original="# Marketing Strategy\n\nCore principles...",
                original_token_count=0
            ),
            metadata=SkillMetadata(
                category="marketing",
                domain="product-marketing",
                tags=["go-to-market", "positioning"]
            )
        )

    def test_segment_description(self):
        """Test description segmentation into semantic clauses."""
        mock_client = self._create_mock_llm_client()

        # Mock the chat completion response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "clauses": [
                {"clause_id": "clause_1", "content": "Product Marketing Manager skill"},
                {"clause_id": "clause_2", "content": "for developing go-to-market strategies"},
                {"clause_id": "clause_3", "content": "product positioning"},
                {"clause_id": "clause_4", "content": "competitive analysis"},
                {"clause_id": "clause_5", "content": "launch planning for enterprise SaaS"}
            ]
        })
        mock_client.client.chat.completions.create.return_value = mock_response

        optimizer = Stage1Optimizer(llm_client=mock_client)
        clauses = optimizer.segment_description("Test description")

        assert len(clauses) == 5
        assert clauses[0].clause_id == "clause_1"
        assert clauses[0].content == "Product Marketing Manager skill"
        mock_client.client.chat.completions.create.assert_called_once()

    def test_generate_adversarial_skill(self):
        """Test generation of adversarial skills."""
        mock_client = self._create_mock_llm_client()
        skill = self._create_test_skill()

        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "adversarial_skills": [
                {
                    "name": "content-marketing-manager",
                    "description": "Content Marketing Manager skill for creating blog posts and content strategy"
                },
                {
                    "name": "demand-gen-specialist",
                    "description": "Demand Generation specialist for lead generation campaigns"
                }
            ]
        })
        mock_client.client.chat.completions.create.return_value = mock_response

        optimizer = Stage1Optimizer(llm_client=mock_client)
        adversarial = optimizer.generate_adversarial_skill(skill, num_candidates=2)

        assert len(adversarial) == 2
        assert adversarial[0]["name"] == "content-marketing-manager"
        assert "content" in adversarial[0]["description"].lower()

    def test_test_routing_success(self):
        """Test routing oracle with successful routing."""
        mock_client = self._create_mock_llm_client()
        skill = self._create_test_skill()

        # Mock successful routing
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "selected_skill": "marketing-strategy-pmm",
            "reasoning": "The query asks for go-to-market strategy which matches this skill"
        })
        mock_client.client.chat.completions.create.return_value = mock_response

        optimizer = Stage1Optimizer(llm_client=mock_client)

        adversarial = [
            {"name": "content-marketing", "description": "Content creation skill"}
        ]

        result = optimizer.test_routing(
            compressed_description="Go-to-market strategy skill",
            target_skill=skill,
            adversarial_skills=adversarial,
            query="Help me create a go-to-market strategy"
        )

        assert result.success is True
        assert result.selected_skill == "marketing-strategy-pmm"

    def test_test_routing_failure(self):
        """Test routing oracle with failed routing."""
        mock_client = self._create_mock_llm_client()
        skill = self._create_test_skill()

        # Mock failed routing (selected wrong skill)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "selected_skill": "content-marketing",
            "reasoning": "The query matches content marketing"
        })
        mock_client.client.chat.completions.create.return_value = mock_response

        optimizer = Stage1Optimizer(llm_client=mock_client)

        adversarial = [
            {"name": "content-marketing", "description": "Content creation skill"}
        ]

        result = optimizer.test_routing(
            compressed_description="Marketing skill",  # Vague description
            target_skill=skill,
            adversarial_skills=adversarial,
            query="Help me with marketing"
        )

        assert result.success is False
        assert result.selected_skill == "content-marketing"

    def test_ddmin_algorithm(self):
        """Test DDMIN algorithm for finding minimal subset."""
        mock_client = self._create_mock_llm_client()
        optimizer = Stage1Optimizer(llm_client=mock_client)

        # Create test clauses
        clauses = [
            SemanticClause(clause_id="1", content="Clause 1"),
            SemanticClause(clause_id="2", content="Clause 2"),
            SemanticClause(clause_id="3", content="Clause 3"),
            SemanticClause(clause_id="4", content="Clause 4"),
        ]

        # Test function: passes with clauses 1 and 3
        def test_func(subset):
            clause_ids = {c.clause_id for c in subset}
            # Only clauses 1 and 3 are needed
            return "1" in clause_ids and "3" in clause_ids

        result = optimizer.ddmin(clauses, test_func)

        # Should find minimal set {1, 3}
        result_ids = {c.clause_id for c in result}
        assert "1" in result_ids
        assert "3" in result_ids
        assert len(result) <= 2

    def test_ddmin_returns_single_clause(self):
        """Test DDMIN with single clause returns it unchanged."""
        mock_client = self._create_mock_llm_client()
        optimizer = Stage1Optimizer(llm_client=mock_client)

        clauses = [SemanticClause(clause_id="1", content="Single clause")]

        def test_func(subset):
            return True

        result = optimizer.ddmin(clauses, test_func)
        assert len(result) == 1
        assert result[0].clause_id == "1"

    def test_rewrite_and_polish(self):
        """Test rewriting clauses into polished description."""
        mock_client = self._create_mock_llm_client()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "description": "Product Marketing Manager skill for developing go-to-market strategies and product positioning."
        })
        mock_client.client.chat.completions.create.return_value = mock_response

        optimizer = Stage1Optimizer(llm_client=mock_client)

        clauses = [
            SemanticClause(clause_id="1", content="Product Marketing Manager skill"),
            SemanticClause(clause_id="2", content="for developing go-to-market strategies"),
            SemanticClause(clause_id="3", content="product positioning")
        ]

        result = optimizer.rewrite_and_polish(clauses)

        assert "go-to-market" in result
        assert "positioning" in result
        mock_client.client.chat.completions.create.assert_called_once()

    def test_rewrite_single_clause(self):
        """Test rewriting a single clause returns it as-is."""
        mock_client = self._create_mock_llm_client()
        optimizer = Stage1Optimizer(llm_client=mock_client)

        clauses = [SemanticClause(clause_id="1", content="Single clause content")]

        result = optimizer.rewrite_and_polish(clauses)

        # Single clause should be returned as-is
        assert result == "Single clause content"

    def test_compress_description_full_pipeline(self):
        """Test full compression pipeline with mocked components."""
        mock_client = self._create_mock_llm_client()
        skill = self._create_test_skill()

        # Mock responses for each step
        responses = [
            # 1. Segmentation
            json.dumps({
                "clauses": [
                    {"clause_id": "1", "content": "Product Marketing Manager skill"},
                    {"clause_id": "2", "content": "for go-to-market strategies"},
                    {"clause_id": "3", "content": "product positioning"},
                    {"clause_id": "4", "content": "competitive analysis"},
                    {"clause_id": "5", "content": "launch planning"}
                ]
            }),
            # 2. Generate adversarial
            json.dumps({
                "adversarial_skills": [
                    {"name": "content-marketing", "description": "Content marketing skill"}
                ]
            }),
            # 3. Generate query
            json.dumps({
                "query": "Help me develop a go-to-market strategy"
            }),
            # 4. Routing test (various calls during DDMIN)
            json.dumps({
                "selected_skill": "marketing-strategy-pmm",
                "reasoning": "Matches the query"
            }),
            # 5. Rewrite
            json.dumps({
                "description": "Product Marketing Manager skill for go-to-market strategies and product positioning."
            })
        ]

        response_idx = [0]

        def mock_create(*args, **kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = responses[response_idx[0]]
            response_idx[0] += 1
            return mock_response

        mock_client.client.chat.completions.create = mock_create

        optimizer = Stage1Optimizer(llm_client=mock_client)
        result = optimizer.compress_description(skill, use_oracle_validation=False)

        assert isinstance(result, CompressionResult)
        assert result.original_description == skill.description.original
        assert result.compressed_description != ""
        assert result.compression_ratio >= 0

    def test_compress_description_empty_skill(self):
        """Test compression with empty description."""
        mock_client = self._create_mock_llm_client()

        skill = Skill(
            name="empty-skill",
            description=Description(original="", original_token_count=0),
            body=Body(original="", original_token_count=0),
            metadata=SkillMetadata()
        )

        optimizer = Stage1Optimizer(llm_client=mock_client)
        result = optimizer.compress_description(skill)

        assert result.original_description == ""
        assert result.compressed_description == ""
        assert result.compression_ratio == 0.0

    def test_compress_skill_updates_skill_object(self):
        """Test that compress_skill updates the Skill object."""
        mock_client = self._create_mock_llm_client()
        skill = self._create_test_skill()

        responses = [
            json.dumps({
                "clauses": [
                    {"clause_id": "1", "content": "Marketing strategy skill"}
                ]
            }),
            json.dumps({"adversarial_skills": []}),
            json.dumps({"query": "Test query"}),
            json.dumps({"selected_skill": "marketing-strategy-pmm", "reasoning": "test"}),
            json.dumps({"description": "Marketing strategy skill for go-to-market planning."})
        ]

        response_idx = [0]

        def mock_create(*args, **kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = responses[response_idx[0]]
            response_idx[0] += 1
            return mock_response

        mock_client.client.chat.completions.create = mock_create

        optimizer = Stage1Optimizer(llm_client=mock_client)
        updated_skill = optimizer.compress_skill(skill, use_oracle_validation=False)

        assert updated_skill.description.compressed is not None
        assert updated_skill.description.compressed_token_count > 0
