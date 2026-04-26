"""
Unit tests for Structured Description Generation.
"""

import pytest
from unittest.mock import Mock, patch

from src.models import (
    RoutingSignal,
    StructuredDescription,
    Skill, Description, Body, References, SkillMetadata
)
from src.stage1_router import Stage1Optimizer


class TestRoutingSignal:
    """Tests for RoutingSignal data model."""

    def test_signal_creation(self):
        """Test creating routing signal."""
        signal = RoutingSignal(
            signal_type="primary_capability",
            content="Handles JWT authentication for API security",
            token_count=25
        )

        assert signal.signal_type == "primary_capability"
        assert "JWT" in signal.signal_type or "JWT" in signal.content
        assert signal.token_count == 25

    def test_signal_types(self):
        """Test different signal types."""
        primary = RoutingSignal(
            signal_type="primary_capability",
            content="Does X",
            token_count=20
        )
        trigger = RoutingSignal(
            signal_type="trigger_condition",
            content="When Y happens",
            token_count=20
        )
        identifier = RoutingSignal(
            signal_type="unique_identifier",
            content="Uses library Z",
            token_count=20
        )

        assert primary.signal_type == "primary_capability"
        assert trigger.signal_type == "trigger_condition"
        assert identifier.signal_type == "unique_identifier"


class TestStructuredDescription:
    """Tests for StructuredDescription data model."""

    def test_description_creation(self):
        """Test creating structured description."""
        primary = RoutingSignal(
            signal_type="primary_capability",
            content="Product marketing skill for go-to-market",
            token_count=25
        )
        trigger = RoutingSignal(
            signal_type="trigger_condition",
            content="When launching new products",
            token_count=20
        )
        identifiers = [
            RoutingSignal(
                signal_type="unique_identifier",
                content="Uses April Dunford methodology",
                token_count=22
            )
        ]

        structured = StructuredDescription(
            primary_capability=primary,
            trigger_condition=trigger,
            unique_identifiers=identifiers,
            combined_description="Product marketing for GTM. Uses April Dunford methodology.",
            total_tokens=67
        )

        assert structured.primary_capability.content == "Product marketing skill for go-to-market"
        assert structured.trigger_condition.content == "When launching new products"
        assert len(structured.unique_identifiers) == 1
        assert structured.total_tokens == 67

    def test_description_with_multiple_identifiers(self):
        """Test description with multiple unique identifiers."""
        structured = StructuredDescription(
            primary_capability=RoutingSignal(
                signal_type="primary_capability",
                content="Does X",
                token_count=20
            ),
            trigger_condition=RoutingSignal(
                signal_type="trigger_condition",
                content="When Y",
                token_count=15
            ),
            unique_identifiers=[
                RoutingSignal(signal_type="unique_identifier", content="Lib A", token_count=10),
                RoutingSignal(signal_type="unique_identifier", content="Lib B", token_count=10),
                RoutingSignal(signal_type="unique_identifier", content="Lib C", token_count=10)
            ],
            combined_description="Combined",
            total_tokens=65
        )

        assert len(structured.unique_identifiers) == 3


class TestDescriptionModel:
    """Tests for Description model with structured field."""

    def test_description_with_structured(self):
        """Test Description with structured field."""
        structured = StructuredDescription(
            primary_capability=RoutingSignal(
                signal_type="primary_capability",
                content="Test",
                token_count=20
            ),
            trigger_condition=RoutingSignal(
                signal_type="trigger_condition",
                content="Test",
                token_count=20
            ),
            unique_identifiers=[],
            combined_description="Test description",
            total_tokens=40
        )

        desc = Description(
            original="Original description",
            structured=structured,
            original_token_count=40
        )

        assert desc.structured is not None
        assert desc.structured.combined_description == "Test description"

    def test_description_to_legacy_format(self):
        """Test converting to legacy format."""
        structured = StructuredDescription(
            primary_capability=RoutingSignal(
                signal_type="primary_capability",
                content="Primary",
                token_count=20
            ),
            trigger_condition=RoutingSignal(
                signal_type="trigger_condition",
                content="Trigger",
                token_count=20
            ),
            unique_identifiers=[],
            combined_description="Structured combined",
            total_tokens=40
        )

        desc = Description(
            original="Original description",
            structured=structured,
            original_token_count=40
        )

        legacy = desc.to_legacy_format()

        assert legacy == "Structured combined"

    def test_description_to_legacy_without_structured(self):
        """Test legacy format without structured field."""
        desc = Description(
            original="Original description",
            original_token_count=30
        )

        legacy = desc.to_legacy_format()

        assert legacy == "Original description"


class TestStage1OptimizerStructuredDescription:
    """Tests for Stage1Optimizer structured description methods."""

    def _create_mock_llm_client(self):
        """Create mock LLM client."""
        mock_client = Mock()
        mock_client.model = "gpt-4o-mini"
        mock_client.client = Mock()

        response_content = json.dumps({
            "primary_capability": {
                "content": "Product marketing skill",
                "token_count": 25
            },
            "trigger_condition": {
                "content": "When launching products",
                "token_count": 20
            },
            "unique_identifiers": [
                {
                    "signal_type": "unique_identifier",
                    "content": "April Dunford methodology",
                    "token_count": 22
                }
            ],
            "combined_description": "Product marketing for GTM. Uses April Dunford methodology."
        })

        mock_client.client.chat.completions.create = Mock(return_value=Mock(
            choices=[Mock(message=Mock(content=response_content))]
        ))
        return mock_client

    def _create_test_skill(self) -> Skill:
        """Create test skill."""
        return Skill(
            name="marketing-strategy-pmm",
            description=Description(
                original="",
                original_token_count=0
            ),
            body=Body(
                original="# Marketing Strategy\n\nProduct marketing methodology...",
                original_token_count=100
            ),
            metadata=SkillMetadata(category="marketing"),
            references=References()
        )

    @patch('src.stage1_router.Stage1Optimizer.__init__', lambda self, client, **kwargs: None)
    def test_validate_routing_signals_valid(self):
        """Test validating valid routing signals."""
        import json

        mock_client = self._create_mock_llm_client()
        optimizer = Stage1Optimizer.__new__(Stage1Optimizer)
        optimizer.llm_client = mock_client

        structured = StructuredDescription(
            primary_capability=RoutingSignal(
                signal_type="primary_capability",
                content="Test capability",
                token_count=25
            ),
            trigger_condition=RoutingSignal(
                signal_type="trigger_condition",
                content="Test trigger",
                token_count=30
            ),
            unique_identifiers=[
                RoutingSignal(signal_type="unique_identifier", content="Lib", token_count=20)
            ],
            combined_description="Test",
            total_tokens=75
        )

        is_valid = optimizer.validate_routing_signals(structured)

        assert is_valid is True

    @patch('src.stage1_router.Stage1Optimizer.__init__', lambda self, client, **kwargs: None)
    def test_validate_routing_signals_invalid_tokens(self):
        """Test validating signals with invalid token counts."""
        mock_client = self._create_mock_llm_client()
        optimizer = Stage1Optimizer.__new__(Stage1Optimizer)
        optimizer.llm_client = mock_client

        structured = StructuredDescription(
            primary_capability=RoutingSignal(
                signal_type="primary_capability",
                content="Too short",
                token_count=5  # Below minimum 20
            ),
            trigger_condition=RoutingSignal(
                signal_type="trigger_condition",
                content="Test",
                token_count=30
            ),
            unique_identifiers=[],
            combined_description="Test",
            total_tokens=35
        )

        is_valid = optimizer.validate_routing_signals(structured)

        assert is_valid is False


import json  # Add import at top of file