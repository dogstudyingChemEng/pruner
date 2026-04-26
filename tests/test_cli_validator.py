"""
Unit tests for CLI Validator (Stage 1 Phase 2 Real CLI Validation).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.cli_validator import (
    StreamEventType,
    StreamEvent,
    TriggerValidationResult,
    StreamEventParser,
    RealTriggerValidator
)


class TestStreamEventParser:
    """Tests for StreamEventParser class."""

    def test_parse_single_json_event(self):
        """Test parsing a single JSON event."""
        parser = StreamEventParser()
        raw_output = '{"type": "assistant", "message": {"text": "Hello"}}'

        events = parser.parse_stream_events(raw_output)

        assert len(events) == 1
        assert events[0].event_type == StreamEventType.ASSISTANT
        assert "Hello" in events[0].content

    def test_parse_multiple_json_events(self):
        """Test parsing multiple JSON events."""
        parser = StreamEventParser()
        raw_output = """
{"type": "init", "data": {}}
{"type": "assistant", "message": {"text": "Response"}}
{"type": "result", "content": "Done"}
"""

        events = parser.parse_stream_events(raw_output)

        assert len(events) == 3

    def test_parse_tool_use_event(self):
        """Test parsing tool use event."""
        parser = StreamEventParser()
        raw_output = '{"type": "tool_use", "name": "read_file", "input": {"path": "test.md"}}'

        events = parser.parse_stream_events(raw_output)

        assert len(events) == 1
        assert events[0].event_type == StreamEventType.TOOL_USE
        assert len(events[0].tool_calls) > 0

    def test_extract_skill_trigger(self):
        """Test extracting skill trigger from events."""
        parser = StreamEventParser()
        events = [
            StreamEvent(
                event_type=StreamEventType.TOOL_USE,
                content="",
                tool_calls=[{"name": "load_skill", "input": {"skill": "marketing-strategy"}}]
            )
        ]

        triggered_skill = parser.extract_skill_trigger(events)

        assert triggered_skill == "marketing-strategy"

    def test_detect_skill_invocation_positive(self):
        """Test detecting skill invocation when present."""
        parser = StreamEventParser()
        events = [
            StreamEvent(
                event_type=StreamEventType.ASSISTANT,
                content="Loading skill: marketing-strategy-pmm",
                tool_calls=[]
            )
        ]

        detected = parser.detect_skill_invocation_patterns(events, "marketing-strategy-pmm")

        assert detected is True

    def test_detect_skill_invocation_negative(self):
        """Test detecting skill invocation when not present."""
        parser = StreamEventParser()
        events = [
            StreamEvent(
                event_type=StreamEventType.ASSISTANT,
                content="Loading skill: different-skill",
                tool_calls=[]
            )
        ]

        detected = parser.detect_skill_invocation_patterns(events, "marketing-strategy-pmm")

        assert detected is False


class TestRealTriggerValidator:
    """Tests for RealTriggerValidator class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        validator = RealTriggerValidator()

        assert validator.cli_path == "claude"
        assert validator.timeout == 60
        assert validator.parser is not None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        validator = RealTriggerValidator(
            cli_path="/custom/path/claude",
            timeout=30,
            skill_library_path="/custom/skills"
        )

        assert validator.cli_path == "/custom/path/claude"
        assert validator.timeout == 30
        assert validator.skill_library_path == "/custom/skills"

    @patch('src.cli_validator.subprocess.run')
    def test_validate_trigger_success(self, mock_run):
        """Test successful trigger validation."""
        mock_run.return_value = MagicMock(
            stdout='{"type": "assistant", "content": "Using skill marketing-strategy"}',
            stderr='',
            returncode=0
        )

        validator = RealTriggerValidator()
        result = validator.validate_trigger(
            skill_name="marketing-strategy",
            compressed_description="Product marketing skill",
            query="Help with go-to-market strategy"
        )

        assert isinstance(result, TriggerValidationResult)

    @patch('src.cli_validator.subprocess.run')
    def test_validate_trigger_timeout(self, mock_run):
        """Test trigger validation timeout handling."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("claude", timeout=60)

        validator = RealTriggerValidator(timeout=60)
        result = validator.validate_trigger(
            skill_name="test-skill",
            compressed_description="Test",
            query="Test query"
        )

        assert result.triggered is False
        assert "timeout" in result.error.lower()

    def test_calculate_trigger_rate(self):
        """Test calculating trigger rate from results."""
        validator = RealTriggerValidator()
        results = [
            TriggerValidationResult(triggered=True),
            TriggerValidationResult(triggered=True),
            TriggerValidationResult(triggered=False),
            TriggerValidationResult(triggered=True)
        ]

        rate = validator.calculate_trigger_rate(results)

        assert rate == 0.75  # 3/4 passed


class TestTriggerValidationResult:
    """Tests for TriggerValidationResult data model."""

    def test_result_success(self):
        """Test creating successful validation result."""
        result = TriggerValidationResult(
            triggered=True,
            triggered_skill="test-skill",
            tool_calls_made=["read_file", "load_skill"],
            execution_time=2.5
        )

        assert result.triggered is True
        assert result.triggered_skill == "test-skill"
        assert len(result.tool_calls_made) == 2
        assert result.execution_time == 2.5

    def test_result_failure(self):
        """Test creating failed validation result."""
        result = TriggerValidationResult(
            triggered=False,
            error="CLI not available"
        )

        assert result.triggered is False
        assert result.error == "CLI not available"