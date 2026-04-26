"""
CLI Validator for Stage 1 Phase 2 Real-Environment Validation.

Per SkillReducer paper Algorithm 1, Phase 2 validates compressed descriptions
by testing whether the actual agent runtime triggers the skill through
Claude Code CLI, parsing stream events to detect skill invocation.

This module implements:
- RealTriggerValidator: Execute CLI subprocess and validate triggers
- StreamEventParser: Parse Claude Code JSON stream events
- TriggerValidationResult: Data model for validation results
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field


class StreamEventType(str, Enum):
    """Types of events in Claude Code CLI stream output."""

    INIT = "init"
    ASSISTANT = "assistant"
    RESULT = "result"
    ERROR = "error"
    SYSTEM = "system"
    TOOL_USE = "tool_use"


class StreamEvent(BaseModel):
    """A single event from Claude Code CLI stream output."""

    event_type: StreamEventType = Field(
        description="Type of stream event"
    )
    content: str = Field(
        default="",
        description="Event content text"
    )
    tool_calls: list[dict] = Field(
        default_factory=list,
        description="Tool calls made in this event"
    )
    timestamp: float = Field(
        default=0.0,
        description="Event timestamp"
    )
    raw_data: dict = Field(
        default_factory=dict,
        description="Raw JSON data for the event"
    )


class TriggerValidationResult(BaseModel):
    """Result of real CLI trigger validation."""

    triggered: bool = Field(
        description="Whether the target skill was triggered"
    )
    triggered_skill: Optional[str] = Field(
        default=None,
        description="Name of the skill that was triggered"
    )
    tool_calls_made: list[str] = Field(
        default_factory=list,
        description="Tool calls made during execution"
    )
    stream_events: list[StreamEvent] = Field(
        default_factory=list,
        description="Parsed stream events from CLI output"
    )
    execution_time: float = Field(
        default=0.0,
        description="Execution time in seconds"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if validation failed"
    )
    cli_output: str = Field(
        default="",
        description="Raw CLI output for debugging"
    )


class StreamEventParser:
    """
    Parse Claude Code CLI stream output for skill invocation events.

    Claude Code CLI outputs JSON stream events. This parser extracts
    events to detect skill invocation and tool usage.
    """

    def parse_stream_events(self, raw_output: str) -> list[StreamEvent]:
        """
        Parse raw CLI output into structured stream events.

        Args:
            raw_output: Raw output from Claude Code CLI.

        Returns:
            List of parsed StreamEvent objects.
        """
        events = []

        # Claude Code CLI outputs events as JSON lines
        # Format varies but typically includes:
        # - {"type": "init", ...}
        # - {"type": "assistant", "message": {...}}
        # - {"type": "result", ...}

        lines = raw_output.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                event = self._parse_single_event(data)
                if event:
                    events.append(event)
            except json.JSONDecodeError:
                # Non-JSON line, try to parse as text event
                if line.strip():
                    events.append(StreamEvent(
                        event_type=StreamEventType.SYSTEM,
                        content=line,
                        timestamp=time.time()
                    ))

        return events

    def _parse_single_event(self, data: dict) -> Optional[StreamEvent]:
        """Parse a single JSON event dict into StreamEvent."""
        event_type_str = data.get("type", data.get("event", "system"))

        # Map common event type strings to enum
        type_mapping = {
            "init": StreamEventType.INIT,
            "assistant": StreamEventType.ASSISTANT,
            "result": StreamEventType.RESULT,
            "error": StreamEventType.ERROR,
            "system": StreamEventType.SYSTEM,
            "tool_use": StreamEventType.TOOL_USE,
            "message": StreamEventType.ASSISTANT,
        }

        event_type = type_mapping.get(event_type_str.lower(), StreamEventType.SYSTEM)

        # Extract content based on event type
        content = ""
        tool_calls = []

        if event_type == StreamEventType.ASSISTANT:
            message = data.get("message", data.get("content", {}))
            if isinstance(message, dict):
                content = message.get("text", "")
                tool_calls = message.get("tool_calls", [])
            elif isinstance(message, str):
                content = message

        elif event_type == StreamEventType.TOOL_USE:
            content = data.get("name", "")
            tool_calls = [data]

        elif event_type in (StreamEventType.INIT, StreamEventType.SYSTEM):
            content = json.dumps(data)

        elif event_type == StreamEventType.ERROR:
            content = data.get("message", data.get("error", ""))

        elif event_type == StreamEventType.RESULT:
            content = data.get("result", data.get("content", ""))

        return StreamEvent(
            event_type=event_type,
            content=content,
            tool_calls=tool_calls,
            timestamp=time.time(),
            raw_data=data
        )

    def extract_skill_trigger(self, events: list[StreamEvent]) -> Optional[str]:
        """
        Extract which skill was triggered from stream events.

        Looks for skill invocation patterns in events:
        - Tool call to skill loading
        - System message mentioning skill
        - Assistant message referencing skill

        Args:
            events: List of parsed stream events.

        Returns:
            Name of triggered skill if detected, None otherwise.
        """
        for event in events:
            # Check tool calls for skill invocation
            for tool_call in event.tool_calls:
                tool_name = tool_call.get("name", tool_call.get("tool_name", ""))
                if "skill" in tool_name.lower() or "load" in tool_name.lower():
                    # Extract skill name from tool input
                    input_data = tool_call.get("input", tool_call.get("arguments", {}))
                    if isinstance(input_data, dict):
                        skill_name = input_data.get("skill", input_data.get("skill_name", ""))
                        if skill_name:
                            return skill_name

            # Check content for skill mentions
            content = event.content.lower()
            if "skill:" in content or "loading skill" in content:
                # Try to extract skill name after "skill:" or "loading skill:"
                # Pattern: "skill: <name>" or "loading skill: <name>"
                original = event.content
                if "skill:" in original:
                    # Extract text after "skill:"
                    parts = original.split("skill:", 1)
                    if len(parts) > 1:
                        potential_name = parts[1].strip().split()[0] if parts[1].strip() else ""
                        if potential_name and len(potential_name) > 3:
                            return potential_name.strip()
                elif "loading skill" in original:
                    # Extract text after "loading skill" or "loading skill:"
                    parts = original.lower().split("loading skill", 1)
                    if len(parts) > 1:
                        remainder = parts[1].strip()
                        # Remove colon if present
                        if remainder.startswith(":"):
                            remainder = remainder[1:].strip()
                        potential_name = remainder.split()[0] if remainder else ""
                        if potential_name and len(potential_name) > 3:
                            return potential_name.strip()

        return None

    def detect_skill_invocation_patterns(self, events: list[StreamEvent], target_skill: str) -> bool:
        """
        Detect if target skill was invoked from events.

        Args:
            events: List of parsed stream events.
            target_skill: Expected skill name.

        Returns:
            True if target skill invocation detected.
        """
        triggered_skill = self.extract_skill_trigger(events)

        if triggered_skill:
            # Check if triggered skill matches target (case-insensitive)
            return triggered_skill.lower() == target_skill.lower()

        # Fallback: search for skill name in any content
        for event in events:
            if target_skill.lower() in event.content.lower():
                return True

        return False


class RealTriggerValidator:
    """
    Validate skill triggers via real Claude Code CLI.

    Per SkillReducer paper Algorithm 1, REALTRIGGER function:
    1. Deploy skill with compressed description
    2. Issue query through Claude Code CLI
    3. Parse stream events for skill invocation
    4. Return whether skill was triggered
    """

    def __init__(
        self,
        cli_path: str = "claude",
        timeout: int = 60,
        skill_library_path: str = "Claude-Skills"
    ):
        """
        Initialize real trigger validator.

        Args:
            cli_path: Path to Claude Code CLI executable.
            timeout: Timeout in seconds for CLI calls.
            skill_library_path: Path to skill library directory.
        """
        self.cli_path = cli_path
        self.timeout = timeout
        self.skill_library_path = skill_library_path
        self.parser = StreamEventParser()

    def validate_trigger(
        self,
        skill_name: str,
        compressed_description: str,
        query: str,
        skill_content: Optional[str] = None
    ) -> TriggerValidationResult:
        """
        Validate if compressed description triggers skill via real CLI.

        Args:
            skill_name: Target skill name.
            compressed_description: Compressed description to test.
            query: Test query to send to CLI.
            skill_content: Optional full skill body content for testing.

        Returns:
            TriggerValidationResult with trigger status and details.
        """
        start_time = time.time()

        try:
            # Create temporary skill file with compressed description
            temp_skill_path = self._create_temp_skill(
                skill_name=skill_name,
                description=compressed_description,
                body_content=skill_content or ""
            )

            # Run CLI with query
            cli_output, exit_code = self._run_cli_subprocess(
                skill_path=temp_skill_path,
                query=query
            )

            # Parse stream events
            events = self.parser.parse_stream_events(cli_output)

            # Detect skill invocation
            triggered = self.parser.detect_skill_invocation_patterns(events, skill_name)
            triggered_skill = self.parser.extract_skill_trigger(events)

            # Extract tool calls
            tool_calls_made = []
            for event in events:
                for tc in event.tool_calls:
                    tool_name = tc.get("name", tc.get("tool_name", ""))
                    if tool_name:
                        tool_calls_made.append(tool_name)

            execution_time = time.time() - start_time

            # Cleanup temp file
            self._cleanup_temp_skill(temp_skill_path)

            return TriggerValidationResult(
                triggered=triggered,
                triggered_skill=triggered_skill,
                tool_calls_made=tool_calls_made,
                stream_events=events,
                execution_time=execution_time,
                cli_output=cli_output
            )

        except subprocess.TimeoutExpired:
            return TriggerValidationResult(
                triggered=False,
                error=f"CLI timeout after {self.timeout} seconds",
                execution_time=self.timeout
            )

        except Exception as e:
            return TriggerValidationResult(
                triggered=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _create_temp_skill(
        self,
        skill_name: str,
        description: str,
        body_content: str
    ) -> Path:
        """
        Create temporary skill file for testing.

        Args:
            skill_name: Skill name.
            description: Description text.
            body_content: Body content.

        Returns:
            Path to temporary skill directory.
        """
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="skill_test_")
        skill_dir = Path(temp_dir) / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Create SKILL.md with compressed description
        skill_md_content = f"""---
name: {skill_name}
description: |
  {description}
---

{body_content}
"""

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill_md_content)

        return skill_dir

    def _run_cli_subprocess(
        self,
        skill_path: Path,
        query: str
    ) -> tuple[str, int]:
        """
        Run Claude Code CLI subprocess.

        Args:
            skill_path: Path to skill directory.
            query: Query to send.

        Returns:
            Tuple of (output, exit_code).
        """
        # Build CLI command
        # Note: Actual command depends on Claude Code CLI interface
        # This is a placeholder for the real implementation
        command = [
            self.cli_path,
            "code",
            "--skill", str(skill_path),
            "--query", query,
            "--json"  # Request JSON output for easier parsing
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Combine stdout and stderr for full output
            output = result.stdout + "\n" + result.stderr
            return output, result.returncode

        except subprocess.TimeoutExpired:
            raise

    def _cleanup_temp_skill(self, skill_path: Path):
        """Remove temporary skill directory."""
        try:
            import shutil
            if skill_path.parent.exists():
                shutil.rmtree(skill_path.parent)
        except Exception:
            pass  # Cleanup failure is non-critical

    def batch_validate(
        self,
        skill_name: str,
        compressed_description: str,
        queries: list[str],
        skill_content: Optional[str] = None
    ) -> list[TriggerValidationResult]:
        """
        Validate multiple queries for same skill.

        Args:
            skill_name: Target skill name.
            compressed_description: Compressed description.
            queries: List of test queries.
            skill_content: Optional skill body content.

        Returns:
            List of TriggerValidationResult for each query.
        """
        results = []
        for query in queries:
            result = self.validate_trigger(
                skill_name=skill_name,
                compressed_description=compressed_description,
                query=query,
                skill_content=skill_content
            )
            results.append(result)

        return results

    def calculate_trigger_rate(
        self,
        results: list[TriggerValidationResult]
    ) -> float:
        """
        Calculate trigger rate from batch results.

        Args:
            results: List of validation results.

        Returns:
            Fraction of queries that triggered the skill.
        """
        if not results:
            return 0.0

        triggered_count = sum(1 for r in results if r.triggered)
        return triggered_count / len(results)


def create_mock_validator() -> RealTriggerValidator:
    """
    Create a mock validator for testing without real CLI.

    Returns:
        RealTriggerValidator instance that simulates responses.
    """
    # This would be used in tests to mock CLI behavior
    return RealTriggerValidator(
        cli_path="mock_claude",
        timeout=10
    )