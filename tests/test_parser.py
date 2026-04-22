"""
Tests for the skill file parser.
"""

from pathlib import Path

import pytest

from src.models import Skill
from src.parser import parse_skill_file


class TestParseSkillFile:
    """Tests for parse_skill_file function."""

    def test_parse_marketing_strategy_pmm(self):
        """Test parsing the marketing-strategy-pmm skill file."""
        # Path to the skill file
        skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")

        # Parse the skill file
        skill = parse_skill_file(skill_path)

        # Verify it returns a Skill object
        assert isinstance(skill, Skill)

        # Verify name is extracted correctly
        assert skill.name == "marketing-strategy-pmm"

        # Verify description is extracted
        assert skill.description.original is not None
        assert len(skill.description.original) > 0
        assert "Product marketing" in skill.description.original

        # Verify body is extracted and not empty
        assert skill.body.original is not None
        assert len(skill.body.original) > 0

        # Print parsed information
        print(f"\n{'='*60}")
        print(f"Parsed Skill: {skill.name}")
        print(f"{'='*60}")
        print(f"Description length: {len(skill.description.original)} characters")
        print(f"Description content:\n{skill.description.original[:200]}...")
        print(f"\nBody length: {len(skill.body.original)} characters")
        print(f"Body preview (first 200 chars):\n{skill.body.original[:200]}...")
        print(f"{'='*60}\n")

    def test_parse_skill_metadata(self):
        """Test that metadata is correctly extracted."""
        skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
        skill = parse_skill_file(skill_path)

        # Verify metadata fields
        assert skill.metadata.license == "MIT + Commons Clause"
        assert skill.metadata.category == "marketing"
        assert skill.metadata.domain == "product-marketing"
        assert skill.metadata.version == "1.0.0"
        assert skill.metadata.author == "borghei"

        # Verify tags
        assert isinstance(skill.metadata.tags, list)
        assert "product-marketing" in skill.metadata.tags
        assert "positioning" in skill.metadata.tags

    def test_parse_skill_with_references(self):
        """Test that references directory is scanned."""
        skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
        skill = parse_skill_file(skill_path)

        # Check if references were loaded (may be empty if no references/ dir)
        assert isinstance(skill.references.files, dict)

    def test_parse_nonexistent_file(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_skill_file("nonexistent/SKILL.md")

    def test_parse_file_without_frontmatter(self, tmp_path):
        """Test that ValueError is raised for files without frontmatter."""
        # Create a temporary file without frontmatter
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# Just a markdown file\n\nNo frontmatter here.")

        with pytest.raises(ValueError, match="No YAML frontmatter"):
            parse_skill_file(skill_file)

    def test_parse_file_missing_name(self, tmp_path):
        """Test that ValueError is raised when 'name' field is missing."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\ndescription: Some description\n---\n\n# Body content")

        with pytest.raises(ValueError, match="Missing required field 'name'"):
            parse_skill_file(skill_file)

    def test_parse_file_missing_description(self, tmp_path):
        """Test that ValueError is raised when 'description' field is missing."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: test-skill\n---\n\n# Body content")

        with pytest.raises(ValueError, match="Missing required field 'description'"):
            parse_skill_file(skill_file)
