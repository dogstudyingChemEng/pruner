"""
Skill file parser for extracting skill data from SKILL.md files.

Parses YAML frontmatter and Markdown body from skill files,
validating and encapsulating data into Skill model objects.
"""

import re
from pathlib import Path
from typing import Optional

import yaml

from .models import Body, Description, References, Skill, SkillMetadata


def parse_skill_file(filepath: str | Path) -> Skill:
    """
    Parse a skill file and return a Skill object.

    Extracts YAML frontmatter from the top of the file, preserves the
    remaining Markdown content as the body, and validates all data
    against the Skill model.

    Args:
        filepath: Path to the SKILL.md file (string or Path object).

    Returns:
        Skill: A validated Skill object containing parsed data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file has invalid YAML frontmatter or missing
            required fields (name, description).
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Skill file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8")
    frontmatter, body_text = _extract_frontmatter(content)

    if frontmatter is None:
        raise ValueError(
            f"No YAML frontmatter found in skill file: {filepath}. "
            "Expected '---' delimiters at the start of the file."
        )

    # Validate required fields
    if "name" not in frontmatter:
        raise ValueError(f"Missing required field 'name' in frontmatter: {filepath}")
    if "description" not in frontmatter:
        raise ValueError(f"Missing required field 'description' in frontmatter: {filepath}")

    # Build Skill object
    skill = _build_skill(frontmatter, body_text, filepath)

    return skill


def _extract_frontmatter(content: str) -> tuple[Optional[dict], str]:
    """
    Extract YAML frontmatter and body from file content.

    YAML frontmatter is expected to be at the start of the file,
    enclosed by '---' delimiters.

    Args:
        content: The raw file content.

    Returns:
        A tuple of (frontmatter_dict, body_text).
        frontmatter_dict is None if no valid frontmatter is found.
        body_text is the remaining content after frontmatter (stripped).
    """
    # Pattern matches: ---\n<yaml content>\n---
    pattern = r"^---\s*\n(.*?)\n---\s*\n?(.*)"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return None, content.strip()

    yaml_content = match.group(1)
    body_text = match.group(2).strip()

    try:
        frontmatter = yaml.safe_load(yaml_content)
        if not isinstance(frontmatter, dict):
            return None, content.strip()
    except yaml.YAMLError:
        return None, content.strip()

    return frontmatter, body_text


def _build_skill(
    frontmatter: dict,
    body_text: str,
    filepath: Path
) -> Skill:
    """
    Build a Skill object from parsed frontmatter and body.

    Args:
        frontmatter: Parsed YAML frontmatter dictionary.
        body_text: The Markdown body content.
        filepath: Path to the skill file (for reference resolution).

    Returns:
        A validated Skill object.
    """
    # Extract name and description (required fields)
    name = frontmatter["name"]
    description_text = frontmatter["description"]

    # Build Description object
    description = Description(
        original=description_text,
        original_token_count=0  # Will be computed by tokenizer later
    )

    # Build Body object
    body = Body(
        original=body_text,
        original_token_count=0  # Will be computed by tokenizer later
    )

    # Build Metadata object
    metadata = _build_metadata(frontmatter)

    # Build References object (parse reference files if they exist)
    references = _build_references(filepath.parent)

    return Skill(
        name=name,
        description=description,
        body=body,
        references=references,
        metadata=metadata
    )


def _build_metadata(frontmatter: dict) -> SkillMetadata:
    """
    Build SkillMetadata from frontmatter.

    Extracts known metadata fields and stores unknown fields in extra.
    Handles both flat structure and nested 'metadata' key structure.

    Args:
        frontmatter: Parsed YAML frontmatter dictionary.

    Returns:
        A SkillMetadata object.
    """
    # Known metadata fields at root level
    known_fields = {
        "version", "author", "category", "domain", "tags",
        "license", "updated", "name", "description", "metadata"
    }

    # Check if metadata is nested under 'metadata' key
    nested_metadata = frontmatter.get("metadata", {})

    # Extract known fields, preferring nested metadata if available
    metadata = SkillMetadata(
        version=nested_metadata.get("version") or frontmatter.get("version"),
        author=nested_metadata.get("author") or frontmatter.get("author"),
        category=nested_metadata.get("category") or frontmatter.get("category"),
        domain=nested_metadata.get("domain") or frontmatter.get("domain"),
        tags=nested_metadata.get("tags", []) or frontmatter.get("tags", []),
        license=frontmatter.get("license"),  # license is typically at root level
        updated=_format_updated(nested_metadata.get("updated") or frontmatter.get("updated")),
    )

    # Store unknown fields in extra (excluding the nested metadata dict)
    extra = {
        key: value
        for key, value in frontmatter.items()
        if key not in known_fields
    }
    metadata.extra = extra

    return metadata


def _format_updated(value) -> Optional[str]:
    """Format the updated field to string if it's a date object."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _build_references(skill_dir: Path) -> References:
    """
    Build References from reference files in the skill directory.

    Looks for a 'references/' subdirectory and loads all .md files.

    Args:
        skill_dir: Path to the skill directory containing SKILL.md.

    Returns:
        A References object with loaded reference files.
    """
    references = References()

    ref_dir = skill_dir / "references"
    if not ref_dir.exists() or not ref_dir.is_dir():
        return references

    for ref_file in ref_dir.glob("*.md"):
        try:
            content = ref_file.read_text(encoding="utf-8")
            references.files[ref_file.name] = content
        except Exception:
            # Skip files that cannot be read
            continue

    return references


def parse_skill_directory(skill_dir: str | Path) -> Optional[Skill]:
    """
    Parse a skill directory and return a Skill object.

    Looks for a SKILL.md file in the directory and parses it.

    Args:
        skill_dir: Path to the skill directory.

    Returns:
        Skill object if SKILL.md exists and is valid, None otherwise.
    """
    skill_dir = Path(skill_dir)
    skill_file = skill_dir / "SKILL.md"

    if not skill_file.exists():
        return None

    try:
        return parse_skill_file(skill_file)
    except (FileNotFoundError, ValueError):
        return None
