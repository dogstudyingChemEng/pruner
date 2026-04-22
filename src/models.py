"""
Data models for SkillReducer framework.

Defines the core data structures for representing skills and their components,
following the taxonomy-driven classification from the SkillReducer paper.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """
    Taxonomy-driven content classification for skill body restructuring.

    Based on SkillReducer paper (Gao et al., 2026), skill body content
    is classified into five types for progressive disclosure optimization.
    """

    CORE_RULE = "core_rule"
    """Actionable instructions - always loaded into context."""

    BACKGROUND = "background"
    """Explanations and context - on-demand module."""

    EXAMPLE = "example"
    """Code snippets and usage examples - on-demand module."""

    TEMPLATE = "template"
    """Boilerplate and ready-to-use templates - on-demand module."""

    REDUNDANT = "redundant"
    """Content that duplicates or provides no value - discarded."""


class ContentBlock(BaseModel):
    """
    A content block within a skill body, classified by type.

    Represents a semantic unit of content extracted from the skill body,
    ready for type-specific compression and progressive disclosure.
    """

    chunk_id: str = Field(
        default="",
        description="Unique identifier for the content chunk"
    )
    content: str = Field(..., description="The raw text content of the block")
    content_type: ContentType = Field(
        default=ContentType.CORE_RULE,
        description="Classification type for progressive disclosure"
    )
    token_count: int = Field(
        default=0,
        description="Number of tokens in the content block"
    )

    model_config = {
        "use_enum_values": True
    }


class Description(BaseModel):
    """
    Skill description (s.d) used for routing layer optimization.

    Short routing text used to match user queries to appropriate skills.
    This is the primary target for Stage 1 compression via delta debugging.
    """

    original: str = Field(..., description="Original description text")
    compressed: Optional[str] = Field(
        default=None,
        description="Compressed description after Stage 1 optimization"
    )
    original_token_count: int = Field(
        default=0,
        description="Token count of original description"
    )
    compressed_token_count: int = Field(
        default=0,
        description="Token count of compressed description"
    )

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (0.0 to 1.0)."""
        if self.original_token_count == 0:
            return 0.0
        return 1.0 - (self.compressed_token_count / self.original_token_count)


class Body(BaseModel):
    """
    Skill body (s.b) used for body restructuring via progressive disclosure.

    Main instruction document injected into LLM context. This is the primary
    target for Stage 2 compression via taxonomy-driven classification and
    progressive disclosure.
    """

    original: str = Field(..., description="Original body text")
    content_blocks: list[ContentBlock] = Field(
        default_factory=list,
        description="Classified content blocks extracted from body"
    )
    restructured: Optional[str] = Field(
        default=None,
        description="Restructured body after Stage 2 optimization"
    )
    original_token_count: int = Field(
        default=0,
        description="Token count of original body"
    )
    compressed_token_count: int = Field(
        default=0,
        description="Token count of restructured body"
    )

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (0.0 to 1.0)."""
        if self.original_token_count == 0:
            return 0.0
        return 1.0 - (self.compressed_token_count / self.original_token_count)

    def get_blocks_by_type(self, content_type: ContentType) -> list[ContentBlock]:
        """Get all content blocks of a specific type."""
        return [b for b in self.content_blocks if b.content_type == content_type]


class References(BaseModel):
    """
    Skill references (s.R) - optional files loaded alongside body.

    Reference files that provide additional context and knowledge.
    Subject to cross-file deduplication with the body content.
    """

    files: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of filename to file content"
    )
    total_token_count: int = Field(
        default=0,
        description="Total token count across all reference files"
    )


class SkillMetadata(BaseModel):
    """
    Metadata associated with a skill.

    Contains version, author, category, domain, tags, and other
    auxiliary information from the skill's YAML frontmatter.
    """

    version: Optional[str] = Field(default=None, description="Skill version")
    author: Optional[str] = Field(default=None, description="Skill author")
    category: Optional[str] = Field(default=None, description="Skill category")
    domain: Optional[str] = Field(default=None, description="Skill domain")
    tags: list[str] = Field(
        default_factory=list,
        description="Tags associated with the skill"
    )
    license: Optional[str] = Field(default=None, description="Skill license")
    updated: Optional[str] = Field(default=None, description="Last update date")
    extra: dict = Field(
        default_factory=dict,
        description="Additional metadata fields not explicitly defined"
    )


class Skill(BaseModel):
    """
    Complete skill model representing an LLM agent skill.

    A skill consists of four components (scripts are out of scope):
    - name: Skill identifier
    - description (s.d): Short routing text for query matching
    - body (s.b): Main instruction document
    - references (s.R): Optional reference files
    - metadata: Auxiliary information

    This model serves as the primary data structure for the SkillReducer
    pipeline, supporting both Stage 1 (routing optimization) and Stage 2
    (body restructuring) compression.
    """

    name: str = Field(..., description="Skill name/identifier")
    description: Description = Field(
        ...,
        description="Skill description for routing layer"
    )
    body: Body = Field(..., description="Skill body with main instructions")
    references: References = Field(
        default_factory=References,
        description="Optional reference files"
    )
    metadata: SkillMetadata = Field(
        default_factory=SkillMetadata,
        description="Skill metadata"
    )

    @property
    def total_original_tokens(self) -> int:
        """Total original tokens across description, body, and references."""
        return (
            self.description.original_token_count
            + self.body.original_token_count
            + self.references.total_token_count
        )

    @property
    def total_compressed_tokens(self) -> int:
        """Total compressed tokens after SkillReducer optimization."""
        return (
            self.description.compressed_token_count
            + self.body.compressed_token_count
            + self.references.total_token_count  # References not compressed in current scope
        )

    @property
    def overall_compression_ratio(self) -> float:
        """Overall compression ratio for the entire skill."""
        if self.total_original_tokens == 0:
            return 0.0
        return 1.0 - (self.total_compressed_tokens / self.total_original_tokens)
