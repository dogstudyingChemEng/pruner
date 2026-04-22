"""
Stage 2 Optimizer for skill body restructuring.

Implements the taxonomy-driven classification and progressive disclosure
optimization from the SkillReducer paper.
"""

from typing import Optional

from .chunker import chunk_markdown_body
from .llm_client import BlockClassificationResult, SkillLLMClient
from .models import ContentBlock, ContentType, Skill


class Stage2Optimizer:
    """
    Stage 2 optimizer for skill body restructuring.

    Uses LLM to classify content blocks into taxonomy types for
    progressive disclosure optimization.
    """

    def __init__(
        self,
        llm_client: SkillLLMClient,
        batch_size: int = 12
    ):
        """
        Initialize the Stage 2 optimizer.

        Args:
            llm_client: LLM client for content classification.
            batch_size: Number of chunks to process per API call.
                       Default 12 to prevent context overflow.
        """
        self.llm_client = llm_client
        self.batch_size = batch_size

    def classify_skill_body(
        self,
        skill: Skill,
        skill_context: Optional[str] = None
    ) -> list[ContentBlock]:
        """
        Classify skill body content blocks into taxonomy types.

        Steps:
        1. Chunk the skill body into semantic blocks
        2. Batch chunks for LLM classification
        3. Map returned classifications back to ContentBlock objects

        Args:
            skill: The skill to classify.
            skill_context: Optional context string (e.g., name + description).

        Returns:
            List of ContentBlock objects with updated content_type.
        """
        # Step 1: Chunk the body
        content_blocks = chunk_markdown_body(skill.body.original)

        if not content_blocks:
            return []

        # Build skill context if not provided
        if skill_context is None:
            skill_context = f"Skill Name: {skill.name}\nDescription: {skill.description.original}"

        # Step 2: Classify in batches
        all_classifications = self._classify_in_batches(content_blocks, skill_context)

        # Step 3: Map classifications back to ContentBlock objects
        classification_map = {
            c.chunk_id: c.content_type
            for c in all_classifications.classifications
        }

        for block in content_blocks:
            if block.chunk_id in classification_map:
                block.content_type = classification_map[block.chunk_id]

        return content_blocks

    def _classify_in_batches(
        self,
        content_blocks: list[ContentBlock],
        skill_context: str
    ) -> BlockClassificationResult:
        """
        Classify content blocks in batches.

        Splits blocks into batches to prevent context overflow and
        structured output truncation.

        Args:
            content_blocks: List of content blocks to classify.
            skill_context: Context about the skill.

        Returns:
            Combined classification results from all batches.
        """
        all_classifications = []

        # Split into batches
        for i in range(0, len(content_blocks), self.batch_size):
            batch = content_blocks[i:i + self.batch_size]

            # Convert to dict format for LLM client
            chunks_for_llm = [
                {"chunk_id": block.chunk_id, "content": block.content}
                for block in batch
            ]

            # Call LLM for classification
            result = self.llm_client.classify_content_blocks(
                content_chunks=chunks_for_llm,
                skill_context=skill_context
            )

            all_classifications.extend(result.classifications)

        return BlockClassificationResult(classifications=all_classifications)

    def get_blocks_by_type(
        self,
        blocks: list[ContentBlock],
        content_type: ContentType
    ) -> list[ContentBlock]:
        """
        Filter content blocks by type.

        Args:
            blocks: List of content blocks.
            content_type: The content type to filter by.

        Returns:
            List of blocks matching the specified type.
        """
        return [b for b in blocks if b.content_type == content_type]

    def get_classification_summary(
        self,
        blocks: list[ContentBlock]
    ) -> dict[ContentType, int]:
        """
        Get summary statistics of block classifications.

        Args:
            blocks: List of classified content blocks.

        Returns:
            Dictionary mapping content type to count.
        """
        summary: dict[ContentType, int] = {}

        for block in blocks:
            if block.content_type not in summary:
                summary[block.content_type] = 0
            summary[block.content_type] += 1

        return summary

    def calculate_compression_potential(
        self,
        blocks: list[ContentBlock],
        original_body_tokens: Optional[int] = None
    ) -> dict[str, float]:
        """
        Calculate potential compression metrics.

        Based on SkillReducer paper:
        - Core rules are always loaded
        - Background, examples, templates can be on-demand
        - Redundant content is discarded

        The compression ratio is calculated as:
        compression_ratio = (original_body_tokens - core_tokens) / original_body_tokens
        This matches the paper's definition where the denominator is the original
        skill file's total tokens, not the chunked tokens.

        Args:
            blocks: List of classified content blocks.
            original_body_tokens: Original body token count before chunking.
                                 If None, uses sum of chunked tokens as fallback.

        Returns:
            Dictionary with compression metrics.
        """
        chunked_tokens = sum(b.token_count for b in blocks)

        # Use original tokens if provided, otherwise fall back to chunked tokens
        if original_body_tokens is not None and original_body_tokens > 0:
            total_tokens = original_body_tokens
        else:
            total_tokens = chunked_tokens

        if total_tokens == 0:
            return {
                "original_tokens": 0,
                "chunked_tokens": 0,
                "always_loaded_tokens": 0,
                "on_demand_tokens": 0,
                "discarded_tokens": 0,
                "compression_ratio": 0.0
            }

        core_tokens = sum(
            b.token_count for b in blocks
            if b.content_type == ContentType.CORE_RULE
        )

        on_demand_tokens = sum(
            b.token_count for b in blocks
            if b.content_type in (
                ContentType.BACKGROUND,
                ContentType.EXAMPLE,
                ContentType.TEMPLATE
            )
        )

        redundant_tokens = sum(
            b.token_count for b in blocks
            if b.content_type == ContentType.REDUNDANT
        )

        # Compression ratio based on original body tokens (paper definition)
        # compression_ratio = (original - core) / original
        # This represents the percentage of tokens that can be saved
        compression_ratio = (total_tokens - core_tokens) / total_tokens

        return {
            "original_tokens": total_tokens,
            "chunked_tokens": chunked_tokens,
            "always_loaded_tokens": core_tokens,
            "on_demand_tokens": on_demand_tokens,
            "discarded_tokens": redundant_tokens,
            "compression_ratio": round(compression_ratio, 4)
        }
