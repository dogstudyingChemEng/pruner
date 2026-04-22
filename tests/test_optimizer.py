"""
Tests for the Stage 2 optimizer.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import BlockClassification, BlockClassificationResult
from src.models import ContentBlock, ContentType, Skill, Description, Body
from src.optimizer import Stage2Optimizer
from src.parser import parse_skill_file


class TestStage2Optimizer:
    """Tests for Stage2Optimizer class."""

    def _create_mock_llm_client(self) -> MagicMock:
        """Create a mock LLM client."""
        return MagicMock()

    def _create_mock_classification_result(
        self,
        chunk_ids: list[str],
        content_types: list[ContentType]
    ) -> BlockClassificationResult:
        """Create a mock classification result."""
        classifications = [
            BlockClassification(
                chunk_id=chunk_id,
                content_type=content_type,
                reasoning=f"Mock reasoning for {chunk_id}"
            )
            for chunk_id, content_type in zip(chunk_ids, content_types)
        ]
        return BlockClassificationResult(classifications=classifications)

    def test_classify_skill_body_with_mock(self):
        """Test skill body classification with mocked LLM client."""
        # Create a test skill
        skill = Skill(
            name="test-skill",
            description=Description(
                original="Test skill description",
                original_token_count=5
            ),
            body=Body(
                original="# Heading 1\n\nContent for heading 1.\n\n## Heading 2\n\nContent for heading 2.\n\n## Code Example\n\n```python\nprint('hello')\n```",
                original_token_count=50
            )
        )

        # Create mock LLM client
        mock_client = self._create_mock_llm_client()

        # Set up mock to return classifications
        def mock_classify(content_chunks, skill_context=None):
            # Return different types for different content
            chunk_ids = [c["chunk_id"] for c in content_chunks]
            # Assign types based on position
            types = []
            for i, _ in enumerate(chunk_ids):
                if i % 5 == 0:
                    types.append(ContentType.CORE_RULE)
                elif i % 5 == 1:
                    types.append(ContentType.BACKGROUND)
                elif i % 5 == 2:
                    types.append(ContentType.EXAMPLE)
                elif i % 5 == 3:
                    types.append(ContentType.TEMPLATE)
                else:
                    types.append(ContentType.REDUNDANT)
            return self._create_mock_classification_result(chunk_ids, types)

        mock_client.classify_content_blocks.side_effect = mock_classify

        # Create optimizer and classify
        optimizer = Stage2Optimizer(llm_client=mock_client, batch_size=10)
        classified_blocks = optimizer.classify_skill_body(skill)

        # Verify results
        assert len(classified_blocks) > 0

        # Verify LLM was called
        assert mock_client.classify_content_blocks.called

        # Verify each block has been classified
        print(f"\n{'='*60}")
        print("Classification Results:")
        print(f"{'='*60}")
        for block in classified_blocks:
            print(f"  {block.chunk_id}: {block.content_type.value} ({block.token_count} tokens)")
        print(f"{'='*60}\n")

        # Check that content types are set
        for block in classified_blocks:
            assert block.content_type in ContentType

    def test_classify_marketing_strategy_pmm_with_mock(self):
        """Test classification of real skill with mocked LLM."""
        skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
        skill = parse_skill_file(skill_path)

        # Get original body tokens
        from src.chunker import count_tokens
        original_body_tokens = count_tokens(skill.body.original)

        # Create mock LLM client
        mock_client = self._create_mock_llm_client()

        # Set up mock to simulate realistic classification
        def mock_classify(content_chunks, skill_context=None):
            chunk_ids = [c["chunk_id"] for c in content_chunks]
            types = []

            for chunk in content_chunks:
                content = chunk["content"].lower()
                # Simulate classification based on content patterns
                if "```" in content:
                    types.append(ContentType.EXAMPLE)
                elif content.startswith("#") and "workflow" in content:
                    types.append(ContentType.CORE_RULE)
                elif "template" in content:
                    types.append(ContentType.TEMPLATE)
                elif "|" in content and "---" in content:  # Table
                    types.append(ContentType.TEMPLATE)
                elif any(word in content for word in ["overview", "introduction", "background"]):
                    types.append(ContentType.BACKGROUND)
                else:
                    types.append(ContentType.CORE_RULE)

            return self._create_mock_classification_result(chunk_ids, types)

        mock_client.classify_content_blocks.side_effect = mock_classify

        # Create optimizer and classify
        optimizer = Stage2Optimizer(llm_client=mock_client, batch_size=12)
        classified_blocks = optimizer.classify_skill_body(skill)

        # Print summary
        summary = optimizer.get_classification_summary(classified_blocks)
        compression = optimizer.calculate_compression_potential(
            classified_blocks, original_body_tokens=original_body_tokens
        )

        print(f"\n{'='*60}")
        print(f"Classification Summary for: {skill.name}")
        print(f"{'='*60}")
        print(f"Total blocks: {len(classified_blocks)}")
        print("\nBy Content Type:")
        for content_type, count in summary.items():
            print(f"  {content_type.value}: {count} blocks")

        print(f"\nCompression Potential:")
        print(f"  Original tokens: {compression['original_tokens']}")
        print(f"  Chunked tokens: {compression['chunked_tokens']}")
        print(f"  Always loaded (core): {compression['always_loaded_tokens']}")
        print(f"  On-demand: {compression['on_demand_tokens']}")
        print(f"  Discarded: {compression['discarded_tokens']}")
        print(f"  Compression ratio: {compression['compression_ratio']:.2%}")
        print(f"{'='*60}\n")

        # Verify
        assert len(classified_blocks) > 0
        assert mock_client.classify_content_blocks.call_count > 0

    def test_batch_processing(self):
        """Test that large skills are processed in batches."""
        # Create a skill with many chunks
        large_body = "\n\n".join([
            f"## Section {i}\n\nContent for section {i}."
            for i in range(30)
        ])

        skill = Skill(
            name="large-skill",
            description=Description(original="Large skill", original_token_count=3),
            body=Body(original=large_body, original_token_count=500)
        )

        # Create mock
        mock_client = self._create_mock_llm_client()
        call_count = [0]

        def mock_classify(content_chunks, skill_context=None):
            call_count[0] += 1
            chunk_ids = [c["chunk_id"] for c in content_chunks]
            types = [ContentType.CORE_RULE] * len(chunk_ids)
            return self._create_mock_classification_result(chunk_ids, types)

        mock_client.classify_content_blocks.side_effect = mock_classify

        # Use small batch size to force multiple calls
        optimizer = Stage2Optimizer(llm_client=mock_client, batch_size=5)
        classified_blocks = optimizer.classify_skill_body(skill)

        # Verify multiple batches were processed
        print(f"\nTotal chunks: {len(classified_blocks)}")
        print(f"API calls made: {call_count[0]}")
        print(f"Batch size: 5")

        assert call_count[0] > 1, "Should have made multiple API calls for batching"
        assert len(classified_blocks) == 30  # 30 sections

    def test_empty_skill_body(self):
        """Test handling of empty skill body."""
        skill = Skill(
            name="empty-skill",
            description=Description(original="Empty skill", original_token_count=2),
            body=Body(original="", original_token_count=0)
        )

        mock_client = self._create_mock_llm_client()
        optimizer = Stage2Optimizer(llm_client=mock_client)

        result = optimizer.classify_skill_body(skill)

        assert result == []
        mock_client.classify_content_blocks.assert_not_called()

    def test_get_blocks_by_type(self):
        """Test filtering blocks by type."""
        blocks = [
            ContentBlock(
                chunk_id="1",
                content="Core content",
                content_type=ContentType.CORE_RULE,
                token_count=10
            ),
            ContentBlock(
                chunk_id="2",
                content="Background info",
                content_type=ContentType.BACKGROUND,
                token_count=20
            ),
            ContentBlock(
                chunk_id="3",
                content="Another core",
                content_type=ContentType.CORE_RULE,
                token_count=15
            ),
        ]

        mock_client = self._create_mock_llm_client()
        optimizer = Stage2Optimizer(llm_client=mock_client)

        core_blocks = optimizer.get_blocks_by_type(blocks, ContentType.CORE_RULE)
        assert len(core_blocks) == 2

        background_blocks = optimizer.get_blocks_by_type(blocks, ContentType.BACKGROUND)
        assert len(background_blocks) == 1

    def test_calculate_compression_potential(self):
        """Test compression potential calculation."""
        blocks = [
            ContentBlock(
                chunk_id="1",
                content="Core",
                content_type=ContentType.CORE_RULE,
                token_count=100
            ),
            ContentBlock(
                chunk_id="2",
                content="Background",
                content_type=ContentType.BACKGROUND,
                token_count=200
            ),
            ContentBlock(
                chunk_id="3",
                content="Example",
                content_type=ContentType.EXAMPLE,
                token_count=150
            ),
            ContentBlock(
                chunk_id="4",
                content="Template",
                content_type=ContentType.TEMPLATE,
                token_count=50
            ),
            ContentBlock(
                chunk_id="5",
                content="Redundant",
                content_type=ContentType.REDUNDANT,
                token_count=100
            ),
        ]

        mock_client = self._create_mock_llm_client()
        optimizer = Stage2Optimizer(llm_client=mock_client)

        result = optimizer.calculate_compression_potential(blocks)

        assert result["original_tokens"] == 600
        assert result["always_loaded_tokens"] == 100
        assert result["on_demand_tokens"] == 400  # background + example + template
        assert result["discarded_tokens"] == 100
        # Compression ratio = (600 - 100) / 600 = 0.8333
        assert result["compression_ratio"] == pytest.approx(0.8333, rel=0.01)

    def test_calculate_compression_potential_with_original_tokens(self):
        """Test compression calculation with original_body_tokens parameter."""
        blocks = [
            ContentBlock(
                chunk_id="1",
                content="Core",
                content_type=ContentType.CORE_RULE,
                token_count=100
            ),
        ]

        mock_client = self._create_mock_llm_client()
        optimizer = Stage2Optimizer(llm_client=mock_client)

        # Test with original_body_tokens parameter
        result = optimizer.calculate_compression_potential(
            blocks,
            original_body_tokens=200
        )

        # original_tokens should use the provided value (200)
        # not the chunked tokens (100)
        assert result["original_tokens"] == 200
        assert result["chunked_tokens"] == 100
        assert result["always_loaded_tokens"] == 100
        # Compression ratio = (200 - 100) / 200 = 0.5
        assert result["compression_ratio"] == 0.5
