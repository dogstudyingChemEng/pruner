"""
Tests for the Stage 2 optimizer.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import BlockClassification, BlockClassificationResult
from src.models import ContentBlock, ContentType, Skill, Description, Body, References, SkillMetadata, OnDemandModules
from src.optimizer import Stage2Optimizer, CompressionMetrics
from src.parser import parse_skill_file


class TestStage2Optimizer:
    """Tests for Stage2Optimizer class."""

    def _create_mock_llm_client(self) -> MagicMock:
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.model = "gpt-4o-mini"
        mock.client = MagicMock()
        return mock

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


class TestStage2CompressionMethods:
    """Tests for Stage 2 compression methods."""

    def _create_mock_llm_client(self) -> MagicMock:
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.model = "gpt-4o-mini"
        mock.client = MagicMock()
        return mock

    def _mock_response(self, content: str):
        """Create a mock response with given content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        return mock_response

    def test_compress_core_rules(self):
        """Test compression of core rules into bullet points."""
        mock_client = self._create_mock_llm_client()

        # Mock compression response
        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "compressed_rules": [
                    "• Analyze market data and identify trends",
                    "• Create positioning strategy based on findings",
                    "• Launch campaigns with defined metrics"
                ],
                "rules_merged": 3
            })
        )

        optimizer = Stage2Optimizer(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="1",
                content="First, you should analyze the market data and identify key trends.",
                content_type=ContentType.CORE_RULE,
                token_count=20
            ),
            ContentBlock(
                chunk_id="2",
                content="Next, create a positioning strategy based on your findings.",
                content_type=ContentType.CORE_RULE,
                token_count=15
            ),
            ContentBlock(
                chunk_id="3",
                content="Finally, launch campaigns with defined success metrics.",
                content_type=ContentType.CORE_RULE,
                token_count=15
            ),
        ]

        result = optimizer.compress_core_rules(blocks)

        # Should return a single compressed block
        assert len(result) == 1
        assert result[0].chunk_id == "compressed_core_rules"
        assert "•" in result[0].content or "-" in result[0].content
        # Compressed should be shorter
        assert result[0].token_count < sum(b.token_count for b in blocks)

    def test_compress_core_rules_empty(self):
        """Test compression with no core rules."""
        mock_client = self._create_mock_llm_client()
        optimizer = Stage2Optimizer(llm_client=mock_client)

        # No core rules
        blocks = [
            ContentBlock(
                chunk_id="1",
                content="Background info",
                content_type=ContentType.BACKGROUND,
                token_count=10
            )
        ]

        result = optimizer.compress_core_rules(blocks)
        assert result == []

    def test_dedup_examples(self):
        """Test deduplication of examples."""
        mock_client = self._create_mock_llm_client()

        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "concept_groups": [
                    {
                        "concept": "API authentication",
                        "examples_in_group": ["1", "2"],
                        "selected_example_id": "1",
                        "reason": "More complete example"
                    }
                ],
                "deduplicated_examples": [
                    {
                        "chunk_id": "1",
                        "content": "response = client.authenticate(token)"
                    }
                ],
                "examples_removed": 1
            })
        )

        optimizer = Stage2Optimizer(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="1",
                content="```python\nresponse = client.authenticate(token)\n```",
                content_type=ContentType.EXAMPLE,
                token_count=20
            ),
            ContentBlock(
                chunk_id="2",
                content="```python\n# Another auth example\nresponse = client.authenticate(api_key)\n```",
                content_type=ContentType.EXAMPLE,
                token_count=25
            ),
        ]

        result, removed = optimizer.dedup_examples(blocks)

        assert len(result) == 1
        assert removed == 1
        assert "authenticate" in result[0].content

    def test_dedup_examples_single(self):
        """Test deduplication with single example returns unchanged."""
        mock_client = self._create_mock_llm_client()
        optimizer = Stage2Optimizer(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="1",
                content="Single example",
                content_type=ContentType.EXAMPLE,
                token_count=10
            )
        ]

        result, removed = optimizer.dedup_examples(blocks)

        assert len(result) == 1
        assert removed == 0

    def test_dedup_templates(self):
        """Test deduplication of templates."""
        mock_client = self._create_mock_llm_client()

        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "concept_groups": [
                    {
                        "concept": "email template",
                        "templates_in_group": ["1", "2"],
                        "selected_template_id": "1"
                    }
                ],
                "deduplicated_templates": [
                    {
                        "chunk_id": "1",
                        "content": "Subject: {{title}}\n\nDear {{name}},"
                    }
                ],
                "templates_removed": 1
            })
        )

        optimizer = Stage2Optimizer(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="1",
                content="Subject: {{title}}\n\nDear {{name}},\n\nContent here.",
                content_type=ContentType.TEMPLATE,
                token_count=15
            ),
            ContentBlock(
                chunk_id="2",
                content="Subject: {{subject}}\n\nHi {{user}},\n\nMessage.",
                content_type=ContentType.TEMPLATE,
                token_count=15
            ),
        ]

        result, removed = optimizer.dedup_templates(blocks)

        assert len(result) == 1
        assert removed == 1

    def test_summarize_background(self):
        """Test summarization of background content."""
        mock_client = self._create_mock_llm_client()

        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "summary": "Product marketing focuses on go-to-market strategy, with key metrics including 15% conversion rate and 30-day trial periods. API endpoints are at api.example.com/v2.",
                "facts_preserved": ["15% conversion rate", "30-day trial", "api.example.com/v2"]
            })
        )

        optimizer = Stage2Optimizer(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="1",
                content="Product marketing is a strategic function. Key metrics include 15% conversion rate.",
                content_type=ContentType.BACKGROUND,
                token_count=25
            ),
            ContentBlock(
                chunk_id="2",
                content="Trials are 30-day periods. API is at api.example.com/v2.",
                content_type=ContentType.BACKGROUND,
                token_count=20
            ),
        ]

        result, merged = optimizer.summarize_background(blocks)

        assert len(result) == 1
        assert merged == 1  # 2 blocks -> 1 block, so 1 merged
        # Should preserve key facts
        assert "15%" in result[0].content
        assert "api.example.com" in result[0].content

    def test_summarize_background_single(self):
        """Test summarization with single block returns unchanged."""
        mock_client = self._create_mock_llm_client()
        optimizer = Stage2Optimizer(llm_client=mock_client)

        blocks = [
            ContentBlock(
                chunk_id="1",
                content="Single background paragraph.",
                content_type=ContentType.BACKGROUND,
                token_count=10
            )
        ]

        result, merged = optimizer.summarize_background(blocks)

        assert len(result) == 1
        assert merged == 0

    def test_dedup_references(self):
        """Test cross-file deduplication with references."""
        mock_client = self._create_mock_llm_client()

        # Mock response for dedup
        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "deduplicated_content": "# Additional Notes\n\nThis is unique reference content not in the body.",
                "duplicates_removed": 2,
                "unique_content_preserved": True
            })
        )

        optimizer = Stage2Optimizer(llm_client=mock_client)

        body_blocks = [
            ContentBlock(
                chunk_id="1",
                content="Core rule 1: Do this.\nCore rule 2: Do that.",
                content_type=ContentType.CORE_RULE,
                token_count=20
            )
        ]

        references = References(
            files={
                "reference1.md": "# Reference\n\nCore rule 1: Do this.\n\nAdditional notes here."
            },
            total_token_count=30
        )

        new_refs, deduped, discarded = optimizer.dedup_references(
            "Core rule 1: Do this.\nCore rule 2: Do that.", references, min_token_threshold=10
        )

        assert deduped == 2
        assert discarded == 0
        assert "reference1.md" in new_refs.files

    def test_dedup_references_discard_small_file(self):
        """Test that reference files under threshold are discarded."""
        mock_client = self._create_mock_llm_client()

        # Mock response - content becomes very short after dedup
        mock_client.client.chat.completions.create.return_value = self._mock_response(
            json.dumps({
                "deduplicated_content": "Short",  # Very short content
                "duplicates_removed": 5,
                "unique_content_preserved": True
            })
        )

        optimizer = Stage2Optimizer(llm_client=mock_client)

        body_blocks = [
            ContentBlock(
                chunk_id="1",
                content="Core rule content",
                content_type=ContentType.CORE_RULE,
                token_count=10
            )
        ]

        references = References(
            files={
                "small_ref.md": "This duplicates the core rule content entirely."
            },
            total_token_count=15
        )

        new_refs, deduped, discarded = optimizer.dedup_references(
            "Core rule content", references, min_token_threshold=30
        )

        # File should be discarded because remaining content < 30 tokens
        assert discarded == 1
        assert len(new_refs.files) == 0

    def test_dedup_references_empty(self):
        """Test dedup with empty references."""
        mock_client = self._create_mock_llm_client()
        optimizer = Stage2Optimizer(llm_client=mock_client)

        body_blocks = [
            ContentBlock(
                chunk_id="1",
                content="Content",
                content_type=ContentType.CORE_RULE,
                token_count=10
            )
        ]

        references = References()

        new_refs, deduped, discarded = optimizer.dedup_references("Content", references)

        assert deduped == 0
        assert discarded == 0
        assert len(new_refs.files) == 0


class TestFullOptimizationPipeline:
    """Tests for the full optimization pipeline."""

    def _create_mock_llm_client(self) -> MagicMock:
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

    def test_optimize_skill_full_pipeline(self):
        """Test full optimization pipeline with all steps."""
        mock_client = self._create_mock_llm_client()

        # Set up mocks for each step (cross-validation happens before compression)
        responses = iter([
            # Classification
            json.dumps({
                "classifications": [
                    {"chunk_id": "chunk_1", "content_type": "core_rule", "reasoning": "test"},
                    {"chunk_id": "chunk_2", "content_type": "example", "reasoning": "test"},
                    {"chunk_id": "chunk_3", "content_type": "background", "reasoning": "test"}
                ]
            }),
            # Cross-validation: block 1 (core_rule, no change)
            json.dumps({"changed": False, "corrected_type": "core_rule", "reasoning": "correct"}),
            # Cross-validation: block 2 (example, no change)
            json.dumps({"changed": False, "corrected_type": "example", "reasoning": "correct"}),
            # Cross-validation: block 3 (background, no change)
            json.dumps({"changed": False, "corrected_type": "background", "reasoning": "correct"}),
            # Compress core
            json.dumps({
                "compressed_rules": ["• Rule 1: Do this"],
                "rules_merged": 1
            }),
            # Dedup examples (single example, returns unchanged)
            # Summarize background
            json.dumps({
                "summary": "Background summary with key facts.",
                "facts_preserved": ["fact1"]
            }),
        ])

        def mock_create(*args, **kwargs):
            return self._mock_response(next(responses))

        mock_client.classify_content_blocks.return_value = BlockClassificationResult(
            classifications=[
                BlockClassification(chunk_id="chunk_1", content_type=ContentType.CORE_RULE),
                BlockClassification(chunk_id="chunk_2", content_type=ContentType.EXAMPLE),
                BlockClassification(chunk_id="chunk_3", content_type=ContentType.BACKGROUND)
            ]
        )
        mock_client.client.chat.completions.create = mock_create

        optimizer = Stage2Optimizer(llm_client=mock_client)

        skill = Skill(
            name="test-skill",
            description=Description(original="Test skill", original_token_count=5),
            body=Body(
                original="# Rules\n\nDo this task.\n\n## Example\n\n```python\ncode()\n```\n\n## Background\n\nSome background info.",
                original_token_count=50
            ),
            references=References(),
            metadata=SkillMetadata()
        )

        core_blocks, on_demand, refs, metrics = optimizer.optimize_skill(skill)

        assert isinstance(metrics, CompressionMetrics)
        assert metrics.original_tokens > 0
        assert metrics.compressed_tokens > 0
        assert isinstance(metrics.compression_ratio, float)
        assert isinstance(on_demand, OnDemandModules)

    def test_optimize_skill_skip_steps(self):
        """Test optimization with some steps disabled."""
        mock_client = self._create_mock_llm_client()

        mock_client.classify_content_blocks.return_value = BlockClassificationResult(
            classifications=[
                BlockClassification(chunk_id="chunk_1", content_type=ContentType.CORE_RULE)
            ]
        )

        optimizer = Stage2Optimizer(llm_client=mock_client)

        skill = Skill(
            name="test-skill",
            description=Description(original="Test", original_token_count=2),
            body=Body(original="# Rule\n\nContent here.", original_token_count=10),
            references=References(),
            metadata=SkillMetadata()
        )

        # Skip all compression steps
        core_blocks, on_demand, refs, metrics = optimizer.optimize_skill(
            skill,
            compress_core=False,
            dedup_examples=False,
            dedup_templates=False,
            summarize_background=False,
            dedup_references=False
        )

        # Should still classify but not compress
        assert len(core_blocks) >= 0
        assert isinstance(metrics, CompressionMetrics)
        assert isinstance(on_demand, OnDemandModules)
