"""
Tests for the markdown chunker.
"""

from pathlib import Path

import pytest

from src.chunker import chunk_markdown_body, count_tokens
from src.models import ContentType
from src.parser import parse_skill_file


class TestChunkMarkdownBody:
    """Tests for chunk_markdown_body function."""

    def test_chunk_marketing_strategy_pmm_body(self):
        """Test chunking the marketing-strategy-pmm skill body."""
        # Parse the skill file to get the body
        skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
        skill = parse_skill_file(skill_path)

        # Chunk the body
        chunks = chunk_markdown_body(skill.body.original)

        # Verify we got chunks
        assert len(chunks) > 0

        # Print results
        print(f"\n{'='*60}")
        print(f"Chunking Results for: {skill.name}")
        print(f"{'='*60}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"\n{'='*60}")
        print("Chunk Details:")
        print(f"{'='*60}")

        total_tokens = 0
        for i, chunk in enumerate(chunks, 1):
            total_tokens += chunk.token_count
            preview = chunk.content[:100].replace('\n', ' ') + "..." if len(chunk.content) > 100 else chunk.content.replace('\n', ' ')
            print(f"\nChunk {i}:")
            print(f"  ID: {getattr(chunk, 'chunk_id', 'N/A')}")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Preview: {preview}")

        print(f"\n{'='*60}")
        print(f"Total tokens across all chunks: {total_tokens}")
        print(f"{'='*60}\n")

    def test_chunk_preserves_content(self):
        """Test that chunking preserves most content."""
        skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
        skill = parse_skill_file(skill_path)

        original_tokens = count_tokens(skill.body.original)
        chunks = chunk_markdown_body(skill.body.original)

        chunked_tokens = sum(c.token_count for c in chunks)

        print(f"\nOriginal body tokens: {original_tokens}")
        print(f"Chunked tokens sum: {chunked_tokens}")
        print(f"Ratio: {chunked_tokens / original_tokens:.2%}")

        # Chunked content should capture significant portion of the original
        # Note: Some loss is expected due to markdown structure elements
        assert chunked_tokens >= original_tokens * 0.5

    def test_chunk_has_token_counts(self):
        """Test that each chunk has a valid token count."""
        skill_path = Path("Claude-Skills/marketing/marketing-strategy-pmm/SKILL.md")
        skill = parse_skill_file(skill_path)

        chunks = chunk_markdown_body(skill.body.original)

        for chunk in chunks:
            assert chunk.token_count > 0
            assert isinstance(chunk.token_count, int)

    def test_chunk_default_content_type(self):
        """Test that chunks have default content type."""
        text = "# Test Heading\n\nThis is a paragraph."
        chunks = chunk_markdown_body(text, default_content_type=ContentType.BACKGROUND)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content_type == ContentType.BACKGROUND

    def test_chunk_empty_body(self):
        """Test chunking empty body."""
        chunks = chunk_markdown_body("")
        assert len(chunks) == 0

        chunks = chunk_markdown_body("   \n\n   ")
        assert len(chunks) == 0

    def test_chunk_simple_markdown(self):
        """Test chunking simple markdown."""
        text = """# Main Title

This is the first paragraph.

## Section 1

This is content under section 1.

- Item 1
- Item 2

### Subsection

More content here.
"""
        chunks = chunk_markdown_body(text)

        assert len(chunks) > 0

        # Should have chunks for headings and their content
        chunk_contents = [c.content for c in chunks]
        print(f"\nSimple markdown chunks: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"  {i+1}: {c.content[:50]}...")

    def test_chunk_with_code_blocks(self):
        """Test chunking markdown with code blocks."""
        text = """## Code Example

Here is some code:

```python
def hello():
    print("Hello, world!")
```

And more text.
"""
        chunks = chunk_markdown_body(text)

        assert len(chunks) > 0

        # Find the chunk with code
        code_chunks = [c for c in chunks if "```" in c.content]
        assert len(code_chunks) > 0

    def test_chunk_with_tables(self):
        """Test chunking markdown with tables."""
        text = """## Data Table

| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |

End of table.
"""
        chunks = chunk_markdown_body(text)

        assert len(chunks) > 0

        # Should contain table content
        all_content = "\n".join(c.content for c in chunks)
        assert "|" in all_content  # Table markers should be present


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_count_empty_text(self):
        """Test token count for empty text."""
        assert count_tokens("") == 0

    def test_count_simple_text(self):
        """Test token count for simple text."""
        text = "Hello, world!"
        count = count_tokens(text)
        assert count > 0
        assert count < 10  # Should be small

    def test_count_code_text(self):
        """Test token count for code."""
        code = "def hello():\n    print('Hello')"
        count = count_tokens(code)
        assert count > 0
