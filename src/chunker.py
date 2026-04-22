"""
Markdown chunker for splitting skill body into semantic content blocks.

Uses markdown-it-py for parsing and tiktoken for token counting.
"""

import uuid
from typing import Optional

from markdown_it import MarkdownIt
from markdown_it.token import Token
import tiktoken

from .models import ContentBlock, ContentType


def chunk_markdown_body(
    body_text: str,
    default_content_type: ContentType = ContentType.CORE_RULE
) -> list[ContentBlock]:
    """
    Split markdown body into semantic content blocks.

    Parses markdown using markdown-it-py and splits content at heading
    and paragraph boundaries, creating meaningful chunks for classification.

    Args:
        body_text: The markdown body text to chunk.
        default_content_type: Default content type for chunks (before classification).

    Returns:
        List of ContentBlock objects with unique IDs and token counts.
    """
    if not body_text or not body_text.strip():
        return []

    # Initialize markdown parser and tokenizer
    md = MarkdownIt()
    encoding = tiktoken.get_encoding("cl100k_base")

    # Parse markdown into tokens
    tokens = md.parse(body_text)

    # Group tokens into semantic chunks
    chunks = _group_tokens_into_chunks(tokens, body_text)

    # Create ContentBlock objects
    content_blocks = []
    for chunk_text in chunks:
        if not chunk_text.strip():
            continue

        chunk_id = _generate_chunk_id()
        token_count = len(encoding.encode(chunk_text))

        block = ContentBlock(
            chunk_id=chunk_id,
            content=chunk_text.strip(),
            content_type=default_content_type,
            token_count=token_count
        )
        content_blocks.append(block)

    return content_blocks


def _group_tokens_into_chunks(
    tokens: list[Token],
    original_text: str
) -> list[str]:
    """
    Group markdown tokens into semantic chunks.

    Strategy:
    - Each heading (h1-h6) starts a new chunk
    - Content following a heading (paragraphs, lists, code blocks, tables)
      is grouped with that heading until the next heading
    - Top-level content (before any heading) forms its own chunk

    Args:
        tokens: Parsed markdown tokens.
        original_text: Original markdown text for position reference.

    Returns:
        List of chunk text strings.
    """
    chunks = []
    current_chunk_lines = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Heading starts a new chunk
        if token.type == "heading_open":
            # Save current chunk if not empty
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []

            # Collect heading and its content
            heading_content = _collect_heading_content(tokens, i)
            current_chunk_lines.append(heading_content)
            i += 3  # Skip heading_open, inline, heading_close
            continue

        # Paragraph content
        if token.type == "paragraph_open":
            paragraph_text = _collect_paragraph_content(tokens, i)
            if paragraph_text.strip():
                current_chunk_lines.append(paragraph_text)
            i += 3  # Skip paragraph_open, inline, paragraph_close
            continue

        # Code block (fenced)
        if token.type == "fence":
            code_block = f"```\n{token.content}\n```" if token.info == "" else f"```{token.info}\n{token.content}\n```"
            current_chunk_lines.append(code_block)
            i += 1
            continue

        # Code block (indented)
        if token.type == "code_block":
            current_chunk_lines.append(f"```\n{token.content}\n```")
            i += 1
            continue

        # List (bullet or ordered)
        if token.type == "bullet_list_open" or token.type == "ordered_list_open":
            list_text = _collect_list_content(tokens, i)
            if list_text.strip():
                current_chunk_lines.append(list_text)
            # Skip to after list_close
            i = _find_matching_close(tokens, i, token.type.replace("open", "close")) + 1
            continue

        # Table
        if token.type == "table_open":
            table_text = _collect_table_content(tokens, i)
            if table_text.strip():
                current_chunk_lines.append(table_text)
            i = _find_matching_close(tokens, i, "table_close") + 1
            continue

        # Blockquote
        if token.type == "blockquote_open":
            quote_text = _collect_blockquote_content(tokens, i)
            if quote_text.strip():
                current_chunk_lines.append(quote_text)
            i = _find_matching_close(tokens, i, "blockquote_close") + 1
            continue

        # Horizontal rule
        if token.type == "hr":
            current_chunk_lines.append("---")
            i += 1
            continue

        # Other tokens (skip most)
        i += 1

    # Don't forget the last chunk
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks


def _collect_heading_content(tokens: list[Token], start_idx: int) -> str:
    """Collect heading text (e.g., '## Section Title')."""
    if start_idx + 2 >= len(tokens):
        return ""

    level = tokens[start_idx].tag  # h1, h2, etc.
    inline_token = tokens[start_idx + 1]

    if inline_token.type != "inline":
        return ""

    heading_text = inline_token.content
    return f"{'#' * int(level[1])} {heading_text}"


def _collect_paragraph_content(tokens: list[Token], start_idx: int) -> str:
    """Collect paragraph text."""
    if start_idx + 2 >= len(tokens):
        return ""

    inline_token = tokens[start_idx + 1]
    if inline_token.type != "inline":
        return ""

    return inline_token.content


def _collect_list_content(tokens: list[Token], start_idx: int) -> str:
    """Collect list items as text."""
    lines = []
    i = start_idx + 1
    depth = 1

    while i < len(tokens) and depth > 0:
        token = tokens[i]

        if "open" in token.type and "list" in token.type:
            depth += 1
        elif "close" in token.type and "list" in token.type:
            depth -= 1
        elif token.type == "list_item_open":
            # Find inline content in this list item
            j = i + 1
            while j < len(tokens) and tokens[j].type != "list_item_close":
                if tokens[j].type == "inline":
                    lines.append(f"- {tokens[j].content}")
                j += 1

        i += 1

    return "\n".join(lines)


def _collect_table_content(tokens: list[Token], start_idx: int) -> str:
    """Collect table as text representation."""
    lines = []
    i = start_idx + 1
    depth = 1

    while i < len(tokens) and depth > 0:
        token = tokens[i]

        if token.type == "table_open":
            depth += 1
        elif token.type == "table_close":
            depth -= 1
        elif token.type == "tr_open":
            row_cells = []
            j = i + 1
            while j < len(tokens) and tokens[j].type != "tr_close":
                if tokens[j].type == "inline":
                    row_cells.append(tokens[j].content)
                j += 1
            if row_cells:
                lines.append("| " + " | ".join(row_cells) + " |")

        i += 1

    return "\n".join(lines)


def _collect_blockquote_content(tokens: list[Token], start_idx: int) -> str:
    """Collect blockquote content."""
    lines = []
    i = start_idx + 1
    depth = 1

    while i < len(tokens) and depth > 0:
        token = tokens[i]

        if token.type == "blockquote_open":
            depth += 1
        elif token.type == "blockquote_close":
            depth -= 1
        elif token.type == "inline":
            lines.append(f"> {token.content}")

        i += 1

    return "\n".join(lines)


def _find_matching_close(tokens: list[Token], start_idx: int, close_type: str) -> int:
    """Find the index of the matching close token."""
    depth = 1
    open_type = close_type.replace("close", "open")

    for i in range(start_idx + 1, len(tokens)):
        if tokens[i].type == open_type:
            depth += 1
        elif tokens[i].type == close_type:
            depth -= 1
            if depth == 0:
                return i

    return len(tokens) - 1


def _generate_chunk_id() -> str:
    """Generate a unique chunk ID."""
    return f"chunk_{uuid.uuid4().hex[:8]}"


def get_chunk_encoding() -> tiktoken.Encoding:
    """Get the tiktoken encoding used for chunking."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count tokens in text using cl100k_base encoding.

    Args:
        text: Text to count tokens for.

    Returns:
        Number of tokens.
    """
    if not text:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
