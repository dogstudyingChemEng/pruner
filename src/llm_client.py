"""
LLM client for SkillReducer framework.

Provides OpenAI-compatible API client for skill content classification
and compression tasks, with support for alternative providers (DeepSeek, Qwen).
"""

import json
import os
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import ContentType


class BlockClassification(BaseModel):
    """Classification result for a single content block."""

    chunk_id: str = Field(..., description="Unique identifier for the content chunk")
    content_type: ContentType = Field(
        ...,
        description="Classification type: core_rule, background, example, template, or redundant"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation for the classification decision"
    )


class BlockClassificationResult(BaseModel):
    """Result containing classifications for all content blocks."""

    classifications: list[BlockClassification] = Field(
        ...,
        description="List of content block classifications"
    )


class SkillLLMClient:
    """
    LLM client for SkillReducer operations.

    Supports OpenAI-compatible APIs including OpenAI, DeepSeek, and Qwen.
    Uses JSON response format for reliable content classification.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var).
            base_url: API base URL (defaults to OPENAI_BASE_URL env var).
                      Set to DeepSeek/Qwen endpoint for alternative providers.
            model: Model name (default: gpt-4o-mini).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client with optional base_url for alternative providers
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def classify_content_blocks(
        self,
        content_chunks: list[dict[str, str]],
        skill_context: Optional[str] = None
    ) -> BlockClassificationResult:
        """
        Classify content blocks into ContentType categories.

        Uses JSON response format for structured output. Works with OpenAI,
        DeepSeek, and other OpenAI-compatible APIs.

        Args:
            content_chunks: List of dicts with 'chunk_id' and 'content' keys.
            skill_context: Optional context about the skill (name, description).

        Returns:
            BlockClassificationResult with classifications for all chunks.

        Raises:
            ValueError: If content_chunks is empty.
            Exception: After 3 failed retries.
        """
        if not content_chunks:
            raise ValueError("content_chunks cannot be empty")

        # Build the classification prompt
        system_prompt = self._build_classification_system_prompt()
        user_prompt = self._build_classification_user_prompt(
            content_chunks, skill_context
        )

        # Use regular chat completion with JSON mode (more compatible)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        # Parse JSON response
        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        # Convert to Pydantic model
        return BlockClassificationResult(**result_dict)

    def _build_classification_system_prompt(self) -> str:
        """Build the system prompt for content classification."""
        return """You are a content classification expert for LLM agent skills.

Your task is to classify content blocks from skill documentation into one of five types:

1. **core_rule**: Actionable instructions that the agent must follow. These are directives,
   rules, workflows, or steps that guide agent behavior. Look for imperative statements,
   numbered steps, checklists, and explicit instructions.

2. **background**: Explanatory content, definitions, context, or educational material.
   This includes "what is" explanations, theoretical frameworks, and conceptual background
   that helps understand but doesn't directly guide action.

3. **example**: Code snippets, usage examples, sample outputs, or demonstration content.
   This includes fenced code blocks, example commands, and illustrative scenarios.

4. **template**: Boilerplate text, ready-to-use formats, fill-in-the-blank structures.
   This includes template strings with placeholders, standard formats, and reusable patterns.

5. **redundant**: Content that is duplicated elsewhere, provides no unique value, or is
   outdated/irrelevant. This includes repeated information and unnecessary verbosity.

Classification guidelines:
- When in doubt between core_rule and background, prefer core_rule if the content
  can guide agent action
- Examples within explanatory text are still background; only standalone examples
  are classified as example
- Templates must be directly usable as-is; partial templates are background
- Be conservative with redundant; only mark content that truly adds no value

IMPORTANT: You must respond with a JSON object in the following format:
{
  "classifications": [
    {
      "chunk_id": "<chunk_id from input>",
      "content_type": "<one of: core_rule, background, example, template, redundant>",
      "reasoning": "<brief explanation>"
    }
  ]
}

Return classifications for each chunk provided."""

    def _build_classification_user_prompt(
        self,
        content_chunks: list[dict[str, str]],
        skill_context: Optional[str]
    ) -> str:
        """Build the user prompt with content to classify."""
        prompt_parts = []

        if skill_context:
            prompt_parts.append(f"Skill Context:\n{skill_context}\n")

        prompt_parts.append("Please classify the following content chunks:\n")

        for chunk in content_chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            content = chunk.get("content", "")
            prompt_parts.append(f"\n[CHUNK_ID: {chunk_id}]\n{content}\n")

        prompt_parts.append(
            "\nReturn a classification for each chunk with the chunk_id, "
            "content_type, and brief reasoning."
        )

        return "\n".join(prompt_parts)

    def test_connection(self) -> bool:
        """
        Test the API connection.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return bool(response.choices)
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
