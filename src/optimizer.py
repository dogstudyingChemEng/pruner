"""
Stage 2 Optimizer for skill body restructuring.

Implements the taxonomy-driven classification and progressive disclosure
optimization from the SkillReducer paper.
"""

import json
from typing import Optional
from dataclasses import dataclass

from .chunker import chunk_markdown_body, count_tokens
from .llm_client import BlockClassificationResult, SkillLLMClient
from .models import ContentBlock, ContentType, Skill, References, OnDemandModules


def _is_content_type(block: ContentBlock, content_type: ContentType) -> bool:
    """
    Check if a block's content_type matches the given type.

    Handles both enum and string values due to Pydantic's use_enum_values.
    """
    if isinstance(block.content_type, str):
        return block.content_type == content_type.value
    return block.content_type == content_type


@dataclass
class CompressionMetrics:
    """Metrics for Stage 2 compression results."""

    original_tokens: int
    compressed_tokens: int
    always_loaded_tokens: int  # Core rules - always in context
    on_demand_tokens: int  # Examples/Templates/Background - saved as references
    discarded_tokens: int  # Redundant - thrown away
    compression_ratio: float
    core_compression_ratio: float
    examples_deduped: int
    templates_deduped: int
    background_summarized: int
    references_deduped: int
    references_discarded: int
    # New fields for progressive disclosure
    on_demand_examples_tokens: int = 0
    on_demand_templates_tokens: int = 0
    on_demand_background_tokens: int = 0


class Stage2Optimizer:
    """
    Stage 2 optimizer for skill body restructuring.

    Uses LLM to classify content blocks into taxonomy types for
    progressive disclosure optimization, then applies compression
    to each content type.
    """

    def __init__(
        self,
        llm_client: SkillLLMClient,
        batch_size: int = 12
    ):
        """
        Initialize the Stage 2 optimizer.

        Args:
            llm_client: LLM client for content classification and compression.
            batch_size: Number of chunks to process per API call.
                       Default 12 to prevent context overflow.
        """
        self.llm_client = llm_client
        self.batch_size = batch_size

    # ============================================================
    # Classification Methods
    # ============================================================

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

    # ============================================================
    # Compression Methods (Post-processing)
    # ============================================================

    def compress_core_rules(
        self,
        blocks: list[ContentBlock]
    ) -> list[ContentBlock]:
        """
        Compress core rules by merging similar rules into bullet points.

        Uses LLM to identify similar rules and merge them into concise
        bullet points. Ensures compressed token count is less than original.

        Args:
            blocks: List of ContentBlock objects with core_rule type.

        Returns:
            List of compressed ContentBlock objects.
        """
        core_blocks = [b for b in blocks if _is_content_type(b, ContentType.CORE_RULE)]
        if not core_blocks:
            return []

        original_tokens = sum(b.token_count for b in core_blocks)

        # Build prompt for compression
        system_prompt = """You are an expert at compressing technical instructions into concise bullet points.

Your task is to take a list of rules/instructions and compress them by:
1. Merging similar or related rules into single bullet points
2. Removing redundancy while preserving all actionable information
3. Using concise, imperative language
4. Keeping each bullet point under 20 words when possible

CRITICAL REQUIREMENTS:
- The compressed version MUST have fewer tokens than the original
- Preserve ALL unique actionable instructions
- Do NOT add any information not present in the original
- Use bullet point format (• or - prefix)

IMPORTANT: Respond with a JSON object:
{
  "compressed_rules": [
    "• First compressed rule",
    "• Second compressed rule"
  ],
  "rules_merged": 2
}"""

        rules_text = "\n\n".join([
            f"[Rule {i+1}]\n{b.content}"
            for i, b in enumerate(core_blocks)
        ])

        user_prompt = f"""Compress the following rules into concise bullet points:

{rules_text}

Total original tokens: {original_tokens}

The compressed version must have fewer tokens while preserving all actionable instructions."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        compressed_rules = result.get("compressed_rules", [])
        if not compressed_rules:
            return core_blocks

        # Create single compressed block
        compressed_content = "\n".join(compressed_rules)
        compressed_tokens = count_tokens(compressed_content)

        # Ensure compression is effective
        if compressed_tokens >= original_tokens:
            return core_blocks

        return [ContentBlock(
            chunk_id="compressed_core_rules",
            content=compressed_content,
            content_type=ContentType.CORE_RULE,
            token_count=compressed_tokens
        )]

    def dedup_examples(
        self,
        blocks: list[ContentBlock]
    ) -> tuple[list[ContentBlock], int]:
        """
        Deduplicate examples by concept, keeping only one per concept.

        Groups examples by concept/functionality and keeps the most
        representative one. Strips redundant comments and boilerplate.

        Args:
            blocks: List of ContentBlock objects with example type.

        Returns:
            Tuple of (deduplicated blocks, number of examples removed).
        """
        example_blocks = [b for b in blocks if _is_content_type(b, ContentType.EXAMPLE)]
        if not example_blocks:
            return [], 0

        if len(example_blocks) == 1:
            return example_blocks, 0

        system_prompt = """You are an expert at analyzing and deduplicating code examples.

Your task is to:
1. Group examples by the concept/technique they demonstrate
2. For each concept group, select the SINGLE most representative example
3. Strip redundant comments and boilerplate code from the selected examples
4. Keep essential comments that explain non-obvious parts

Guidelines:
- Examples that demonstrate the same API/pattern/technique belong to the same concept
- Prefer examples that are self-contained and clear
- Remove: redundant imports, print statements for demo, excessive comments
- Keep: essential imports, the core logic, necessary context

IMPORTANT: Respond with a JSON object:
{
  "concept_groups": [
    {
      "concept": "concept_name",
      "examples_in_group": ["chunk_id_1", "chunk_id_2"],
      "selected_example_id": "chunk_id_1",
      "reason": "Why this example was selected"
    }
  ],
  "deduplicated_examples": [
    {
      "chunk_id": "original_chunk_id",
      "content": "stripped and cleaned example content"
    }
  ],
  "examples_removed": 3
}"""

        examples_text = "\n\n".join([
            f"[{b.chunk_id}]\n{b.content}"
            for b in example_blocks
        ])

        user_prompt = f"""Analyze and deduplicate the following examples:

{examples_text}

Group by concept and keep only the most representative example for each concept."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        deduped_examples = result.get("deduplicated_examples", [])
        examples_removed = result.get("examples_removed", 0)

        # Create new blocks from deduplicated examples
        new_blocks = []
        for ex in deduped_examples:
            chunk_id = ex.get("chunk_id", "")
            content = ex.get("content", "")
            if content:
                new_blocks.append(ContentBlock(
                    chunk_id=chunk_id or f"dedup_example_{len(new_blocks)}",
                    content=content,
                    content_type=ContentType.EXAMPLE,
                    token_count=count_tokens(content)
                ))

        return new_blocks, examples_removed

    def dedup_templates(
        self,
        blocks: list[ContentBlock]
    ) -> tuple[list[ContentBlock], int]:
        """
        Deduplicate templates by concept, keeping only one per concept.

        Similar to dedup_examples but for template content.

        Args:
            blocks: List of ContentBlock objects with template type.

        Returns:
            Tuple of (deduplicated blocks, number of templates removed).
        """
        template_blocks = [b for b in blocks if _is_content_type(b, ContentType.TEMPLATE)]
        if not template_blocks:
            return [], 0

        if len(template_blocks) == 1:
            return template_blocks, 0

        system_prompt = """You are an expert at analyzing and deduplicating templates.

Your task is to:
1. Group templates by their purpose/structure
2. For each group, select the SINGLE most representative template
3. Strip redundant boilerplate while keeping essential placeholders

Guidelines:
- Templates with the same purpose belong to the same group
- Prefer templates that are complete and well-structured
- Remove: redundant sections, excessive boilerplate
- Keep: essential placeholders, core structure

IMPORTANT: Respond with a JSON object:
{
  "concept_groups": [
    {
      "concept": "template_purpose",
      "templates_in_group": ["chunk_id_1", "chunk_id_2"],
      "selected_template_id": "chunk_id_1"
    }
  ],
  "deduplicated_templates": [
    {
      "chunk_id": "original_chunk_id",
      "content": "cleaned template content"
    }
  ],
  "templates_removed": 2
}"""

        templates_text = "\n\n".join([
            f"[{b.chunk_id}]\n{b.content}"
            for b in template_blocks
        ])

        user_prompt = f"""Analyze and deduplicate the following templates:

{templates_text}

Group by purpose and keep only the most representative template for each purpose."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        deduped_templates = result.get("deduplicated_templates", [])
        templates_removed = result.get("templates_removed", 0)

        # Create new blocks from deduplicated templates
        new_blocks = []
        for tmpl in deduped_templates:
            chunk_id = tmpl.get("chunk_id", "")
            content = tmpl.get("content", "")
            if content:
                new_blocks.append(ContentBlock(
                    chunk_id=chunk_id or f"dedup_template_{len(new_blocks)}",
                    content=content,
                    content_type=ContentType.TEMPLATE,
                    token_count=count_tokens(content)
                ))

        return new_blocks, templates_removed

    def summarize_background(
        self,
        blocks: list[ContentBlock]
    ) -> tuple[list[ContentBlock], int]:
        """
        Summarize background content into a single paragraph.

        Summarizes while preserving all factual information like numbers,
        thresholds, API endpoints, etc.

        Args:
            blocks: List of ContentBlock objects with background type.

        Returns:
            Tuple of (summarized blocks, number of blocks merged).
        """
        background_blocks = [b for b in blocks if _is_content_type(b, ContentType.BACKGROUND)]
        if not background_blocks:
            return [], 0

        if len(background_blocks) == 1:
            return background_blocks, 0

        system_prompt = """You are an expert at summarizing technical background content.

Your task is to combine multiple background/explanation paragraphs into a SINGLE
concise paragraph.

CRITICAL REQUIREMENTS:
- You MUST preserve ALL specific facts, including:
  * Numbers and percentages
  * Thresholds and limits
  * API endpoints and URLs
  * Version numbers
  * Configuration values
  * Command names and flags
- Do NOT invent any information not in the original
- Remove redundancy and improve flow
- Keep the summary under 200 words if possible

IMPORTANT: Respond with a JSON object:
{
  "summary": "The combined paragraph summary",
  "facts_preserved": ["list of key facts that were preserved"]
}"""

        background_text = "\n\n".join([
            f"[Background {i+1}]\n{b.content}"
            for i, b in enumerate(background_blocks)
        ])

        user_prompt = f"""Summarize the following background content into a single paragraph:

{background_text}

CRITICAL: Preserve all numbers, thresholds, API endpoints, and specific values exactly."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        summary = result.get("summary", "")
        if not summary:
            return background_blocks, 0

        summary_tokens = count_tokens(summary)

        return [ContentBlock(
            chunk_id="summarized_background",
            content=summary,
            content_type=ContentType.BACKGROUND,
            token_count=summary_tokens
        )], len(background_blocks) - 1

    def dedup_references(
        self,
        body_blocks: list[ContentBlock],
        references: References,
        min_token_threshold: int = 30
    ) -> tuple[References, int, int]:
        """
        Cross-file deduplication between body and references.

        Detects and removes content in references that duplicates rules
        already present in the body. Discards reference files that end
        up with fewer than min_token_threshold tokens.

        Args:
            body_blocks: Content blocks from the skill body.
            references: Reference files to deduplicate.
            min_token_threshold: Minimum tokens to keep a reference file.

        Returns:
            Tuple of (updated References, blocks deduped, files discarded).
        """
        if not references.files:
            return references, 0, 0

        # Extract core rules and key concepts from body
        core_rules_text = "\n".join([
            b.content for b in body_blocks
            if b.content_type == ContentType.CORE_RULE
        ])

        if not core_rules_text:
            return references, 0, 0

        new_files = {}
        total_deduped = 0
        files_discarded = 0

        for filename, content in references.files.items():
            system_prompt = """You are an expert at detecting duplicate content in technical documentation.

Your task is to analyze a reference file and remove any content that duplicates
information already present in the main skill body rules.

Guidelines:
- Remove paragraphs/sections that convey the same rules or information
- Keep content that provides ADDITIONAL information not in the body
- Preserve unique examples, edge cases, or detailed explanations
- Maintain document structure where possible

IMPORTANT: Respond with a JSON object:
{
  "deduplicated_content": "The reference content with duplicates removed",
  "duplicates_removed": 3,
  "unique_content_preserved": true
}"""

            user_prompt = f"""Main Body Rules (already present):
{core_rules_text[:3000]}  # Limit to avoid context overflow

Reference File Content:
{content}

Remove any content from the reference that duplicates the main body rules.
Keep only unique, additional information."""

            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content
            result = json.loads(response_text)

            deduped_content = result.get("deduplicated_content", content)
            duplicates_removed = result.get("duplicates_removed", 0)

            # Check if remaining content meets threshold
            remaining_tokens = count_tokens(deduped_content)

            if remaining_tokens < min_token_threshold:
                files_discarded += 1
            else:
                new_files[filename] = deduped_content
                total_deduped += duplicates_removed

        new_references = References(
            files=new_files,
            total_token_count=sum(
                count_tokens(c) for c in new_files.values()
            )
        )

        return new_references, total_deduped, files_discarded

    # ============================================================
    # Full Pipeline
    # ============================================================

    def optimize_skill(
        self,
        skill: Skill,
        compress_core: bool = True,
        dedup_examples: bool = True,
        dedup_templates: bool = True,
        summarize_background: bool = True,
        dedup_references: bool = True
    ) -> tuple[list[ContentBlock], OnDemandModules, References, CompressionMetrics]:
        """
        Full Stage 2 optimization pipeline with Progressive Disclosure.

        This method implements the Progressive Disclosure architecture from the SkillReducer paper:
        - Core rules (core_rule): Stay in main body - ALWAYS loaded
        - Examples (example): Saved as on-demand module - loaded when needed
        - Templates (template): Saved as on-demand module - loaded when needed
        - Background (background): Saved as on-demand module - loaded when needed
        - Redundant (redundant): DISCARDED - thrown away

        Steps:
        1. Classify body content into taxonomy types
        2. Compress core rules (stays in body)
        3. Deduplicate examples (saved as on-demand)
        4. Deduplicate templates (saved as on-demand)
        5. Summarize background (saved as on-demand)
        6. Cross-file deduplication with existing references

        Args:
            skill: The skill to optimize.
            compress_core: Whether to compress core rules.
            dedup_examples: Whether to deduplicate examples.
            dedup_templates: Whether to deduplicate templates.
            summarize_background: Whether to summarize background.
            dedup_references: Whether to deduplicate references.

        Returns:
            Tuple of:
            - core_blocks: Content blocks for main body (ONLY core_rule type)
            - on_demand_modules: OnDemandModules containing examples/templates/background
            - updated_references: Updated References (existing refs + on-demand refs)
            - compression_metrics: CompressionMetrics with detailed stats
        """
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")

        # Original tokens
        original_body_tokens = len(encoding.encode(skill.body.original))
        original_ref_tokens = sum(
            len(encoding.encode(c)) for c in skill.references.files.values()
        )
        total_original_tokens = original_body_tokens + original_ref_tokens

        # Step 1: Classify
        blocks = self.classify_skill_body(skill)

        # Track metrics
        core_compression_ratio = 0.0
        examples_removed = 0
        templates_removed = 0
        background_merged = 0
        ref_blocks_deduped = 0
        ref_files_discarded = 0

        # Separate blocks by type for processing
        core_blocks = [b for b in blocks if _is_content_type(b, ContentType.CORE_RULE)]
        example_blocks = [b for b in blocks if _is_content_type(b, ContentType.EXAMPLE)]
        template_blocks = [b for b in blocks if _is_content_type(b, ContentType.TEMPLATE)]
        background_blocks = [b for b in blocks if _is_content_type(b, ContentType.BACKGROUND)]
        redundant_blocks = [b for b in blocks if _is_content_type(b, ContentType.REDUNDANT)]

        # Step 2: Compress core rules (STAYS IN BODY)
        if compress_core and core_blocks:
            original_core_tokens = sum(b.token_count for b in core_blocks)
            core_blocks = self.compress_core_rules(core_blocks)
            new_core_tokens = sum(b.token_count for b in core_blocks)
            if original_core_tokens > 0:
                core_compression_ratio = 1.0 - (new_core_tokens / original_core_tokens)

        # Step 3: Deduplicate examples (SAVED AS ON-DEMAND MODULE)
        if dedup_examples and example_blocks:
            example_blocks, examples_removed = self.dedup_examples(example_blocks)

        # Step 4: Deduplicate templates (SAVED AS ON-DEMAND MODULE)
        if dedup_templates and template_blocks:
            template_blocks, templates_removed = self.dedup_templates(template_blocks)

        # Step 5: Summarize background (SAVED AS ON-DEMAND MODULE)
        if summarize_background and background_blocks:
            background_blocks, background_merged = self.summarize_background(background_blocks)

        # Create OnDemandModules for progressive disclosure
        on_demand_modules = OnDemandModules(
            examples=example_blocks,
            templates=template_blocks,
            background=background_blocks
        )

        # Step 6: Cross-file deduplication with existing references
        # Note: We dedupe against CORE RULES only, not on-demand modules
        updated_references = skill.references
        if dedup_references and skill.references.files:
            updated_references, ref_blocks_deduped, ref_files_discarded = self.dedup_references(
                core_blocks, skill.references  # Only core blocks for dedup
            )

        # Add on-demand modules to references
        on_demand_files = on_demand_modules.to_reference_files()
        all_reference_files = {**updated_references.files, **on_demand_files}

        # Calculate final metrics
        always_loaded_tokens = sum(b.token_count for b in core_blocks)
        on_demand_examples_tokens = sum(b.token_count for b in example_blocks)
        on_demand_templates_tokens = sum(b.token_count for b in template_blocks)
        on_demand_background_tokens = sum(b.token_count for b in background_blocks)
        on_demand_tokens = on_demand_examples_tokens + on_demand_templates_tokens + on_demand_background_tokens
        discarded_tokens = sum(b.token_count for b in redundant_blocks)

        # Total tokens in output (core + on-demand + references)
        total_reference_tokens = sum(count_tokens(c) for c in all_reference_files.values())
        total_compressed_tokens = always_loaded_tokens + total_reference_tokens

        compression_ratio = 0.0
        if total_original_tokens > 0:
            compression_ratio = 1.0 - (total_compressed_tokens / total_original_tokens)

        metrics = CompressionMetrics(
            original_tokens=total_original_tokens,
            compressed_tokens=total_compressed_tokens,
            always_loaded_tokens=always_loaded_tokens,
            on_demand_tokens=on_demand_tokens,
            discarded_tokens=discarded_tokens,
            compression_ratio=round(compression_ratio, 4),
            core_compression_ratio=round(core_compression_ratio, 4),
            examples_deduped=examples_removed,
            templates_deduped=templates_removed,
            background_summarized=background_merged,
            references_deduped=ref_blocks_deduped,
            references_discarded=ref_files_discarded,
            on_demand_examples_tokens=on_demand_examples_tokens,
            on_demand_templates_tokens=on_demand_templates_tokens,
            on_demand_background_tokens=on_demand_background_tokens
        )

        # Create final references with both original and on-demand files
        final_references = References(
            files=all_reference_files,
            total_token_count=total_reference_tokens
        )

        # Return ONLY core blocks for body, on-demand modules separately
        return core_blocks, on_demand_modules, final_references, metrics

    # ============================================================
    # Utility Methods
    # ============================================================

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
