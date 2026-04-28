"""
Stage 1 Router Optimizer for skill description compression.

Implements the delta debugging approach from the SkillReducer paper for
optimizing the routing layer (skill descriptions).

Key improvements aligned with paper:
- Phase 1: DDMIN for minimal clause subset
- Phase 2: Real-world validation with selective restore
- TF-IDF based real distractors + 1 LLM-generated adversarial skill
- Pre-generation for missing/short descriptions
- Structured routing signals (primary capability, trigger condition, unique identifiers)
"""

import json
import os
import math
from typing import Optional
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import Skill, StructuredDescription, RoutingSignal
from .llm_client import SkillLLMClient
from .chunker import count_tokens


class SemanticClause(BaseModel):
    """A semantic clause extracted from skill description."""

    clause_id: str = Field(..., description="Unique identifier for the clause")
    content: str = Field(..., description="The clause text content")


class SegmentationResult(BaseModel):
    """Result of description segmentation into semantic clauses."""

    clauses: list[SemanticClause] = Field(
        ...,
        description="List of semantic clauses extracted from description"
    )


class RoutingTestResult(BaseModel):
    """Result of a routing test with simulated oracle."""

    success: bool = Field(..., description="Whether routing was successful")
    selected_skill: str = Field(..., description="The skill selected by the oracle")
    reasoning: Optional[str] = Field(
        default=None,
        description="Explanation of the routing decision"
    )


@dataclass
class CompressionResult:
    """Result of Stage 1 compression."""

    original_description: str
    compressed_description: str
    original_token_count: int
    compressed_token_count: int
    clauses_removed: int
    compression_ratio: float
    phase1_passed: bool = True
    phase2_passed: bool = True
    restored_clauses: list[str] = None


class SkillLibraryTFIDF:
    """
    TF-IDF index for the skill library.

    Used to select real distractors based on semantic similarity.
    """

    def __init__(self, skill_library_path: str = "Claude-Skills"):
        self.skill_library_path = skill_library_path
        self.skills_data: list[dict] = []  # [{"name": ..., "description": ...}]
        self.idf: dict[str, float] = {}
        self._initialized = False

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        text = text.lower()
        tokens = []
        current = ""
        for char in text:
            if char.isalnum():
                current += char
            else:
                if current:
                    tokens.append(current)
                    current = ""
        if current:
            tokens.append(current)
        return tokens

    def initialize(self):
        """Load all skills from library and compute IDF."""
        if self._initialized:
            return

        skill_files = []
        for root, dirs, files in os.walk(self.skill_library_path):
            if "SKILL.md" in files:
                skill_files.append(os.path.join(root, "SKILL.md"))

        all_docs = []
        for skill_file in skill_files:
            try:
                with open(skill_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse YAML frontmatter
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        import yaml
                        try:
                            frontmatter = yaml.safe_load(parts[1])
                            name = frontmatter.get('name', '')
                            description = frontmatter.get('description', '')
                            if isinstance(description, str):
                                self.skills_data.append({
                                    "name": name,
                                    "description": description,
                                    "file_path": skill_file
                                })
                                all_docs.append(self._tokenize(description))
                        except:
                            pass
            except Exception:
                continue

        # Compute IDF
        num_docs = len(all_docs)
        if num_docs > 0:
            doc_freq = Counter()
            for doc in all_docs:
                unique_terms = set(doc)
                for term in unique_terms:
                    doc_freq[term] += 1

            for term, df in doc_freq.items():
                self.idf[term] = math.log(num_docs / (1 + df))

        self._initialized = True

    def get_tfidf_vector(self, text: str) -> dict[str, float]:
        """Compute TF-IDF vector for a text."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1

        tfidf = {}
        for term, count in tf.items():
            idf = self.idf.get(term, math.log(len(self.skills_data) + 1))
            tfidf[term] = (count / total) * idf

        return tfidf

    def cosine_similarity(self, vec1: dict[str, float], vec2: dict[str, float]) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0

        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)

        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_real_distractors(self, target_skill: Skill, n: int = 4) -> list[dict]:
        """
        Get N most similar skills from the library as real distractors.

        Uses TF-IDF cosine similarity to find skills that are semantically
        similar but not identical.
        """
        self.initialize()

        target_vec = self.get_tfidf_vector(target_skill.description.original)

        similarities = []
        for skill_data in self.skills_data:
            if skill_data["name"] == target_skill.name:
                continue

            skill_vec = self.get_tfidf_vector(skill_data["description"])
            sim = self.cosine_similarity(target_vec, skill_vec)

            similarities.append({
                "name": skill_data["name"],
                "description": skill_data["description"],
                "similarity": sim
            })

        # Sort by similarity (highest first) and take top N
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return [
            {"name": s["name"], "description": s["description"]}
            for s in similarities[:n]
        ]


class Stage1Optimizer:
    """
    Stage 1 optimizer for routing layer optimization.

    Uses delta debugging (DDMIN) algorithm to compress skill descriptions
    while preserving routing capability.

    Two-phase validation:
    - Phase 1: DDMIN with simulated oracle
    - Phase 2: Real-world validation with selective restore
    """

    def __init__(
        self,
        llm_client: SkillLLMClient,
        oracle_model: Optional[str] = None,
        skill_library_path: str = "Claude-Skills",
        use_real_cli: bool = False,
        cli_path: str = "claude",
        cli_timeout: int = 60
    ):
        """
        Initialize the Stage 1 optimizer.

        Args:
            llm_client: LLM client for segmentation and oracle testing.
            oracle_model: Model to use for oracle (defaults to client's model).
            skill_library_path: Path to skill library for TF-IDF distractor selection.
            use_real_cli: Use real Claude Code CLI for Phase 2 validation (paper default: True).
            cli_path: Path to Claude Code CLI executable.
            cli_timeout: Timeout in seconds for CLI subprocess calls.
        """
        self.llm_client = llm_client
        self.oracle_model = oracle_model or llm_client.model
        self.tfidf_index = SkillLibraryTFIDF(skill_library_path)
        self.use_real_cli = use_real_cli
        self.cli_path = cli_path
        self.cli_timeout = cli_timeout
        self._cli_validator = None  # Lazy init

    # ============================================================
    # Pre-generation for missing/short descriptions
    # ============================================================

    def ensure_description(
        self,
        skill: Skill,
        min_tokens: int = 40,
        enable_structured: bool = False,
        validate_with_oracle: bool = True,
        max_retries: int = 3
    ) -> Skill:
        """
        Ensure skill has a valid description with minimum tokens.

        For missing or too-short descriptions, generates one from body.
        Per paper: "The generated description is validated through Phase 1's
        simulated oracle before acceptance."

        Args:
            skill: The skill to check/update.
            min_tokens: Minimum token count required.
            enable_structured: Generate structured routing signals per paper methodology.
            validate_with_oracle: Validate generated description via simulated oracle.
            max_retries: Maximum generation retries if validation fails.

        Returns:
            Updated Skill with valid description.
        """
        current_desc = skill.description.original
        current_tokens = count_tokens(current_desc) if current_desc else 0

        if current_desc and current_tokens > min_tokens:
            return skill

        # Generate description from body (with retries and oracle validation)
        for attempt in range(max_retries):
            if enable_structured:
                structured_desc = self.generate_structured_description(skill)
                if structured_desc:
                    generated_desc = structured_desc.combined_description
                else:
                    generated_desc = None
            else:
                generated_desc = self._generate_description_from_body(skill)

            if not generated_desc:
                return skill

            # Per paper: validate generated description through simulated oracle before acceptance
            if validate_with_oracle:
                # Get distractors for oracle validation
                try:
                    distractors = self.get_distractors(skill, num_real=4, num_adversarial=1)
                    # Temporarily set the generated description for oracle testing
                    temp_skill = skill.model_copy(deep=True)
                    temp_skill.description.original = generated_desc
                    temp_skill.description.original_token_count = count_tokens(generated_desc)

                    result = self.test_routing(
                        compressed_description=generated_desc,
                        target_skill=temp_skill,
                        distractors=distractors
                    )
                    if result.success:
                        # Oracle validates: accept the generated description
                        break
                    # Oracle rejects: retry generation
                except Exception:
                    # Oracle validation failed due to error - accept the description anyway
                    break
            else:
                break
        # After all retries, use the last generated description (even if oracle rejected it,
        # as it's better than no description)

        if enable_structured and generated_desc:
            skill.description.structured = structured_desc
            skill.description.original = structured_desc.combined_description
            skill.description.original_token_count = structured_desc.total_tokens
        elif generated_desc:
            skill.description.original = generated_desc
            skill.description.original_token_count = count_tokens(generated_desc)

        return skill

    def _generate_description_from_body(self, skill: Skill) -> Optional[str]:
        """Generate a description from the skill body content."""
        if not skill.body.original or not skill.body.original.strip():
            return None

        system_prompt = """You are an expert at writing concise skill descriptions for routing.

Your task is to generate a brief, informative description for a skill based on its body content.

Guidelines:
1. The description should be 2-3 sentences (40-80 tokens)
2. Focus on WHAT the skill does, not HOW
3. Include key capabilities and primary use cases
4. Be specific enough to distinguish from similar skills
5. Use clear, professional language

IMPORTANT: Respond with a JSON object:
{
  "description": "The generated description text"
}"""

        user_prompt = f"""Skill Name: {skill.name}
Category: {skill.metadata.category or 'general'}

Skill Body Content:
{skill.body.original[:3000]}

Generate a concise description for routing purposes."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        return result_dict.get("description")

    def generate_structured_description(
        self,
        skill: Skill
    ) -> Optional[StructuredDescription]:
        """
        Generate structured routing signals per SkillReducer paper.

        Per paper methodology, description generation produces three routing signals,
        each 20-40 tokens:
        - primary_capability: What the skill does
        - trigger_condition: When to invoke the skill
        - unique_identifiers: Libraries, APIs, frameworks referenced (max 3)

        Args:
            skill: The skill to generate description for.

        Returns:
            StructuredDescription with three routing signals.
        """
        if not skill.body.original or not skill.body.original.strip():
            return None

        system_prompt = """You are an expert at extracting routing signals from skill documentation.

Your task is to identify THREE distinct routing signals from the skill body, each signal
should be 20-40 tokens:

1. PRIMARY CAPABILITY: What does this skill do? Focus on the main function/purpose.
2. TRIGGER CONDITION: When should this skill be invoked? What scenarios require it?
3. UNIQUE IDENTIFIERS: What specific libraries, APIs, frameworks, or tools does this skill
   reference? These help distinguish it from similar skills.

Guidelines:
- Each signal should be concise but informative (20-40 tokens)
- Be specific - use actual names of tools/libraries when mentioned
- Avoid generic phrases like "helps with..." or "provides..."
- Focus on distinguishing features that help routing

IMPORTANT: Respond with a JSON object:
{
  "primary_capability": {
    "content": "What the skill does (20-40 tokens)",
    "token_count": 25
  },
  "trigger_condition": {
    "content": "When to invoke the skill (20-40 tokens)",
    "token_count": 30
  },
  "unique_identifiers": [
    {
      "signal_type": "unique_identifier",
      "content": "Library/API/framework name with context (20-40 tokens)",
      "token_count": 20
    }
  ],
  "combined_description": "Natural language description combining all signals"
}"""

        user_prompt = f"""Skill Name: {skill.name}
Category: {skill.metadata.category or 'general'}
Tags: {', '.join(skill.metadata.tags) if skill.metadata.tags else 'none'}

Skill Body Content (first 3000 chars):
{skill.body.original[:3000]}

Extract THREE routing signals: primary capability, trigger condition, and unique identifiers."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        try:
            # Build RoutingSignal objects
            primary_cap = RoutingSignal(
                signal_type="primary_capability",
                content=result_dict.get("primary_capability", {}).get("content", ""),
                token_count=result_dict.get("primary_capability", {}).get("token_count", 0)
            )

            trigger_cond = RoutingSignal(
                signal_type="trigger_condition",
                content=result_dict.get("trigger_condition", {}).get("content", ""),
                token_count=result_dict.get("trigger_condition", {}).get("token_count", 0)
            )

            unique_ids = []
            for uid in result_dict.get("unique_identifiers", [])[:3]:
                unique_ids.append(RoutingSignal(
                    signal_type="unique_identifier",
                    content=uid.get("content", ""),
                    token_count=uid.get("token_count", 0)
                ))

            combined = result_dict.get("combined_description", "")
            if not combined:
                # Combine signals if combined_description not provided
                combined = f"{primary_cap.content}. {trigger_cond.content}."
                if unique_ids:
                    combined += f" Uses: {', '.join([u.content for u in unique_ids])}."

            total_tokens = primary_cap.token_count + trigger_cond.token_count + sum(u.token_count for u in unique_ids)

            return StructuredDescription(
                primary_capability=primary_cap,
                trigger_condition=trigger_cond,
                unique_identifiers=unique_ids,
                combined_description=combined,
                total_tokens=total_tokens
            )

        except Exception:
            # Fallback to unstructured description
            fallback_desc = result_dict.get("description", result_dict.get("combined_description", ""))
            if fallback_desc:
                return StructuredDescription(
                    primary_capability=RoutingSignal(
                        signal_type="primary_capability",
                        content=fallback_desc,
                        token_count=40
                    ),
                    trigger_condition=RoutingSignal(
                        signal_type="trigger_condition",
                        content="",
                        token_count=0
                    ),
                    unique_identifiers=[],
                    combined_description=fallback_desc,
                    total_tokens=40
                )
            return None

    def validate_routing_signals(
        self,
        structured_desc: StructuredDescription
    ) -> bool:
        """
        Validate routing signals meet token requirements.

        Per paper, each signal should be 20-40 tokens.

        Args:
            structured_desc: Structured description to validate.

        Returns:
            True if signals meet requirements.
        """
        # Check primary capability (required, 20-40 tokens)
        primary_tokens = structured_desc.primary_capability.token_count
        if primary_tokens < 20 or primary_tokens > 40:
            # Allow some flexibility
            if primary_tokens < 10:
                return False

        # Check trigger condition (20-40 tokens)
        trigger_tokens = structured_desc.trigger_condition.token_count
        if trigger_tokens > 0 and (trigger_tokens < 10 or trigger_tokens > 50):
            return False

        # Unique identifiers optional but should be reasonable if present
        for uid in structured_desc.unique_identifiers:
            if uid.token_count > 50:
                return False

        return True

    def segment_description(
        self,
        description: str
    ) -> list[SemanticClause]:
        """
        Segment description into semantic clauses using LLM.

        Args:
            description: The skill description to segment.

        Returns:
            List of SemanticClause objects.
        """
        if not description or not description.strip():
            return []

        system_prompt = """You are an expert at analyzing and segmenting text into semantic clauses.

Your task is to split a skill description into independent semantic clauses (meaningful units that convey a complete thought or concept).

Guidelines:
1. Each clause should be a self-contained, meaningful unit
2. Clauses should be as independent as possible
3. Preserve the original wording - do not paraphrase
4. Split at natural boundaries (commas, conjunctions, sentence boundaries)
5. Each clause should contribute unique information about the skill

IMPORTANT: You must respond with a JSON object in the following format:
{
  "clauses": [
    {
      "clause_id": "clause_1",
      "content": "first semantic clause"
    },
    {
      "clause_id": "clause_2",
      "content": "second semantic clause"
    }
  ]
}"""

        user_prompt = f"""Please segment the following skill description into semantic clauses:

"{description}"

Return a JSON object with the list of clauses."""

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
        result_dict = json.loads(response_text)
        result = SegmentationResult(**result_dict)

        return result.clauses

    def generate_adversarial_skill(
        self,
        skill: Skill,
        num_candidates: int = 1
    ) -> list[dict]:
        """
        Generate adversarial skills (distractors) for routing test.

        Creates skills that are in the same domain but have different
        functionality, to test if the compressed description can still
        distinguish the target skill.

        Args:
            skill: The target skill.
            num_candidates: Number of adversarial skills to generate (default 1 per paper).

        Returns:
            List of adversarial skill dictionaries with name and description.
        """
        system_prompt = """You are an expert at creating realistic skill definitions.

Your task is to generate "adversarial skills" - skills that are in the same
domain or category as the target skill, but have DIFFERENT functionality.

These adversarial skills will be used as distractors in a routing test to
ensure the target skill's description is specific enough to be correctly selected.

Guidelines:
1. The adversarial skills should be plausible and realistic
2. They should be related to the target skill's domain
3. But they should have clearly different functionality/purpose
4. Names should be descriptive but different from the target
5. Descriptions should be similar in length and style

IMPORTANT: You must respond with a JSON object in the following format:
{
  "adversarial_skills": [
    {
      "name": "skill-name-1",
      "description": "Description of adversarial skill 1"
    },
    {
      "name": "skill-name-2",
      "description": "Description of adversarial skill 2"
    }
  ]
}"""

        context = f"""Target Skill:
- Name: {skill.name}
- Description: {skill.description.original}
- Category: {skill.metadata.category or 'general'}
- Domain: {skill.metadata.domain or 'general'}

Generate {num_candidates} adversarial skills that could be confused with this target skill."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        return result_dict.get("adversarial_skills", [])

    def get_distractors(
        self,
        skill: Skill,
        num_real: int = 4,
        num_adversarial: int = 1
    ) -> list[dict]:
        """
        Get combined distractors: TF-IDF real distractors + LLM adversarial.

        Per SkillReducer paper: 4 real distractors from skill library + 1 adversarial.

        Args:
            skill: The target skill.
            num_real: Number of real distractors from TF-IDF (default 4).
            num_adversarial: Number of LLM-generated adversarial skills (default 1).

        Returns:
            List of distractor skill dictionaries.
        """
        distractors = []

        # Get real distractors from skill library using TF-IDF
        real_distractors = self.tfidf_index.get_real_distractors(skill, n=num_real)
        distractors.extend(real_distractors)

        # Get LLM-generated adversarial skill
        adversarial = self.generate_adversarial_skill(skill, num_candidates=num_adversarial)
        distractors.extend(adversarial)

        return distractors

    def test_routing(
        self,
        compressed_description: str,
        target_skill: Skill,
        distractors: list[dict],
        query: Optional[str] = None,
        randomize_order: bool = True
    ) -> RoutingTestResult:
        """
        Simulated Oracle O_sim: Test if compressed description routes correctly.

        Given a compressed description, target skill, and distractor skills,
        tests whether the LLM can correctly route to the target skill.

        Per SkillReducer paper Section IV-A: "the order of candidates in C is
        randomized for each query" to ensure selection is based on semantic merit
        rather than positional bias.

        Args:
            compressed_description: The compressed description to test.
            target_skill: The target skill that should be selected.
            distractors: List of distractor skills (real + adversarial).
            query: Optional query that triggered the routing (generated if not provided).
            randomize_order: Whether to randomize candidate order (default True per paper).

        Returns:
            RoutingTestResult indicating success or failure.
        """
        import random

        # Generate a query if not provided
        if query is None:
            query = self._generate_routing_query(target_skill)

        # Build the routing prompt
        system_prompt = """You are a skill routing system. Your task is to select the most appropriate skill for a given user query.

You will be given:
1. A user query
2. A list of available skills with their names and descriptions

Select the ONE skill that best matches the user's needs based on the skill descriptions.

IMPORTANT: You must respond with a JSON object in the following format:
{
  "selected_skill": "exact-name-of-selected-skill",
  "reasoning": "Brief explanation of why this skill was selected"
}"""

        # Build skill list with target and distractors
        skills_list = []
        skills_list.append({
            "name": target_skill.name,
            "description": compressed_description
        })
        for adv in distractors:
            skills_list.append({
                "name": adv["name"],
                "description": adv["description"]
            })

        # Per paper: randomize order to avoid positional bias
        if randomize_order:
            random.shuffle(skills_list)

        # Format skills for prompt
        skills_text = "\n\n".join([
            f"Skill: {s['name']}\nDescription: {s['description']}"
            for s in skills_list
        ])

        user_prompt = f"""User Query:
{query}

Available Skills:
{skills_text}

Select the most appropriate skill for this query."""

        response = self.llm_client.client.chat.completions.create(
            model=self.oracle_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        selected = result_dict.get("selected_skill", "")
        success = selected == target_skill.name

        return RoutingTestResult(
            success=success,
            selected_skill=selected,
            reasoning=result_dict.get("reasoning")
        )

    def _generate_routing_query(self, skill: Skill) -> str:
        """Generate a typical query that should route to this skill."""
        system_prompt = """You are a user of an AI assistant system.

Generate a realistic, natural query that a user might ask when they need the
specific skill described. The query should:
1. Be in natural language (not too formal, not too casual)
2. Clearly indicate the need for this specific skill's capabilities
3. Be specific enough to distinguish from similar skills
4. Be realistic - something a real user would actually type

IMPORTANT: Respond with a JSON object:
{
  "query": "the generated query text"
}"""

        user_prompt = f"""Skill Name: {skill.name}
Skill Description: {skill.description.original}

Generate a realistic user query that would need this skill."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        return result_dict.get("query", f"Help me with {skill.name}")

    # ============================================================
    # Phase 1: DDMIN Algorithm
    # ============================================================

    def ddmin(
        self,
        clauses: list[SemanticClause],
        test_func: callable,
        n: int = 2
    ) -> list[SemanticClause]:
        """
        Delta Debugging (DDMIN) algorithm for 1-minimal subset.

        Canonical implementation from Zeller & Hildebrandt (2002):
        Phase 1 — Complement testing: try removing each subset.
        Phase 2 — Subset testing: test each subset alone.

        Iteratively refines until no further reduction is possible,
        yielding a 1-minimal set where every clause is individually
        necessary.

        Args:
            clauses: List of semantic clauses.
            test_func: Function that takes a list of clauses and returns bool (pass/fail).
            n: Initial number of subsets (default 2 for binary split).

        Returns:
            1-minimal subset of clauses that passes the test.
        """
        current = list(clauses)
        if len(current) <= 1:
            return current

        n = 2
        while len(current) >= 2:
            # Split current set into n roughly equal subsets
            subset_size = max(1, len(current) // n)
            subsets = []
            for i in range(0, len(current), subset_size):
                subsets.append(current[i:i + subset_size])

            found = False

            # Phase 1: Complement testing — try deleting each subset
            for i, subset_to_remove in enumerate(subsets):
                complement = []
                for j, subset in enumerate(subsets):
                    if j != i:
                        complement.extend(subset)
                if not complement:
                    continue
                if test_func(complement):
                    # Complement passes — reduce to complement,
                    # decrease granularity for finer search
                    current = complement
                    n = max(n - 1, 2)
                    found = True
                    break

            if not found:
                # Phase 2: Subset testing — test each subset alone
                if n < len(current):
                    for subset in subsets:
                        if test_func(subset):
                            # Subset alone is sufficient — reset to
                            # this subset, reset granularity
                            current = subset
                            n = 2
                            found = True
                            break

            if not found:
                # Neither phase reduced the set — increase granularity
                if n < len(current):
                    n = min(n * 2, len(current))
                else:
                    # At maximum granularity with no reduction possible:
                    # current is 1-minimal
                    break

        return current

    # ============================================================
    # Phase 2: Real-world Validation with Selective Restore
    # ============================================================

    def _get_cli_validator(self):
        """Lazy-init the CLI validator for Phase 2 real-environment validation."""
        if self._cli_validator is None:
            from .cli_validator import RealTriggerValidator
            self._cli_validator = RealTriggerValidator(
                cli_path=self.cli_path,
                timeout=self.cli_timeout
            )
        return self._cli_validator

    def phase2_validate_with_restore(
        self,
        d_fast: str,
        paraphrased_clauses: list[SemanticClause],
        original_clauses: list[SemanticClause],
        skill: Skill,
        distractors: list[dict],
        num_queries: int = 5,
        max_restore_steps: int = 3
    ) -> tuple[str, bool]:
        """
        Phase 2: Real-world validation with selective restore.

        Per SkillReducer paper Algorithm 1 (lines 7-19):
        1. Establish Qval: queries that trigger with the original description
        2. Test d_fast (polished compressed description) against all Qval
        3. If all pass, return d_fast immediately
        4. Deleted units D = U minus U* (original clauses not in DDMIN result)
        5. For each restore step (max 3): try each deleted unit individually
           with the paraphrased clause set, greedily add the best unit
        6. If all Qval trigger after a restore step, return JOIN(U*)
        7. If max steps reached, return current best (caller falls back to original)

        Args:
            d_fast: POLISH(JOIN(paraphrased_clauses)) — the polished candidate.
            paraphrased_clauses: Current U* after DDMIN + paraphrase.
            original_clauses: All original clauses from SEGMENT (before DDMIN).
            skill: The target skill.
            distractors: Distractor skills for routing test.
            num_queries: Number of diverse test queries to use.
            max_restore_steps: Maximum restore iterations (default 3 per paper).

        Returns:
            Tuple of (final_description, passed).
            - If d_fast passes: (d_fast, True)
            - If selective restore succeeds: (JOIN(restored_clauses), True)
            - If all fails: (best_description, False) — caller falls back to original
        """
        # Generate diverse test queries
        queries = self._generate_diverse_queries(skill, num_queries)

        # Compute D = U \ U* (original deleted clauses, not paraphrased)
        # Use clause_id for comparison since paraphrasing changes content text
        retained_ids = {c.clause_id for c in paraphrased_clauses}
        deleted_clauses = [c for c in original_clauses if c.clause_id not in retained_ids]

        if self.use_real_cli:
            cli = self._get_cli_validator()

            # Build original description for Qval baseline
            original_desc = " ".join(c.content for c in original_clauses)

            # Paper line 7: Qval ← {q ∈ Q | REALTRIGGER(q, s.d) = 1}
            Qval = []
            for q in queries:
                result = cli.validate_trigger(
                    skill_name=skill.name,
                    compressed_description=original_desc,
                    query=q,
                    skill_content=skill.body.original[:2000] if skill.body.original else None
                )
                if result.triggered:
                    Qval.append(q)

            if not Qval:
                return d_fast, False

            def _test_description(desc: str) -> float:
                """Test a description string (d_fast or JOIN) against Qval via REALTRIGGER."""
                triggered = 0
                for q in Qval:
                    result = cli.validate_trigger(
                        skill_name=skill.name,
                        compressed_description=desc,
                        query=q,
                        skill_content=skill.body.original[:2000] if skill.body.original else None
                    )
                    if result.triggered:
                        triggered += 1
                return triggered / len(Qval)
        else:
            Qval = list(queries)

            def _test_description(desc: str) -> float:
                """Test a description string against Qval via simulated oracle."""
                triggered = 0
                for q in Qval:
                    result = self.test_routing(
                        compressed_description=desc,
                        target_skill=skill,
                        distractors=distractors,
                        query=q
                    )
                    if result.success:
                        triggered += 1
                return triggered / len(Qval)

        # Per Algorithm 1 line 8: test if d_fast passes all Qval queries
        if _test_description(d_fast) == 1.0:
            return d_fast, True

        # Per Algorithm 1 lines 12-18: greedy selective restore (max 3 steps)
        current_clauses = list(paraphrased_clauses)

        for _ in range(max_restore_steps):
            if not deleted_clauses:
                break

            # Try each deleted unit, pick the one that maximizes trigger rate
            best_unit: Optional[SemanticClause] = None
            best_rate = _test_description(
                " ".join(c.content for c in current_clauses)
            )

            for unit in deleted_clauses:
                test_desc = " ".join(
                    c.content for c in (current_clauses + [unit])
                )
                rate = _test_description(test_desc)
                if rate > best_rate:
                    best_rate = rate
                    best_unit = unit

            if best_unit is None:
                break

            current_clauses.append(best_unit)
            deleted_clauses.remove(best_unit)

            # Per Algorithm 1 line 15: if all Qval trigger, return JOIN(U*)
            if best_rate == 1.0:
                return " ".join(c.content for c in current_clauses), True

        # Per Algorithm 1 line 19: restore failed
        best_desc = " ".join(c.content for c in current_clauses)
        return best_desc, False

    def _generate_diverse_queries(self, skill: Skill, num_queries: int) -> list[str]:
        """Generate diverse test queries for Phase 2 validation."""
        system_prompt = """You are a user of an AI assistant system.

Generate diverse, realistic queries that a user might ask when they need this skill.

Requirements:
1. Generate exactly the requested number of queries
2. Each query should be different in style/focus
3. Some should be direct, some indirect
4. Some should use technical terms, some should use lay language
5. All should realistically route to this skill

IMPORTANT: Respond with a JSON object:
{
  "queries": ["query 1", "query 2", ...]
}"""

        user_prompt = f"""Skill Name: {skill.name}
Skill Description: {skill.description.original}

Generate {num_queries} diverse user queries for this skill."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        return result_dict.get("queries", [f"Help me with {skill.name}"] * num_queries)

    def _paraphrase_clauses(
        self,
        clauses: list[SemanticClause],
        target_skill: Skill,
        distractors: list[dict],
        queries: Optional[list[str]] = None
    ) -> list[SemanticClause]:
        """
        Try shorter paraphrase for each clause, keeping if oracle passes.

        Per Algorithm 1 (lines 3-5): For each ui in U*, try a shorter
        paraphrase; keep if O_sim passes, otherwise keep the original.

        Uses the same fixed query set as DDMIN for oracle consistency.

        Args:
            clauses: The minimal clause subset from DDMIN.
            target_skill: The target skill for oracle testing.
            distractors: Distractor skills for oracle testing.
            queries: Fixed query set for oracle testing (same as DDMIN queries).

        Returns:
            List of clauses with shorter paraphrases where oracle passes.
        """
        paraphrased = []
        for clause in clauses:
            shorter = self._try_paraphrase_clause(clause)
            if shorter and shorter != clause.content:
                # Build test description: replace this clause with paraphrase
                remaining = [c for c in clauses if c.clause_id != clause.clause_id]
                test_desc = " ".join([shorter] + [c.content for c in remaining])

                # Per paper: keep if O_sim passes for ALL queries
                if queries:
                    passed = True
                    for q in queries:
                        result = self.test_routing(
                            compressed_description=test_desc,
                            target_skill=target_skill,
                            distractors=distractors,
                            query=q
                        )
                        if not result.success:
                            passed = False
                            break
                    if passed:
                        paraphrased.append(SemanticClause(
                            clause_id=clause.clause_id,
                            content=shorter
                        ))
                    else:
                        paraphrased.append(clause)
                else:
                    # Fallback: single random query (less strict)
                    result = self.test_routing(
                        compressed_description=test_desc,
                        target_skill=target_skill,
                        distractors=distractors
                    )
                    if result.success:
                        paraphrased.append(SemanticClause(
                            clause_id=clause.clause_id,
                            content=shorter
                        ))
                    else:
                        paraphrased.append(clause)
            else:
                paraphrased.append(clause)
        return paraphrased

    def _try_paraphrase_clause(
        self,
        clause: SemanticClause
    ) -> str:
        """
        Try to produce a shorter paraphrase of a single clause.

        Args:
            clause: The semantic clause to paraphrase.

        Returns:
            Shorter paraphrase string, or original if shortening fails.
        """
        if not clause.content or count_tokens(clause.content) <= 5:
            return clause.content

        system_prompt = """You are an expert at writing concise text.

Your task is to rewrite a single clause to be shorter while preserving ALL
semantic information. Remove unnecessary words but keep the exact meaning.

IMPORTANT: Respond with a JSON object:
{
  "shorter_version": "the shortened clause"
}"""

        user_prompt = f"""Original clause:
"{clause.content}"

Rewrite this clause to be shorter but preserve ALL meaning."""

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content
            result = json.loads(response_text)
            shorter = result.get("shorter_version", clause.content)

            # Only use if actually shorter
            if shorter and count_tokens(shorter) < count_tokens(clause.content):
                return shorter
            return clause.content

        except Exception:
            return clause.content

    def rewrite_and_polish(
        self,
        clauses: list[SemanticClause]
    ) -> str:
        """
        Rewrite and polish the selected clauses into a coherent description.

        Uses LLM to combine the minimal clauses into a fluent, natural
        description while preserving all semantic information.

        Args:
            clauses: List of semantic clauses to combine.

        Returns:
            Polished description string.
        """
        if not clauses:
            return ""

        if len(clauses) == 1:
            return clauses[0].content.strip()

        system_prompt = """You are an expert at writing clear, concise skill descriptions.

Your task is to combine multiple semantic clauses into a single, coherent,
and well-written description paragraph.

Guidelines:
1. Preserve ALL semantic information from the input clauses
2. Remove redundancy and improve flow
3. Use clear, professional language
4. The result should read as a natural description, not a list
5. Do not add any new information not present in the clauses
6. Keep the description concise but complete

IMPORTANT: Respond with a JSON object:
{
  "description": "the polished description"
}"""

        clauses_text = "\n".join([
            f"- {clause.content}"
            for clause in clauses
        ])

        user_prompt = f"""Combine the following semantic clauses into a coherent description:

{clauses_text}

Write a polished, professional skill description."""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        return result_dict.get("description", " ".join(c.content for c in clauses))

    def compress_description(
        self,
        skill: Skill,
        use_oracle_validation: bool = True,
        enable_phase2: bool = True
    ) -> CompressionResult:
        """
        Full Stage 1 compression pipeline for a skill description.

        Steps:
        0. Ensure description exists (pre-generate if missing/short)
        1. Segment description into semantic clauses
        2. Get distractors (4 TF-IDF real + 1 LLM adversarial)
        3. Apply DDMIN (Phase 1) to find 1-minimal subset
        4. Phase 2: Real-world validation with selective restore
        5. Per-clause paraphrase: try shorter version for each clause (Alg 1 lines 3-5)
        6. Rewrite and polish the result

        Args:
            skill: The skill to compress.
            use_oracle_validation: Whether to use oracle validation in DDMIN.
            enable_phase2: Whether to run Phase 2 validation with selective restore.

        Returns:
            CompressionResult with original and compressed descriptions.
        """
        # Step 0: Ensure description has minimum tokens
        skill = self.ensure_description(skill, min_tokens=40)

        original_description = skill.description.original
        original_tokens = count_tokens(original_description)

        # Step 1: Segment description
        clauses = self.segment_description(original_description)

        if not clauses:
            return CompressionResult(
                original_description=original_description,
                compressed_description=original_description,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                clauses_removed=0,
                compression_ratio=0.0
            )

        # Step 2: Get distractors (4 TF-IDF real + 1 LLM adversarial)
        distractors = self.get_distractors(skill, num_real=4, num_adversarial=1)

        # Step 3: Define test function for DDMIN (Phase 1)
        # Per paper: the oracle O(d, Q, C) takes test queries Q = {q1, ..., qk}
        # and "returns 1 if and only if the routing model consistently selects
        # the target skill for EVERY query in Q."
        # We pre-generate k=5 fixed queries to ensure monotonicity and cover
        # diverse routing scenarios.
        _ddmin_queries = self._generate_diverse_queries(skill, num_queries=5)

        def test_routing_with_clauses(clause_subset: list[SemanticClause]) -> bool:
            """Test if clause subset maintains routing capability for ALL queries."""
            if not clause_subset:
                return False

            # Combine clauses into temporary description
            temp_description = " ".join(c.content for c in clause_subset)

            # Per paper: oracle O(d, Q, C) returns 1 iff routing succeeds
            # for EVERY query in Q
            if use_oracle_validation:
                for query in _ddmin_queries:
                    result = self.test_routing(
                        compressed_description=temp_description,
                        target_skill=skill,
                        distractors=distractors,
                        query=query
                    )
                    if not result.success:
                        return False
                return True
            else:
                # Without oracle, always pass (not recommended)
                return True

        # Step 4: Apply DDMIN (Phase 1) — find 1-minimal subset U*
        minimal_clauses = self.ddmin(clauses, test_routing_with_clauses)
        phase1_passed = len(minimal_clauses) > 0

        # Step 5: Per-clause paraphrase (Algorithm 1 lines 3-5)
        # Paraphrase BEFORE Phase 2, per paper: each ui in U* is
        # shortened; keep if O_sim passes. Uses same fixed query set
        # as DDMIN for oracle consistency.
        if use_oracle_validation:
            paraphrased_clauses = self._paraphrase_clauses(
                clauses=minimal_clauses,
                target_skill=skill,
                distractors=distractors,
                queries=_ddmin_queries
            )
        else:
            paraphrased_clauses = minimal_clauses

        # Step 6: POLISH(JOIN(U*)) — Algorithm 1 line 6
        d_fast = self.rewrite_and_polish(paraphrased_clauses)

        # Step 7: Phase 2 — validate d_fast (Algorithm 1 lines 7-19)
        phase2_passed = True

        if enable_phase2 and use_oracle_validation:
            final_description, phase2_passed = self.phase2_validate_with_restore(
                d_fast=d_fast,
                paraphrased_clauses=paraphrased_clauses,
                original_clauses=clauses,
                skill=skill,
                distractors=distractors,
                num_queries=5
            )

            # Per Algorithm 1 line 19: fallback to original description if restore fails
            if not phase2_passed:
                return CompressionResult(
                    original_description=original_description,
                    compressed_description=original_description,
                    original_token_count=original_tokens,
                    compressed_token_count=original_tokens,
                    clauses_removed=0,
                    compression_ratio=0.0,
                    phase1_passed=phase1_passed,
                    phase2_passed=False,
                    restored_clauses=[]
                )

            compressed_description = final_description
        else:
            compressed_description = d_fast

        compressed_tokens = count_tokens(compressed_description)

        # Calculate metrics
        clauses_removed = len(clauses) - len(paraphrased_clauses)
        compression_ratio = 0.0
        if original_tokens > 0:
            compression_ratio = 1.0 - (compressed_tokens / original_tokens)

        return CompressionResult(
            original_description=original_description,
            compressed_description=compressed_description,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            clauses_removed=clauses_removed,
            compression_ratio=round(compression_ratio, 4),
            phase1_passed=phase1_passed,
            phase2_passed=phase2_passed,
            restored_clauses=[]
        )

    def compress_skill(
        self,
        skill: Skill,
        use_oracle_validation: bool = True,
        enable_phase2: bool = True
    ) -> Skill:
        """
        Compress a skill's description and return updated Skill object.

        Args:
            skill: The skill to compress.
            use_oracle_validation: Whether to use oracle validation.
            enable_phase2: Whether to run Phase 2 validation.

        Returns:
            Updated Skill object with compressed description.
        """
        result = self.compress_description(skill, use_oracle_validation, enable_phase2)

        # Update skill description
        skill.description.compressed = result.compressed_description
        skill.description.compressed_token_count = result.compressed_token_count

        return skill
