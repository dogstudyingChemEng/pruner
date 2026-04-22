"""
Stage 1 Router Optimizer for skill description compression.

Implements the delta debugging approach from the SkillReducer paper for
optimizing the routing layer (skill descriptions).
"""

import json
from typing import Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import Skill
from .llm_client import SkillLLMClient


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


class Stage1Optimizer:
    """
    Stage 1 optimizer for routing layer optimization.

    Uses delta debugging (DDMIN) algorithm to compress skill descriptions
    while preserving routing capability.
    """

    def __init__(
        self,
        llm_client: SkillLLMClient,
        oracle_model: Optional[str] = None
    ):
        """
        Initialize the Stage 1 optimizer.

        Args:
            llm_client: LLM client for segmentation and oracle testing.
            oracle_model: Model to use for oracle (defaults to client's model).
        """
        self.llm_client = llm_client
        self.oracle_model = oracle_model or llm_client.model

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
        num_candidates: int = 3
    ) -> list[dict]:
        """
        Generate adversarial skills (distractors) for routing test.

        Creates skills that are in the same domain but have different
        functionality, to test if the compressed description can still
        distinguish the target skill.

        Args:
            skill: The target skill.
            num_candidates: Number of adversarial skills to generate.

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

    def test_routing(
        self,
        compressed_description: str,
        target_skill: Skill,
        adversarial_skills: list[dict],
        query: Optional[str] = None
    ) -> RoutingTestResult:
        """
        Simulated Oracle O_sim: Test if compressed description routes correctly.

        Given a compressed description, target skill, and adversarial skills,
        tests whether the LLM can correctly route to the target skill.

        Args:
            compressed_description: The compressed description to test.
            target_skill: The target skill that should be selected.
            adversarial_skills: List of distractor skills.
            query: Optional query that triggered the routing (generated if not provided).

        Returns:
            RoutingTestResult indicating success or failure.
        """
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

        # Build skill list
        skills_list = []
        skills_list.append({
            "name": target_skill.name,
            "description": compressed_description
        })
        for adv in adversarial_skills:
            skills_list.append({
                "name": adv["name"],
                "description": adv["description"]
            })

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

    def ddmin(
        self,
        clauses: list[SemanticClause],
        test_func: callable,
        n: int = 2
    ) -> list[SemanticClause]:
        """
        Delta Debugging (DDMIN) algorithm for 1-minimal subset.

        Recursively divides clauses and tests subsets to find the minimal
        set that still passes the routing test.

        Args:
            clauses: List of semantic clauses.
            test_func: Function that takes a list of clauses and returns bool (pass/fail).
            n: Number of subsets to divide into (default 2 for binary split).

        Returns:
            1-minimal subset of clauses that passes the test.
        """
        if len(clauses) <= 1:
            return clauses

        # Split into n subsets
        subset_size = max(1, len(clauses) // n)
        subsets = []
        for i in range(0, len(clauses), subset_size):
            subsets.append(clauses[i:i + subset_size])

        # Try removing each subset
        for i, subset_to_remove in enumerate(subsets):
            # Create complement (all clauses except the subset to remove)
            complement = []
            for j, subset in enumerate(subsets):
                if j != i:
                    complement.extend(subset)

            if not complement:
                continue

            # Test if complement still passes
            if test_func(complement):
                # Complement passes, continue DDMIN on complement
                return self.ddmin(complement, test_func, n)

        # No subset can be removed, try increasing granularity
        if n < len(clauses):
            return self.ddmin(clauses, test_func, min(n * 2, len(clauses)))

        # Already at max granularity, return current set
        return clauses

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
        use_oracle_validation: bool = True
    ) -> CompressionResult:
        """
        Full Stage 1 compression pipeline for a skill description.

        Steps:
        1. Segment description into semantic clauses
        2. Generate adversarial skills for oracle testing
        3. Apply DDMIN to find 1-minimal subset
        4. Rewrite and polish the result

        Args:
            skill: The skill to compress.
            use_oracle_validation: Whether to use oracle validation in DDMIN.

        Returns:
            CompressionResult with original and compressed descriptions.
        """
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")

        original_description = skill.description.original
        original_tokens = len(encoding.encode(original_description))

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

        # Step 2: Generate adversarial skills
        adversarial_skills = self.generate_adversarial_skill(skill)

        # Step 3: Define test function for DDMIN
        def test_routing_with_clauses(clause_subset: list[SemanticClause]) -> bool:
            """Test if clause subset maintains routing capability."""
            if not clause_subset:
                return False

            # Combine clauses into temporary description
            temp_description = " ".join(c.content for c in clause_subset)

            # Test with oracle
            if use_oracle_validation:
                result = self.test_routing(
                    compressed_description=temp_description,
                    target_skill=skill,
                    adversarial_skills=adversarial_skills
                )
                return result.success
            else:
                # Without oracle, always pass (not recommended)
                return True

        # Step 4: Apply DDMIN
        minimal_clauses = self.ddmin(clauses, test_routing_with_clauses)

        # Step 5: Rewrite and polish
        compressed_description = self.rewrite_and_polish(minimal_clauses)
        compressed_tokens = len(encoding.encode(compressed_description))

        # Calculate metrics
        clauses_removed = len(clauses) - len(minimal_clauses)
        compression_ratio = 0.0
        if original_tokens > 0:
            compression_ratio = 1.0 - (compressed_tokens / original_tokens)

        return CompressionResult(
            original_description=original_description,
            compressed_description=compressed_description,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            clauses_removed=clauses_removed,
            compression_ratio=round(compression_ratio, 4)
        )

    def compress_skill(
        self,
        skill: Skill,
        use_oracle_validation: bool = True
    ) -> Skill:
        """
        Compress a skill's description and return updated Skill object.

        Args:
            skill: The skill to compress.
            use_oracle_validation: Whether to use oracle validation.

        Returns:
            Updated Skill object with compressed description.
        """
        result = self.compress_description(skill, use_oracle_validation)

        # Update skill description
        skill.description.compressed = result.compressed_description
        skill.description.compressed_token_count = result.compressed_token_count

        return skill
