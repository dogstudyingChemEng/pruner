"""
Configuration for SkillReducer pipeline.

Provides feature toggles for enabling/disabling paper methodology components.
All new features are disabled by default for backward compatibility.
"""

from pydantic import BaseModel, Field


class SkillReducerConfig(BaseModel):
    """
    Configuration for SkillReducer compression pipeline.

    Controls which paper methodology components are enabled.
    All new features disabled by default for backward compatibility.
    """

    # ============================================================
    # Stage 1: Routing Layer Optimization
    # ============================================================

    use_real_cli: bool = Field(
        default=False,
        description="Use real Claude Code CLI for Phase 2 validation (requires CLI installed)"
    )
    cli_path: str = Field(
        default="claude",
        description="Path to Claude Code CLI executable"
    )
    cli_timeout: int = Field(
        default=60,
        description="Timeout in seconds for CLI subprocess calls"
    )
    enable_structured_description: bool = Field(
        default=False,
        description="Generate structured routing signals (primary capability, trigger condition, unique identifiers)"
    )

    # ============================================================
    # Stage 2: Body Restructuring
    # ============================================================

    compress_core: bool = Field(
        default=True,
        description="Compress core rules after classification"
    )
    dedup_examples: bool = Field(
        default=True,
        description="Deduplicate examples (keep 1 per concept)"
    )
    dedup_templates: bool = Field(
        default=True,
        description="Deduplicate templates"
    )
    summarize_background: bool = Field(
        default=True,
        description="Summarize background content"
    )
    dedup_references: bool = Field(
        default=True,
        description="Cross-file deduplication with existing references"
    )

    # ============================================================
    # Quality Gates
    # ============================================================

    enable_gate1: bool = Field(
        default=True,
        description="Enable Gate 1: Faithfulness verification"
    )
    enable_gate2: bool = Field(
        default=True,
        description="Enable Gate 2: Task-based evaluation with feedback loop"
    )
    faithfulness_threshold: float = Field(
        default=1.0,
        description="Minimum ratio of preserved concepts for Gate 1"
    )
    max_loop_iterations: int = Field(
        default=2,
        description="Maximum iterations for Gate 2 feedback loop"
    )

    # ============================================================
    # Three Condition Evaluation (Paper Gate 2)
    # ============================================================

    use_three_conditions: bool = Field(
        default=False,
        description="Enable three-condition evaluation (D/A/C) for Gate 2"
    )
    retention_threshold: float = Field(
        default=1.0,
        description="Minimum retention threshold (score_C / score_A)"
    )
    enable_read_file_tool: bool = Field(
        default=True,
        description="Enable read_file tool simulation for Condition C"
    )
    max_tool_calls: int = Field(
        default=6,
        description="Maximum read_file tool calls per task"
    )

    # ============================================================
    # Hybrid Scoring (Paper Gate 2)
    # ============================================================

    use_hybrid_scoring: bool = Field(
        default=False,
        description="Enable hybrid scoring (pytest + LLM judge)"
    )
    pytest_weight: float = Field(
        default=0.523,
        description="Weight for pytest score in hybrid evaluation"
    )
    llm_judge_weight: float = Field(
        default=0.477,
        description="Weight for LLM judge score in hybrid evaluation"
    )
    pytest_timeout: int = Field(
        default=30,
        description="Timeout in seconds for pytest execution"
    )

    # ============================================================
    # Model Separation (Paper Section V-A)
    # ============================================================

    evaluator_model: str = Field(
        default="",
        description="Separate model for evaluation. Per paper, compression and "
                    "evaluation use different models to prevent information leakage. "
                    "Leave empty to use the same model (warning will be logged)."
    )
    evaluator_api_key: str = Field(
        default="",
        description="API key for evaluator model. Uses OPENAI_API_KEY if empty."
    )
    evaluator_base_url: str = Field(
        default="",
        description="Base URL for evaluator model API. Uses OPENAI_BASE_URL if empty."
    )

    # ============================================================
    # Output and Logging
    # ============================================================

    output_dir: str = Field(
        default="./output",
        description="Output directory for compressed skills"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )
    skip_errors: bool = Field(
        default=False,
        description="Skip errors and continue processing"
    )

    def get_stage1_kwargs(self) -> dict:
        """Get kwargs for Stage1Optimizer initialization."""
        return {
            "use_real_cli": self.use_real_cli,
            "cli_path": self.cli_path,
            "cli_timeout": self.cli_timeout,
            "enable_structured_description": self.enable_structured_description,
        }

    def get_stage2_kwargs(self) -> dict:
        """Get kwargs for Stage2Optimizer.optimize_skill()."""
        return {
            "compress_core": self.compress_core,
            "dedup_examples": self.dedup_examples,
            "dedup_templates": self.dedup_templates,
            "summarize_background": self.summarize_background,
            "dedup_references": self.dedup_references,
        }

    def get_quality_gates_kwargs(self) -> dict:
        """Get kwargs for QualityGates initialization."""
        return {
            "faithfulness_threshold": self.faithfulness_threshold,
            "max_loop_iterations": self.max_loop_iterations,
            "use_three_conditions": self.use_three_conditions,
            "use_hybrid_scoring": self.use_hybrid_scoring,
            "pytest_weight": self.pytest_weight,
            "llm_judge_weight": self.llm_judge_weight,
            "enable_read_file_tool": self.enable_read_file_tool,
            "max_tool_calls": self.max_tool_calls,
        }


# Default configuration instance
DEFAULT_CONFIG = SkillReducerConfig()


def get_paper_config() -> SkillReducerConfig:
    """
    Get configuration fully aligned with paper methodology.

    Enables all paper-specific features:
    - Real CLI validation
    - Structured descriptions
    - Three-condition evaluation
    - Hybrid scoring
    - read_file tool simulation

    Note: Per paper Section V-A, compression (DeepSeek-V3) and evaluation
    (Qwen3.5) should use separate models to prevent information leakage.
    Set evaluator_model and evaluator_api_key accordingly.
    """
    return SkillReducerConfig(
        use_real_cli=True,
        enable_structured_description=True,
        use_three_conditions=True,
        use_hybrid_scoring=True,
        enable_read_file_tool=True,
    )


def get_fast_config() -> SkillReducerConfig:
    """
    Get configuration for fast processing without expensive features.

    Disables:
    - Real CLI validation (uses simulated oracle)
    - Three-condition evaluation
    - Hybrid scoring

    Suitable for batch processing large skill libraries.
    """
    return SkillReducerConfig(
        use_real_cli=False,
        use_three_conditions=False,
        use_hybrid_scoring=False,
        enable_read_file_tool=False,
        verbose=False,
        skip_errors=True,
    )