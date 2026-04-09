from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── TRUE CONSTRAINTS (hard limits) ──
    MAX_CONTEXT_TOKENS_PER_CALL: int = 3000  # hard cap on input tokens per LLM call
    MAX_COST_PER_SESSION: float = 0.10       # USD — hard budget ceiling

    # ── ORCHESTRATION PARAMETERS (adaptive, derived from constraints) ──
    # These are UPPER BOUNDS — the orchestrator may use fewer based on budget
    MAX_SUB_QUERIES: int = 8        # ceiling, orchestrator decides actual count
    EPISODIC_BUFFER_SIZE: int = 10  # max buffer, compression triggers adaptively
    VECTOR_TOP_K: int = 5           # retrieve top K from vector store

    # Model routing — heavy model for synthesis, light model for routine tasks
    LLM_MODEL: str = "claude-sonnet-4-20250514"       # synthesis (quality matters)
    LLM_MODEL_LIGHT: str = "claude-haiku-4-5-20251001"  # decompose, compress, critique

    # Pricing per model (per 1K tokens)
    SONNET_INPUT_COST_PER_1K: float = 0.003
    SONNET_OUTPUT_COST_PER_1K: float = 0.015
    HAIKU_INPUT_COST_PER_1K: float = 0.0008
    HAIKU_OUTPUT_COST_PER_1K: float = 0.004

    # API Keys
    ANTHROPIC_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    model_config = {"env_file": ".env"}


settings = Settings()
