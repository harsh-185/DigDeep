"""
Adaptive orchestrator — derives runtime parameters from budget state.

The only TRUE constraints are:
  - MAX_COST_PER_SESSION ($0.10)
  - MAX_CONTEXT_TOKENS_PER_CALL (3000)

Everything else (sub-query count, buffer behavior, critique decision)
is decided here based on how much budget remains.
"""

from agent.config import settings
from agent.memory.cost_tracker import CostTracker

# Estimated cost per pipeline phase (from measured runs)
EST_DECOMPOSE = 0.001   # ~$0.001 per decomposition
EST_COMPRESS = 0.008    # ~$0.008 per batch compression
EST_SYNTHESIZE = 0.015  # ~$0.015 per synthesis
EST_CRITIQUE = 0.002    # ~$0.002 per critique call
EST_PER_GAP_SEARCH_COMPRESS = 0.010  # gap search + compress


def plan_research(cost_tracker: CostTracker) -> dict:
    """
    Given current budget state, decide how to orchestrate the pipeline.

    Returns a plan dict with:
      - max_sub_queries: how many sub-queries to allow
      - enable_critique: whether self-critique loop is affordable
      - max_gap_queries: how many gap-filling searches if critique runs
      - buffer_size: effective episodic buffer size
      - tavily_results: results per sub-query
    """
    remaining = settings.MAX_COST_PER_SESSION - cost_tracker.total_cost

    # Reserve budget for mandatory phases
    reserved = EST_COMPRESS + EST_SYNTHESIZE  # must have compression + final synthesis
    available = remaining - reserved

    if available <= 0:
        return {
            "max_sub_queries": 1,
            "enable_critique": False,
            "max_gap_queries": 0,
            "buffer_size": settings.EPISODIC_BUFFER_SIZE,
            "tavily_results": 3,
            "reason": "Budget nearly exhausted — minimal pipeline",
        }

    # Decomposition costs ~$0.001 regardless of sub-query count
    available -= EST_DECOMPOSE

    # Decide if we can afford the critique loop
    critique_cost = EST_CRITIQUE + EST_PER_GAP_SEARCH_COMPRESS + EST_SYNTHESIZE
    enable_critique = available > (critique_cost + EST_SYNTHESIZE)

    if enable_critique:
        # Reserve budget for critique loop
        available -= critique_cost
        max_gap_queries = min(4, max(1, int(available / 0.003)))
    else:
        max_gap_queries = 0

    # Sub-query count: Tavily is free, but more sub-queries = more findings
    # = more evictions = potentially more compression cost
    # Heuristic: each sub-query adds ~5 findings, buffer overflow triggers compression
    # With buffer_size 10, overflow starts at sub-query 3-4 (3 results each)
    if available > 0.04:
        max_sub_queries = min(8, settings.MAX_SUB_QUERIES)
        tavily_results = 5
    elif available > 0.02:
        max_sub_queries = min(6, settings.MAX_SUB_QUERIES)
        tavily_results = 4
    elif available > 0.01:
        max_sub_queries = min(4, settings.MAX_SUB_QUERIES)
        tavily_results = 3
    else:
        max_sub_queries = 2
        tavily_results = 3

    # Buffer size: larger when we have budget for compression
    if available > 0.03:
        buffer_size = settings.EPISODIC_BUFFER_SIZE  # full buffer
    else:
        buffer_size = max(5, settings.EPISODIC_BUFFER_SIZE - 3)  # smaller = fewer evictions

    reason = (
        f"Budget ${remaining:.3f} remaining → "
        f"{max_sub_queries} sub-queries, "
        f"{'critique ON' if enable_critique else 'critique OFF'}"
        f"{f', {max_gap_queries} gap queries' if enable_critique else ''}"
    )

    return {
        "max_sub_queries": max_sub_queries,
        "enable_critique": enable_critique,
        "max_gap_queries": max_gap_queries,
        "buffer_size": buffer_size,
        "tavily_results": tavily_results,
        "reason": reason,
    }
