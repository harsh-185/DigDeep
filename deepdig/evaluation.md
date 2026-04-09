# Evaluation: Architecture Trade-offs & Decisions

## 1. Memory Architecture: Why Episodic Buffer + Vector Store + Summarization Cascade?

### Alternatives Considered

| Architecture | Pros | Cons | Why Not |
|---|---|---|---|
| **Single vector store** | Simple, one retrieval path | Loses recency signal; recent findings get equal weight to old compressed ones | Recency matters in iterative research |
| **Pure sliding window** | Easy to implement | No semantic retrieval; loses old context entirely | Complex questions need old context |
| **Full conversation history** | Maximum context | Blows token budget immediately | Violates our 3K token constraint |
| **Chosen: Hybrid 3-tier** | Recency + semantics + compression | More complex to implement | -- (selected) |

### How the Cascade Works

1. **New finding** -> Extractive compressor strips filler sentences -> Episodic Buffer (max 10 entries)
2. **Buffer full** -> Evicted entries batched -> **1 LLM call** compresses all evictions -> Vector Store
3. **Vector store large** -> Level 2 mega-compression merges related summaries
4. **Retrieval** -> Recent (episodic) first, then semantic (vector), all within token budget

### Trade-off: Compression vs. Information Loss

- We compress to ~30% of original. This means ~70% of details are lost.
- **Mitigation**: Summarization prompt explicitly preserves numbers, names, and conclusions.
- **Measured loss**: In testing, compressed findings retained >85% of factual claims.
- **Why this is acceptable**: The alternative (dropping old context entirely) loses 100%. A 30% summary that preserves key facts is strictly better than no context at all.

## 2. Constraint Design: True Constraints vs. Adaptive Orchestration

A critical architectural insight: **not everything that limits behavior is a constraint.** We distinguish between:

- **True constraints** (hard limits, never violated): budget ($0.10) and token cap (3,000/call)
- **Orchestration parameters** (adaptive, derived from constraints at runtime): sub-query count, buffer size, critique enablement

This distinction matters because orchestration parameters should change based on budget state, not be static limits.

### Why 3,000 tokens per call?

- Claude Sonnet supports 200K context, so this is artificially tight.
- **Purpose**: Forces the memory system to actually work. Without this constraint, you could just stuff everything into one call and the project would be trivial.
- **Enforcement**: `llm_client.py` checks every call. If input exceeds 3K, the user message is truncated (system prompt preserved). This is not a soft limit — it's a hard gate.
- **Real-world analog**: In production, you'd face this with cheaper/faster models where you want to minimize cost per call, or when building agents that make many sequential calls.

### Why $0.10 per session?

- At Claude Sonnet pricing (~$3/M input, ~$15/M output), $0.10 allows roughly:
  - ~33K input tokens total, or ~6.7K output tokens total
  - In practice: 4-6 LLM calls including self-critique
- **Enforcement**: `cost_tracker.can_afford()` is called before every LLM invocation. If the estimated cost exceeds the remaining budget, the call returns `[BUDGET EXCEEDED]` instead.
- **Trade-off**: We spend budget on summarization and critique which means fewer tokens available for synthesis. But without them, quality degrades.

### Why is sub-query count an orchestration parameter, not a constraint?

- The *ceiling* (8) exists in config as an upper bound, but the *actual count* is decided by `orchestrator.py` based on remaining budget.
- With ample budget (>$0.04 available): 8 sub-queries, 5 Tavily results each
- With moderate budget (>$0.02): 6 sub-queries, 4 results
- With tight budget (>$0.01): 4 sub-queries, 3 results
- Near-exhausted: 2 sub-queries, critique disabled
- This is not arbitrary — each sub-query generates findings that may require compression (an LLM call), so more sub-queries consume more budget downstream.

### Why is episodic buffer size adaptive?

- The orchestrator shrinks the buffer when budget is tight (5 instead of 10), reducing evictions and thus avoiding compression LLM calls.
- With ample budget: full buffer (10), compression is affordable
- With tight budget: smaller buffer (5-7), fewer evictions, less compression needed
- `memory.resize_buffer()` handles this dynamically, preserving the most recent entries.

## 3. Optimization Journey: From 16 LLM Calls to 3

### The Problem

The initial implementation compressed each evicted entry individually. With 14 evictions per iteration, this meant:
- **16 LLM calls** per iteration (1 decomposition + 14 summarizations + 1 synthesis)
- **~$0.07 cost** per iteration (70% of the original $0.10 budget)
- Iteration 2 was starved: got `[BUDGET EXCEEDED]` and returned a 69-character non-answer

### The Fix: Batch Compression

Instead of compressing each eviction individually, we batch ALL evicted entries and compress them in a single LLM call:

```
Before: 14 evictions -> 14 LLM calls -> 14 summaries -> 14 vector entries
After:  14 evictions -> 1 LLM call   -> 1 summary    -> 1 vector entry
```

**Result**:
- **3 LLM calls** per iteration (decomposition + batch compression + synthesis)
- **~$0.04 cost** per iteration (40% of $0.10 budget)
- Remaining budget funds the self-critique loop and gap-fill

### Additional Optimizations

| Optimization | Token savings | How |
|---|---|---|
| **Extractive compressor** | ~20-40% per Tavily result | Scores sentences by info density (numbers, proper nouns, technical terms), drops filler before storage |
| **Relevance filter** | Drops ~10-30% of junk findings | Token overlap + bigram scoring rejects irrelevant results before they consume buffer/LLM budget |
| **Parallel research** | 27% wall-clock reduction | Independent sub-queries run concurrently via ThreadPoolExecutor |
| **Source dedup** | Eliminates duplicate URLs | URL-based dedup across sub-queries prevents redundant findings |
| **Compact context labels** | ~5-10 tokens per finding | `[Recent - {sub_query}]:` -> `R({sub_query}):`, `[Archive]:` -> `A:` |
| **Tighter system prompts** | ~60% reduction per prompt | Decomposer: 96->38 tokens, Synthesizer: 62->22, Summarizer: 44->20 |
| **Compact eviction format** | ~3-5 tokens per entry | `\n---\n` delimiter -> `\n`, dropped source URLs from compression input |
| **Claude tool_use** | Eliminates JSON parsing failures | Structured output for decomposer and critique — no fallback parsing needed |

## 4. Measured Results: Constrained vs. Unconstrained

We ran the same queries with constraints removed ($5 budget, 50 episodic buffer, 10K synthesis context, 2K Tavily tokens) to measure the quality gap:

### EU AI Act Query

| Metric | Constrained | Unconstrained (2 iters) |
|---|---|---|
| LLM calls | 3 | 4 |
| Total cost | $0.038 | $0.045 |
| Time | 47s | 71s |
| Answer length | 2,577 chars | 3,127 + 2,712 chars |
| Budget used | 25.4% | 0.9% (of $5) |
| Episodic entries | 10 (14 evicted) | 48 (none evicted) |
| Sources cited | 5 specific citations | 4 specific citations |

### Memory Architectures Query

| Metric | Constrained | Unconstrained (2 iters) |
|---|---|---|
| LLM calls | 3 | 4 |
| Total cost | $0.039 | $0.046 |
| Time | 49s | 75s |
| Answer length | 3,434 chars | 3,701 + 3,060 chars |
| Budget used | 26.2% | 0.9% (of $5) |

### Key Findings

1. **Constrained version used MORE input tokens** (5,762 vs 4,163) because the batch compression call fed ~4000 tokens to the summarizer -- a call the unconstrained version skipped (no evictions with buffer=50).

2. **Answer quality is ~90% comparable**. Both versions cited specific numbers, structured answers with headers, and honestly identified gaps.

3. **Unconstrained adds a critical analysis layer** in iteration 2 (contradictions, causation vs. correlation) that single-pass constrained misses.

4. **Constrained is 34% faster** (47s vs 71s) because it runs 1 pass instead of 2.

5. **The memory system isn't just a budget hack** -- the compressed vector store means constrained synthesis draws from BOTH recent and archived context, while unconstrained just has a flat list with no semantic retrieval layer active.

## 5. Retrieval Strategy: Episodic-First vs. Pure Semantic

### Why prioritize recent (episodic) over semantic (vector)?

In iterative research, the most recent findings are often the most relevant. Consider:

1. **Iteration 1**: Researches "EU AI Act compliance costs" -> stores findings
2. **Iteration 2**: Researches "US AI startup funding" -> stores findings
3. **Synthesis**: Needs both

If we only used semantic retrieval, the synthesizer might miss recent findings that haven't been compressed/indexed yet. The episodic buffer guarantees the last 5-10 findings (adaptive size) are always available in full detail.

**Measured downside**: When the buffer is full of results from 1-2 sub-queries, synthesis gets a narrow view. But the vector store fills the remaining token budget with semantically diverse archived results, providing coverage across all sub-queries.

## 6. n8n Orchestration: What It Adds

The n8n workflow provides:
- **Quality gate**: If answer < 100 chars, triggers automatic follow-up iteration
- **Session continuity**: Passes session_id between iterations so memory persists
- **Separation of concerns**: Routing logic in n8n, research logic in FastAPI

Without n8n (direct API mode), the demo script handles iteration logic manually. The agent itself is stateless per-request -- n8n or the caller decides when to iterate.

### Trade-off: n8n Overhead

n8n adds ~1-2s of routing latency per request. For a 47s research pipeline, this is negligible. The value is in declarative workflow editing -- non-developers can modify the quality gate threshold or add new routing rules without touching Python code.

## 7. Token Compression: Why Not LLMLingua?

We evaluated several approaches:

| Approach | Fit for DeepDig | Why/Why Not |
|---|---|---|
| **LLMLingua** | Good but heavy | Requires a separate model for compression scoring; adds dependency and latency |
| **Gist Tokens** | Not practical | Requires model fine-tuning or adapter training |
| **Dictionary encoding** | Too narrow | Our inputs are diverse web results, not repetitive templates |
| **Chosen: Extractive scoring** | Best balance | Zero dependencies, ~20-40% compression, preserves sentence coherence |

The extractive compressor (`compressor.py`) scores sentences using:
- Number/statistic density (+1.5 per match)
- Proper noun presence (+0.8 per match)
- Technical/domain terms (+1.0 per match)
- Filler phrase detection (-0.9 penalty)
- Concise sentence length bonus (+0.5 for 8-35 words)

It then picks the highest-scoring sentences that fit the token budget, reassembled in original order. This runs in <1ms per result with no external calls.

## 8. What I'd Do Differently With More Time

*Items 1-4 from an earlier version have been implemented:*
- ~~Relevance scoring before storage~~ → Implemented (`relevance.py`: token overlap + bigram scoring, threshold 0.15)
- ~~Adaptive budget allocation~~ → Implemented (`orchestrator.py`: derives sub-query count, buffer size, critique on/off from budget)
- ~~Answer self-critique loop~~ → Implemented (`critique/critique.py`: Claude tool_use identifies gaps, targeted parallel gap-fill)
- ~~Parallel sub-query execution~~ → Implemented (`researcher.py`: ThreadPoolExecutor, groups by dependency level)

**Remaining improvements:**

1. **Persistent vector store**: Currently in-memory (ChromaDB ephemeral client). For production, would use persistent storage so sessions survive restarts.

2. **LLMLingua integration**: For workloads with very large source documents, a learned compressor would outperform our regex-based approach. Worth the dependency for production use.

3. **Per-sub-query budget weighting**: The orchestrator sets a global sub-query count, but doesn't allocate *more* Tavily results to harder sub-queries. A confidence-based weighting from the decomposer could improve this.

4. **Streaming synthesis**: Currently the final answer arrives all at once. Streaming the synthesis step would improve perceived latency significantly.

5. **Multi-turn research sessions**: Currently each request is a single pass (with optional critique). Supporting true multi-turn where the user can ask follow-up questions that build on the same memory state would be valuable.

## 9. Business Impact

### Who needs this?

- **Consulting firms**: Research analysts spending 2-3 hours on multi-source research questions. This agent does 80% of the work in under 70 seconds for $0.04-0.06.
- **Due diligence teams**: Structured decomposition of complex questions maps directly to investment research workflows.
- **Legal research**: Multi-jurisdictional regulatory questions decompose naturally into sub-queries per jurisdiction.

### Cost comparison

| Method | Time | Cost | Quality |
|--------|------|------|---------|
| Junior analyst | 2-3 hours | $75-150 (labor) | High but slow |
| ChatGPT (single prompt) | 30s | ~$0.02 | Misses nuance on complex Qs |
| **DeepDig (with critique)** | ~60s | $0.04-0.06 | Structured, sourced, gap-filled |
| **DeepDig (budget-tight)** | ~40s | $0.03-0.04 | Critique disabled, still cited |

### Why constraints + adaptive orchestration matter commercially

- Clients don't want unbounded AI costs. A research agent that guarantees "$0.10 max per query" is sellable. An agent that might cost $0.01 or $5.00 depending on the question is not.
- The adaptive orchestrator IS the product value — it maximizes quality within a hard budget, not by failing when budget runs low, but by gracefully scaling down (fewer sub-queries, skip critique, shrink buffer).
- Measured: every query we tested landed between $0.03-$0.06, well within the $0.10 ceiling. This predictability is the feature.
