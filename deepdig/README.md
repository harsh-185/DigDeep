# DeepDig

A deep research agent that answers complex, multi-part questions under strict cost and memory constraints. It decomposes queries, searches the web in parallel, manages a 3-tier memory system, self-critiques its draft, fills gaps, and synthesizes a cited answer — all within a $0.10 budget and 3,000 tokens per LLM call.

**Live demo**: [Railway deployment](https://digdeep-production.up.railway.app) | **Repo**: [github.com/harsh-185/DigDeep](https://github.com/harsh-185/DigDeep)

---

## Why This Design?

The assignment asks for a research agent with memory constraints. The naive approach — stuff everything into one LLM call — doesn't work under real constraints. The interesting problem is: **how do you maximize research quality when every LLM call costs money and every token counts?**

My answer: an adaptive orchestrator that treats budget as the single source of truth, deriving every runtime decision from how much budget remains. This isn't a pipeline with fixed parameters — it's a system that gracefully degrades when budget is tight and takes advantage when budget is plentiful.

---

## Architecture Overview

```
User / UI (SSE streaming)
    |
    | POST /research/stream
    v
+------------------------------------------------------------------+
|                        FastAPI Backend                             |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |  ADAPTIVE ORCHESTRATOR                                      |  |
|  |  Reads budget state -> derives ALL runtime parameters       |  |
|  |  (sub-query count, buffer size, critique on/off, etc.)      |  |
|  +-----------------------------+------------------------------+  |
|                                |                                  |
|  Step 1: Session        Step 2: Plan         Step 3: Decompose   |
|  (new/resume)           (budget -> params)   (question -> subs)  |
|                                                                   |
|  Step 4: Research       Step 5: Compress     Step 6: Draft       |
|  (parallel Tavily)      (evictions -> store) (episodic + vector)  |
|                                                                   |
|  Step 7: Critique       Step 8: Synthesize                       |
|  (gaps -> fill)         (final cited answer)                      |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |  MEMORY MANAGER                                             |  |
|  |  Episodic Buffer (10) --> Summarizer --> Vector Store        |  |
|  |        recent, full         1 LLM call     semantic search  |  |
|  |  Cost Tracker gates every LLM call                          |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

---

## The Two Hard Constraints

Everything in this system flows from two immovable limits:

| Constraint | Value | Why It Exists |
|---|---|---|
| **Max cost per session** | $0.10 | Forces budget-aware decisions. A research agent that guarantees "$0.10 max" is predictable and sellable. One that costs $0.01 or $5.00 depending on the question is not. |
| **Max tokens per LLM call** | 3,000 | Forces the memory system to actually work. Without this, you'd stuff everything into one Claude call (200K context) and the project would be trivial. This is the constraint that makes the 3-tier memory architecture necessary. |

**Everything else is adaptive.** Sub-query count, buffer size, critique enablement, gap-fill depth — these are orchestration decisions made at runtime based on remaining budget, not static configuration.

| Parameter | Range | Decision Logic |
|---|---|---|
| Sub-queries | 1-8 | >$0.04 remaining: 8. >$0.02: 6. >$0.01: 4. Exhausted: 2 |
| Episodic buffer | 5-10 | Shrinks when budget can't afford compression LLM calls |
| Critique loop | on/off | Enabled only if budget covers critique + gap-search + re-synthesis |
| Gap-fill queries | 0-4 | Allocated from remaining budget after critique |

---

## The 8-Step Pipeline

Each step streams to the UI in real-time via Server-Sent Events (SSE).

| Step | What Happens | LLM? | Cost |
|---|---|---|---|
| 1. **Session** | Create or resume (memory persists across requests) | No | $0 |
| 2. **Orchestrate** | Read budget -> decide sub-query count, critique on/off, buffer size | No | $0 |
| 3. **Decompose** | Break question into 1-8 dependency-ordered sub-queries via Claude tool_use | Yes (Sonnet) | ~$0.003 |
| 4. **Research** | Parallel Tavily searches -> dedup URLs -> relevance filter -> episodic buffer | No | $0 (Tavily free tier) |
| 5. **Compress** | Batch-compress ALL evicted buffer entries in 1 LLM call -> vector store | Yes (Sonnet) | ~$0.008 |
| 6. **Draft** | Synthesize from episodic (recent) + vector (archived) context | Yes (Sonnet) | ~$0.015 |
| 7. **Critique** | Identify gaps in draft -> targeted parallel searches -> store findings | Yes (Sonnet) | ~$0.012 |
| 8. **Synthesize** | Final answer with all context (original + gap-fill) | Yes (Sonnet) | ~$0.015 |

**Total: 4-6 LLM calls per query, $0.04-0.08 (40-80% of budget)**

---

## Memory System: The Core Design Decision

### The Problem

With a 3,000 token context window, you can't fit all research findings into one LLM call. A single Tavily search returns ~800 tokens. With 8 sub-queries, that's ~6,400 tokens of raw findings — more than double our limit.

### The Solution: 3-Tier Memory

```
New finding -> Relevance filter -> Extractive compressor -> Episodic Buffer (10 slots)
                                                                    |
                                                              buffer full?
                                                                    |
                                                              evict oldest
                                                                    v
                                                         Batch Summarizer
                                                         (1 LLM call for ALL evictions)
                                                                    |
                                                                    v
                                                            Vector Store (ChromaDB)
                                                            (semantic search)
```

**Episodic Buffer** (tier 1): Last 10 findings in full detail. Prioritized during synthesis because recent findings are most relevant in iterative research.

**Vector Store** (tier 2): Compressed, semantically indexed archive. Fills remaining context budget after episodic entries. Gives the synthesizer access to older research that's still relevant.

**Summarization Cascade** (tier 3): The bridge between tiers. When the buffer evicts entries, they're batch-compressed (1 LLM call for all evictions, not 1 per eviction) and stored in the vector store.

### Why Not Just a Vector Store?

A pure vector store loses the recency signal. In iterative research, the findings you just retrieved are usually the most important. The episodic buffer guarantees they're always available in full fidelity.

### Why Not a Sliding Window?

A sliding window drops old context entirely. With complex multi-part questions, you need findings from early sub-queries when synthesizing the final answer. The vector store preserves them (compressed but searchable).

### The Batch Compression Optimization

This was the single biggest optimization. Early versions compressed each evicted entry individually:

```
Before: 14 evictions -> 14 LLM calls -> $0.07 (70% of budget blown on compression alone)
After:  14 evictions -> 1 LLM call   -> $0.008 (8% of budget)
```

This freed budget for the self-critique loop, which meaningfully improves answer quality.

---

## Key Trade-offs

### 1. Compression vs. Information Loss

We compress to ~30% of original text. This loses ~70% of details. But:
- The summarization prompt explicitly preserves numbers, names, and conclusions
- Measured: compressed findings retained >85% of factual claims
- The alternative (dropping old context entirely) loses 100%
- A 30% summary is strictly better than nothing

### 2. Self-Critique Loop vs. Budget

The critique step costs ~$0.012 (critique LLM call + gap-fill searches). On a $0.10 budget, that's 12%. The trade-off:
- **With critique**: Identifies 2-4 gaps, fills them with targeted searches, answer is more complete
- **Without critique**: Saves budget, but misses blind spots the decomposer didn't anticipate
- **Our approach**: The orchestrator decides at runtime. If budget is tight, critique is skipped — no crash, just graceful degradation

### 3. Extractive Compression vs. LLMLingua

We use a regex-based sentence scorer instead of a learned compressor:

| Approach | Compression | Latency | Dependencies |
|---|---|---|---|
| LLMLingua | ~50% | 200-500ms (needs GPU) | Separate model |
| **Ours (extractive)** | ~20-40% | <1ms | None |

For this project's constraints (fast, cheap, no extra infra), extractive wins. In production with larger documents, LLMLingua would be worth the dependency.

### 4. Sonnet for Everything vs. Model Routing

We considered using cheaper Haiku for routine tasks (decomposition, compression, critique) and Sonnet only for synthesis. The math showed Opus is too expensive ($0.15 for one synthesis call > entire budget), and Haiku's quality gap is noticeable on decomposition. Sonnet everywhere at ~$0.04-0.08 per query fits the budget with consistent quality.

### 5. Parallel Research vs. Sequential

Sub-queries run in parallel (ThreadPoolExecutor) when they have no dependencies. This cuts wall-clock time by ~27% (50s vs 70s). The trade-off: parallel queries can't build on each other's findings. We handle this by grouping sub-queries by dependency level — independent ones run in parallel, dependent ones wait.

---

## UI: Real-Time Streaming

The web UI connects via SSE to show each pipeline step as it completes:

- **Pipeline pills** light up as steps complete (blue = active, green = done)
- **Thought process timeline** shows step-by-step details: orchestration decisions, sub-queries with purposes, sources with relevance scores, gaps identified, cost breakdown
- **Loading indicators** with contextual labels ("Searching the web...", "Critiquing & filling gaps...")
- **Answer panel** with inline source citations that highlight on hover

No framework — single HTML file with vanilla JS. Keeps deployment simple and the bundle at zero.

---

## Measured Performance

| Metric | With Critique | Without Critique | Notes |
|---|---|---|---|
| LLM calls | 5-6 | 3-4 | Critique adds 1-2 calls |
| Cost/query | $0.04-0.08 | $0.03-0.04 | Well within $0.10 ceiling |
| Time/query | 50-70s | 35-50s | Parallel research saves ~27% |
| Budget used | 40-80% | 30-40% | Orchestrator adapts to spend wisely |
| Answer quality | Structured, cited, gap-filled | Good but may miss blind spots | Self-critique closes quality gap |

### Constrained vs. Unconstrained Comparison

We tested with constraints removed ($5 budget, 50 buffer, 10K synthesis context):

- **Answer quality is ~90% comparable** — both cite numbers, use structure, identify gaps
- **Constrained is 34% faster** (1 pass vs 2)
- **The memory system isn't just a budget hack** — compressed vector store means synthesis draws from both recent and archived context, while unconstrained just has a flat list

---

## Project Structure

```
deepdig/
  agent/
    main.py              # FastAPI app, SSE streaming endpoint, 8-step pipeline
    config.py            # Hard constraints + orchestration ceilings
    orchestrator.py      # Adaptive orchestrator (derives params from budget)
    models.py            # Pydantic schemas
    decomposer/          # Question -> sub-queries (Claude tool_use)
    researcher/          # Parallel Tavily search + dedup + relevance filter
    critique/            # Self-critique + gap identification (Claude tool_use)
    synthesizer/         # Context -> cited answer
    memory/
      manager.py         # Unified interface, batch eviction, dynamic resize
      episodic_buffer.py # Recent findings (deque, adaptive size)
      vector_store.py    # ChromaDB semantic search
      summarizer.py      # Compression cascade (L1 individual, L2 mega)
      cost_tracker.py    # Per-call token counting, budget gating, audit log
    utils/
      llm_client.py      # Central LLM gateway (all calls route through here)
      token_counter.py   # tiktoken-based counting + truncation
      compressor.py      # Extractive sentence compressor (regex, no LLM)
      relevance.py       # Token overlap + bigram relevance scorer
  ui/index.html          # SSE streaming dashboard (vanilla JS)
  tests/                 # 33 unit tests + 3 e2e tests
  demo/                  # Demo queries + runner script
  evaluation.md          # Detailed trade-off analysis
  n8n/workflows/         # n8n workflow (optional orchestration layer)
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)
- [Tavily API key](https://tavily.com/)

### Run Locally

```bash
cd deepdig
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env

uvicorn agent.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Run Tests

```bash
pytest tests/ -v
```

### Deploy (Railway)

The repo includes a root `Dockerfile`. Connect to Railway, set `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` as environment variables, and deploy. The UI is served at `/`.

---

## What I'd Improve With More Time

1. **Persistent vector store** — Currently in-memory (ChromaDB ephemeral). Production would use persistent storage so sessions survive restarts.
2. **Per-sub-query budget weighting** — The orchestrator sets a global count, but doesn't allocate more results to harder sub-queries. Confidence-based weighting from the decomposer would help.
3. **Streaming synthesis** — The final answer arrives all at once. Streaming the synthesis step would improve perceived latency.
4. **Multi-turn sessions** — Currently single-pass (with critique). True multi-turn where follow-up questions build on the same memory state would be valuable.
5. **LLMLingua for large documents** — The extractive compressor works well for web snippets, but a learned compressor would handle longer source documents better.
