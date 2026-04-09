# DeepDig: Deep Research Agent with Memory Constraints

A research agent that answers complex, multi-part queries by decomposing them into sub-questions, researching each using web search, managing context within strict memory/cost budgets, and synthesizing a final answer — orchestrable through n8n or direct API.

## Constraints & Adaptive Orchestration

### Hard Constraints (True Limits)

| Constraint | Value | Purpose |
|---|---|---|
| Max context per LLM call | 3,000 tokens | Forces memory management — truncates input if over |
| Max cost per session | $0.10 USD | Enforces budget-aware decisions across all phases |

### Orchestration Parameters (Adaptive, Derived from Constraints)

Everything below is **not a constraint** — it's an orchestration decision made at runtime by the adaptive orchestrator based on remaining budget:

| Parameter | Range | Decided By |
|---|---|---|
| Sub-query count | 1–8 | Budget remaining after reserving for compression + synthesis |
| Episodic buffer size | 5–10 | Budget available for compression |
| Critique loop | on/off | Whether budget can afford critique + gap-fill + re-synthesis |
| Gap-fill queries | 0–4 | Remaining budget after critique |
| Tavily results/query | 3–5 | Budget tier |

## Architecture

```
                          ┌─────────────────────┐
                          │   n8n ORCHESTRATION  │
                          │  (optional routing)  │
                          └──────────┬──────────┘
                                     │ POST /research
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Agent                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  ADAPTIVE ORCHESTRATOR (derives all params from budget)  │    │
│  │  → sub-query count, buffer size, critique on/off, etc.   │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             ▼                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │Decomposer│▶│Researcher│▶│ Critique │▶│Synthesizer│           │
│  │(Claude)  │ │(Tavily)  │ │(Claude)  │ │(Claude)   │           │
│  └──────────┘ └────┬─────┘ └──────────┘ └─────▲─────┘           │
│                    │                           │                  │
│              ┌─────▼───────────────────────────┘─────┐           │
│              │        MEMORY MANAGER                 │           │
│              │  ┌───────────┐  ┌──────────┐          │           │
│              │  │ Episodic  │  │  Vector  │          │           │
│              │  │Buffer(5-10)│  │  Store   │          │           │
│              │  └─────┬─────┘  └──────▲───┘          │           │
│              │        │evict          │store          │           │
│              │  ┌─────▼──────────────┐│              │           │
│              │  │ Batch Summarizer   ├┘              │           │
│              │  │ (1 LLM call)      │               │           │
│              │  └────────────────────┘               │           │
│              │  ┌────────────────────┐               │           │
│              │  │ Cost Tracker       │               │           │
│              │  │ (gates every call) │               │           │
│              │  └────────────────────┘               │           │
│              └───────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

### Components

| Component | What It Does | Tech |
|---|---|---|
| **Adaptive Orchestrator** | Analyzes budget, derives all pipeline parameters at runtime | Python |
| **Query Decomposer** | Breaks complex question into ranked, dependency-ordered sub-queries | Claude tool_use |
| **Research Executor** | Runs parallel web searches, deduplicates, filters by relevance | Tavily API + async |
| **Extractive Compressor** | Scores sentences by information density, drops filler before storage | Regex-based scorer |
| **Relevance Scorer** | Filters low-quality findings before storage (no LLM calls) | Token overlap + bigrams |
| **Memory Manager** | Routes between episodic buffer, vector store, and summarizer | Python |
| **Episodic Buffer** | Holds last 5–10 findings in full detail for recency (adaptive size) | In-memory deque |
| **Vector Store** | Semantic search over compressed archived findings | ChromaDB |
| **Batch Summarizer** | Compresses ALL evicted entries in 1 LLM call (not N) | Claude API |
| **Cost Tracker** | Per-call token counting, budget gating, call-by-call audit log | tiktoken |
| **Self-Critique** | Identifies gaps in draft answer, generates targeted follow-up queries | Claude tool_use |
| **Synthesizer** | Combines episodic + vector context into cited final answer | Claude API |

### Pipeline Flow (8 Steps)

```
Step 1: Session Setup      → New or resumed session with budget state
Step 2: Orchestration Plan → Analyze budget → decide sub-query count, critique on/off, buffer size
Step 3: Decomposition      → Complex question → 1-8 sub-queries (1 LLM call, Claude tool_use)
Step 4: Primary Research   → Parallel Tavily search → dedup → relevance filter → episodic buffer
Step 5: Memory Compress    → Batch-compress evicted entries → vector store (0-1 LLM calls)
Step 6: Draft Synthesis    → Episodic + vector context → draft answer (1 LLM call)
Step 7: Critique + Gap-Fill→ Identify gaps → targeted parallel research → compress (if budget allows)
Step 8: Final Synthesis    → Re-synthesize with gap-fill context → final cited answer (1 LLM call)

Total: 3-6 LLM calls per query (~$0.04-0.06, 40-60% of budget)
```

## Quick Start

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Tavily API key](https://tavily.com/) (for web search)

### Local Development

```bash
cd deepdig
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys

# Start the agent
uvicorn agent.main:app --host 0.0.0.0 --port 8000
```

### Run Demo

```bash
# Direct API mode (2 iterations: research + gap-filling)
python -m demo.run_demo

# Through n8n orchestration (n8n handles quality gate + follow-up)
python -m demo.run_demo --n8n
```

### n8n Setup (Optional)

```bash
# Start n8n
npx n8n

# Open http://localhost:5678
# Import n8n/workflows/deepdig_main.json
# Activate the workflow (toggle in top-right)
# Production webhook: POST http://localhost:5678/webhook/research
```

### Web UI

Open http://localhost:8000 for the interactive research dashboard with 8 demo categories, pipeline visualization, and source references.

### Run Tests

```bash
pytest tests/ -v    # 38 tests
```

## API Endpoints

### `POST /research`

Run a full research pipeline.

```json
{
  "question": "How has the EU AI Act affected AI startups?",
  "session_id": null
}
```

Response includes: `answer`, `sub_queries`, `session_id`, `session_report`, `steps` (8-step pipeline log).

### `GET /session/{session_id}/report`

Get cost and memory report for a session.

### `GET /health`

Health check.

## How the Memory System Works

1. **New finding** arrives → relevance filter → extractive compressor strips filler → stored in **Episodic Buffer** (adaptive size 5–10)
2. **Buffer full** → evicted entries batched → **Batch Summarizer** compresses ALL evictions in 1 LLM call → stored in **Vector Store**
3. **Context retrieval** → recent entries from episodic buffer first (high priority), then semantic search from vector store (fill remaining token budget)
4. **Budget check** before every LLM call → orchestrator adapts pipeline (fewer sub-queries, disable critique, shrink buffer) rather than failing

### Key Optimization: Batch Compression

Early versions compressed each evicted entry individually (~14 LLM calls per iteration, consuming 70% of budget). The batch approach compresses all evictions in a single call, reducing LLM calls from ~16 to ~3 per iteration and cost from ~$0.07 to ~$0.04.

## Measured Performance

| Metric | Constrained (with critique) | Unconstrained | Notes |
|---|---|---|---|
| LLM calls/query | 4–6 | 4 (2 iters) | Critique adds 1-2 calls |
| Cost/query | $0.04–0.06 | $0.045 | Within $0.10 budget |
| Time/query | ~50–70s | ~71s | Parallel research saves time |
| Answer quality | Structured, cited, gap-filled | +20% more text | Self-critique closes gap |
| Budget used | 40–60% | 0.9% (of $5) | Orchestrator adapts to spend wisely |

## Project Structure

```
deepdig/
├── agent/
│   ├── main.py                # FastAPI app (8-step adaptive pipeline)
│   ├── config.py              # True constraints + orchestration ceilings
│   ├── orchestrator.py        # Adaptive orchestrator (derives params from budget)
│   ├── models.py              # Pydantic schemas (StepLog, etc.)
│   ├── decomposer/            # Query → sub-queries (Claude tool_use)
│   ├── researcher/            # Parallel async research (Tavily + dedup)
│   ├── critique/              # Self-critique + gap identification (Claude tool_use)
│   ├── memory/                # 3-tier memory system
│   │   ├── manager.py         # Unified interface + batch eviction + dynamic resize
│   │   ├── episodic_buffer.py # Recent findings (deque, adaptive size 5-10)
│   │   ├── vector_store.py    # ChromaDB semantic search
│   │   ├── summarizer.py      # Compression cascade (L1 + L2)
│   │   └── cost_tracker.py    # Token budget enforcement
│   ├── synthesizer/           # Context → final answer (Claude)
│   └── utils/
│       ├── llm_client.py      # Central LLM gateway (all calls here)
│       ├── token_counter.py   # tiktoken-based counting
│       ├── compressor.py      # Extractive sentence compressor
│       └── relevance.py       # Token overlap + bigram relevance scorer
├── n8n/workflows/             # Importable n8n workflow
├── ui/index.html              # Interactive research dashboard (8 demo categories)
├── tests/                     # 38 tests (unit + integration + e2e)
├── demo/                      # Demo queries + run script
├── evaluation.md              # Architecture trade-off analysis
└── requirements.txt
```

## Evaluation

See [evaluation.md](evaluation.md) for detailed trade-off analysis covering:
- Memory architecture alternatives and why 3-tier was chosen
- Constraint design rationale and why each limit exists
- Compression vs. information loss trade-offs (measured)
- Optimization journey: from 16 LLM calls to 3
- Constrained vs. unconstrained quality comparison
- Business impact analysis
