"""
DeepDig Demo — 2 iteration cycles with step-by-step output.

Supports two modes:
  --n8n     Route through n8n orchestration (n8n handles quality gate + iteration 2)
  --direct  Call FastAPI agent directly (demo script handles iteration 2)

Default: --direct

Usage:
    python -m demo.run_demo              # direct API
    python -m demo.run_demo --n8n        # through n8n
"""

import sys
import time

import httpx

N8N_URL = "http://localhost:5678/webhook/research"
API_URL = "http://127.0.0.1:8000/research"


DEMO_QUERIES = [
    {
        "name": "Multi-part Geopolitical Question",
        "question": (
            "How has the EU AI Act affected the competitive landscape between "
            "European and American AI startups in 2025-2026? Compare regulatory "
            "compliance costs, market access changes, and venture capital flows "
            "in both regions."
        ),
    },
    {
        "name": "Technical Architecture Deep Dive",
        "question": (
            "Compare the memory architectures used in GPT-4, Claude, and Gemini "
            "for handling long-context tasks. What are the trade-offs between "
            "sparse attention, sliding window, and retrieval-augmented approaches "
            "in terms of cost, latency, and accuracy?"
        ),
    },
]


def get_mode() -> str:
    if "--n8n" in sys.argv:
        return "n8n"
    return "direct"


def call_api(question: str, session_id: str | None = None) -> dict:
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    return httpx.post(API_URL, json=payload, timeout=300).json()


def call_n8n(question: str) -> dict:
    return httpx.post(N8N_URL, json={"question": question}, timeout=300).json()


def print_step(step: dict):
    icons = {1: "🔗", 2: "⚙️", 3: "🔍", 4: "🌐", 5: "📦", 6: "✨", 7: "🔬", 8: "✅"}
    icon = icons.get(step["step"], "•")
    print(f"\n  {icon}  Step {step['step']}: {step['name']}")
    print(f"     {step['description']}")

    data = step.get("data", {})

    if "sub_queries" in data:
        for sq in data["sub_queries"]:
            print(f"       [{sq['id']}] {sq['query']}")
            print(f"           Purpose: {sq['purpose']}")

    if "researched" in data:
        for r in data["researched"]:
            print(f"       📄 {r['sub_query']} → {r['findings_count']} results")
            for f in r.get("findings", []):
                src = f["source"][:60] + "..." if len(f["source"]) > 60 else f["source"]
                print(f"          └─ {src}")

    if "episodic_count" in data:
        print(f"       Episodic: {data['episodic_count']} | "
              f"Archived: {data['vector_count']} | "
              f"Evictions compressed: {data['evictions_compressed']}")

    if "cost_report" in data:
        c = data["cost_report"]
        print(f"       💰 Cost: ${c['total_cost']:.4f} ({c['budget_used_pct']:.1f}% of budget)")


def print_answer(answer: str, label: str):
    print(f"\n  {'━' * 64}")
    print(f"  {label}:")
    print(f"  {'━' * 64}")
    for line in answer.split("\n"):
        print(f"  {line}")
    print(f"  {'━' * 64}")


def print_cost_report(data: dict):
    # Handle both direct API (session_report.cost_report) and n8n (cost_report at top level)
    if "session_report" in data and isinstance(data["session_report"], dict):
        report = data["session_report"]
        cost = report.get("cost_report", {})
    elif "cost_report" in data and isinstance(data["cost_report"], dict):
        report = data
        cost = data["cost_report"]
    else:
        cost = data
        report = data

    # Parse session_report if it's a string (from n8n Set nodes)
    if isinstance(report, str):
        try:
            import json
            report = json.loads(report)
            cost = report.get("cost_report", cost)
        except Exception:
            pass

    if isinstance(cost, str):
        try:
            import json
            cost = json.loads(cost)
        except Exception:
            pass

    print(f"\n  📊 SESSION SUMMARY")
    print(f"     Episodic buffer: {report.get('episodic_entries', '?')}")
    print(f"     Vector archive:  {report.get('vector_entries', '?')} entries")
    print(f"     LLM calls:       {len(cost.get('call_log', []))}")
    print(f"     Input tokens:    {cost.get('total_input_tokens', 0):,}")
    print(f"     Output tokens:   {cost.get('total_output_tokens', 0):,}")
    print(f"     Total cost:      ${cost.get('total_cost', 0):.4f} / $0.10")
    print(f"     Budget used:     {cost.get('budget_used_pct', 0):.1f}%")
    print(f"     Budget left:     ${cost.get('budget_remaining', 0):.4f}")

    if cost.get("call_log"):
        print(f"\n     Call-by-call breakdown:")
        for i, c in enumerate(cost["call_log"], 1):
            print(f"       {i}. {c['purpose']:25s} "
                  f"in={c['input_tokens']:>5d}  out={c['output_tokens']:>5d}  "
                  f"${c['cost']:.4f}  (cum: ${c['cumulative_cost']:.4f})")


def run_direct(demo: dict):
    """Direct API mode: demo script controls both iterations."""
    # ── ITERATION 1 ──
    print(">>> ITERATION 1: Initial Research (direct API)")
    t0 = time.time()
    r1 = call_api(demo["question"])
    t1 = time.time()
    print(f"  ⏱  Time: {t1 - t0:.1f}s")

    for step in r1.get("steps", []):
        print_step(step)

    print_answer(r1.get("answer", "[No answer]"), "ANSWER (Iteration 1)")

    # ── ITERATION 2 ──
    follow_up = (
        f"What are the most important gaps or weakly supported claims "
        f"in the previous answer about: {demo['question']}"
    )
    print("\n>>> ITERATION 2: Gap-Filling Research (direct API)")
    t2 = time.time()
    r2 = call_api(follow_up, r1.get("session_id"))
    t3 = time.time()
    print(f"  ⏱  Time: {t3 - t2:.1f}s")

    for step in r2.get("steps", []):
        print_step(step)

    print_answer(r2.get("answer", "[No answer]"), "ANSWER (Iteration 2)")

    print_cost_report(r2)


def run_n8n(demo: dict):
    """n8n mode: n8n orchestrates quality gate + optional iteration 2."""
    print(">>> Research via n8n orchestration")
    print("    n8n flow: Webhook → Research → Quality Gate → (Follow-up?) → Return")
    t0 = time.time()
    result = call_n8n(demo["question"])
    t1 = time.time()
    print(f"  ⏱  Time: {t1 - t0:.1f}s")

    # Show steps if the agent included them
    for step in result.get("steps", []):
        print_step(step)

    # Show answer
    answer = result.get("answer", "[No answer]")
    print_answer(answer, "ANSWER (via n8n)")

    # Show cost
    print_cost_report(result)

    # Show n8n orchestration info
    if result.get("orchestrated_by") == "n8n":
        print(f"\n  🔄 Orchestrated by n8n")


def run_demo():
    mode = get_mode()

    print("=" * 70)
    print("DeepDig Research Agent — Demo")
    print(f"Mode: {'n8n orchestration' if mode == 'n8n' else 'direct API'}")
    print("=" * 70)

    for demo in DEMO_QUERIES:
        print(f"\n{'─' * 70}")
        print(f"QUERY: {demo['name']}")
        print(f"{'─' * 70}")
        print(f"Q: {demo['question']}\n")

        if mode == "n8n":
            run_n8n(demo)
        else:
            run_direct(demo)

    print(f"\n{'=' * 70}")
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
