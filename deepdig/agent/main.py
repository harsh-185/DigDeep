from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import asyncio
import json
import os

from agent.config import settings
from agent.critique.critique import critique_answer
from agent.decomposer.decomposer import decompose_query
from agent.memory.manager import MemoryManager
from agent.models import ResearchRequest, ResearchResponse, StepLog
from agent.orchestrator import plan_research
from agent.researcher.researcher import research_sub_queries_parallel
from agent.synthesizer.synthesizer import synthesize_answer
from agent.utils.relevance import relevance_score

app = FastAPI(title="DeepDig Research Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, MemoryManager] = {}

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
UI_DIR = os.path.join(ROOT_DIR, "ui")
DEMO_DIR = os.path.join(ROOT_DIR, "demo")

RELEVANCE_THRESHOLD = 0.15


@app.get("/")
async def serve_ui():
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "UI not found. Access the API at /docs"}


@app.get("/demo/demo_queries.json")
async def serve_demo_queries():
    path = os.path.join(DEMO_DIR, "demo_queries.json")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json")
    return []


async def _do_research(
    sub_queries: list[dict],
    question: str,
    memory: MemoryManager,
    seen_urls: set[str],
) -> tuple[list[dict], int, int, int]:
    """Run parallel research, filter, dedup, store."""
    research_result = await research_sub_queries_parallel(sub_queries, seen_urls)

    all_findings: list[dict] = []
    filtered_count = 0
    stored_count = 0

    for item in research_result["results"]:
        sq = item["sub_query"]
        findings = item["findings"]

        sq_findings = []
        for f in findings:
            score = relevance_score(question, f["content"])
            if score < RELEVANCE_THRESHOLD:
                filtered_count += 1
                continue

            memory.store_finding(
                content=f["content"],
                source=f["source"],
                sub_query=sq["query"],
            )
            stored_count += 1
            sq_findings.append({
                "source": f["source"],
                "content_preview": f["content"][:150] + "..." if len(f["content"]) > 150 else f["content"],
                "tokens": f.get("token_count", 0),
                "relevance": round(score, 2),
            })

        all_findings.append({
            "sub_query": sq["query"],
            "findings_count": len(sq_findings),
            "findings": sq_findings,
        })

    return all_findings, stored_count, filtered_count, research_result["dedup_count"]


@app.post("/research", response_model=ResearchResponse)
async def research(request: Request):
    """
    Adaptive research pipeline — orchestrator decides sub-query count,
    critique enablement, and gap-filling depth based on budget state.

    Hard constraints: $0.10 budget, 3000 tokens/call.
    Everything else is adaptive.
    """
    body = await request.json()
    print(f"[DeepDig] Raw body: {body}")
    req = ResearchRequest(
        question=body.get("question", ""),
        session_id=body.get("session_id"),
    )
    print(f"[DeepDig] Parsed: question='{req.question[:80]}', session_id={req.session_id}")
    steps: list[StepLog] = []

    # ── Step 1: Session Setup ──
    if req.session_id and req.session_id in sessions:
        memory = sessions[req.session_id]
        steps.append(StepLog(
            step=1,
            name="Session Resumed",
            description=f"Reusing session {memory.session_id} with "
                        f"{len(memory.episodic.get_all())} episodic + "
                        f"{memory.vector.count()} archived entries.",
            data={"session_id": memory.session_id, "is_new": False},
        ))
    else:
        memory = MemoryManager()
        sessions[memory.session_id] = memory
        steps.append(StepLog(
            step=1,
            name="Session Created",
            description=f"Session {memory.session_id} — "
                        f"${settings.MAX_COST_PER_SESSION} budget, "
                        f"{settings.MAX_CONTEXT_TOKENS_PER_CALL} tokens/call.",
            data={"session_id": memory.session_id, "is_new": True},
        ))

    # ── Orchestrator: plan the pipeline based on budget ──
    plan = plan_research(memory.cost_tracker)
    print(f"[DeepDig] Orchestrator: {plan['reason']}")

    steps.append(StepLog(
        step=2,
        name="Orchestration Plan",
        description=plan["reason"],
        data={
            "max_sub_queries": plan["max_sub_queries"],
            "enable_critique": plan["enable_critique"],
            "max_gap_queries": plan["max_gap_queries"],
            "buffer_size": plan["buffer_size"],
            "tavily_results": plan["tavily_results"],
        },
    ))

    # Resize buffer if orchestrator says so
    if plan["buffer_size"] != memory.episodic.buffer.maxlen:
        memory.resize_buffer(plan["buffer_size"])

    # ── Step 3: Query Decomposition (capped by orchestrator) ──
    sub_queries = decompose_query(req.question, memory.cost_tracker)
    sub_queries = sub_queries[: plan["max_sub_queries"]]

    steps.append(StepLog(
        step=3,
        name="Query Decomposition",
        description=f"Decomposed into {len(sub_queries)} sub-queries "
                    f"(orchestrator cap: {plan['max_sub_queries']}).",
        data={
            "sub_queries": sub_queries,
            "cost_after": memory.cost_tracker.get_report()["total_cost"],
        },
    ))

    # ── Step 4: Primary Research (parallel + dedup + relevance) ──
    seen_urls: set[str] = set()
    all_findings, stored, filtered, deduped = await _do_research(
        sub_queries, req.question, memory, seen_urls,
    )

    total_raw = stored + filtered + deduped
    steps.append(StepLog(
        step=4,
        name="Primary Research",
        description=f"Searched {len(sub_queries)} sub-queries (parallel). "
                    f"Stored {stored}/{total_raw}."
                    + (f" Deduped {deduped}." if deduped else "")
                    + (f" Filtered {filtered} low-relevance." if filtered else ""),
        data={
            "researched": all_findings,
            "dedup_count": deduped,
            "filtered_count": filtered,
        },
    ))

    # ── Step 5: Memory Compression ──
    eviction_count = len(memory._eviction_batch)
    memory.flush_evictions()
    steps.append(StepLog(
        step=5,
        name="Memory Compression",
        description=f"Buffer: {len(memory.episodic.get_all())}/{plan['buffer_size']}. "
                    + (f"Compressed {eviction_count} evictions → "
                       f"{memory.vector.count()} archived."
                       if eviction_count > 0
                       else f"No evictions. {memory.vector.count()} archived."),
        data={
            "episodic_count": len(memory.episodic.get_all()),
            "vector_count": memory.vector.count(),
            "evictions_compressed": eviction_count,
            "cost_after": memory.cost_tracker.get_report()["total_cost"],
        },
    ))

    # ── Step 6: Draft Synthesis ──
    draft = synthesize_answer(req.question, memory)
    steps.append(StepLog(
        step=6,
        name="Draft Synthesis",
        description=f"Draft: {len(draft['answer'])} chars. "
                    f"Cost: ${memory.cost_tracker.get_report()['total_cost']:.4f}.",
        data={
            "answer_length": len(draft["answer"]),
            "cost_after": memory.cost_tracker.get_report()["total_cost"],
        },
    ))

    # ── Step 7: Self-Critique + Gap-Filling (if orchestrator enabled it) ──
    if plan["enable_critique"] and memory.cost_tracker.can_afford():
        gaps = critique_answer(req.question, draft["answer"], memory.cost_tracker)

        if gaps:
            gap_queries = [
                {"id": i + 100, "query": g["search_query"], "purpose": g["gap"], "depends_on": []}
                for i, g in enumerate(gaps[: plan["max_gap_queries"]])
            ]

            gap_findings, gap_stored, gap_filtered, gap_deduped = await _do_research(
                gap_queries, req.question, memory, seen_urls,
            )

            gap_evictions = len(memory._eviction_batch)
            memory.flush_evictions()

            steps.append(StepLog(
                step=7,
                name="Critique + Gap-Fill",
                description=f"Found {len(gaps)} gaps → researched {len(gap_queries)} "
                            f"(parallel). +{gap_stored} findings."
                            + (f" Deduped {gap_deduped}." if gap_deduped else "")
                            + (f" Compressed {gap_evictions}." if gap_evictions else ""),
                data={
                    "gaps": gaps,
                    "researched": gap_findings,
                    "gap_stored": gap_stored,
                },
            ))
        else:
            steps.append(StepLog(
                step=7,
                name="Critique + Gap-Fill",
                description="Critique found no significant gaps — draft is solid.",
                data={"gaps": []},
            ))
    else:
        reason = "budget insufficient" if not memory.cost_tracker.can_afford() else "orchestrator disabled"
        steps.append(StepLog(
            step=7,
            name="Critique + Gap-Fill",
            description=f"Skipped ({reason}).",
            data={"skipped": True, "reason": reason},
        ))

    # ── Step 8: Final Synthesis ──
    final = synthesize_answer(req.question, memory)
    steps.append(StepLog(
        step=8,
        name="Final Synthesis",
        description=f"Final answer: {len(final['answer'])} chars. "
                    f"Total: ${memory.cost_tracker.get_report()['total_cost']:.4f} "
                    f"({memory.cost_tracker.get_report()['budget_used_pct']:.0f}% of budget).",
        data={
            "answer_length": len(final["answer"]),
            "cost_report": memory.cost_tracker.get_report(),
        },
    ))

    return ResearchResponse(
        answer=final["answer"],
        sub_queries=sub_queries,
        session_id=memory.session_id,
        session_report=final["session_report"],
        steps=steps,
    )


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


@app.post("/research/stream")
async def research_stream(request: Request):
    """
    SSE streaming version — sends each pipeline step as it completes.
    Events: step, answer, done, error
    """
    body = await request.json()
    req = ResearchRequest(
        question=body.get("question", ""),
        session_id=body.get("session_id"),
    )

    async def generate():
        try:
            # ── Step 1: Session Setup ──
            if req.session_id and req.session_id in sessions:
                memory = sessions[req.session_id]
                step1 = StepLog(
                    step=1, name="Session Resumed",
                    description=f"Reusing session {memory.session_id} with "
                                f"{len(memory.episodic.get_all())} episodic + "
                                f"{memory.vector.count()} archived entries.",
                    data={"session_id": memory.session_id, "is_new": False},
                )
            else:
                memory = MemoryManager()
                sessions[memory.session_id] = memory
                step1 = StepLog(
                    step=1, name="Session Created",
                    description=f"Session {memory.session_id} — "
                                f"${settings.MAX_COST_PER_SESSION} budget, "
                                f"{settings.MAX_CONTEXT_TOKENS_PER_CALL} tokens/call.",
                    data={"session_id": memory.session_id, "is_new": True},
                )
            yield _sse_event("step", step1.model_dump())
            await asyncio.sleep(0)

            # ── Step 2: Orchestration Plan ──
            plan = plan_research(memory.cost_tracker)
            if plan["buffer_size"] != memory.episodic.buffer.maxlen:
                memory.resize_buffer(plan["buffer_size"])

            step2 = StepLog(
                step=2, name="Orchestration Plan",
                description=plan["reason"],
                data={
                    "max_sub_queries": plan["max_sub_queries"],
                    "enable_critique": plan["enable_critique"],
                    "max_gap_queries": plan["max_gap_queries"],
                    "buffer_size": plan["buffer_size"],
                    "tavily_results": plan["tavily_results"],
                },
            )
            yield _sse_event("step", step2.model_dump())
            await asyncio.sleep(0)

            # ── Step 3: Query Decomposition ──
            sub_queries = decompose_query(req.question, memory.cost_tracker)
            sub_queries = sub_queries[: plan["max_sub_queries"]]

            step3 = StepLog(
                step=3, name="Query Decomposition",
                description=f"Decomposed into {len(sub_queries)} sub-queries "
                            f"(orchestrator cap: {plan['max_sub_queries']}).",
                data={
                    "sub_queries": sub_queries,
                    "cost_after": memory.cost_tracker.get_report()["total_cost"],
                },
            )
            yield _sse_event("step", step3.model_dump())
            await asyncio.sleep(0)

            # ── Step 4: Primary Research ──
            seen_urls: set[str] = set()
            all_findings, stored, filtered, deduped = await _do_research(
                sub_queries, req.question, memory, seen_urls,
            )
            total_raw = stored + filtered + deduped

            step4 = StepLog(
                step=4, name="Primary Research",
                description=f"Searched {len(sub_queries)} sub-queries (parallel). "
                            f"Stored {stored}/{total_raw}."
                            + (f" Deduped {deduped}." if deduped else "")
                            + (f" Filtered {filtered} low-relevance." if filtered else ""),
                data={
                    "researched": all_findings,
                    "dedup_count": deduped,
                    "filtered_count": filtered,
                },
            )
            yield _sse_event("step", step4.model_dump())
            await asyncio.sleep(0)

            # ── Step 5: Memory Compression ──
            eviction_count = len(memory._eviction_batch)
            memory.flush_evictions()

            step5 = StepLog(
                step=5, name="Memory Compression",
                description=f"Buffer: {len(memory.episodic.get_all())}/{plan['buffer_size']}. "
                            + (f"Compressed {eviction_count} evictions → "
                               f"{memory.vector.count()} archived."
                               if eviction_count > 0
                               else f"No evictions. {memory.vector.count()} archived."),
                data={
                    "episodic_count": len(memory.episodic.get_all()),
                    "vector_count": memory.vector.count(),
                    "evictions_compressed": eviction_count,
                    "cost_after": memory.cost_tracker.get_report()["total_cost"],
                },
            )
            yield _sse_event("step", step5.model_dump())
            await asyncio.sleep(0)

            # ── Step 6: Draft Synthesis ──
            draft = synthesize_answer(req.question, memory)

            step6 = StepLog(
                step=6, name="Draft Synthesis",
                description=f"Draft: {len(draft['answer'])} chars. "
                            f"Cost: ${memory.cost_tracker.get_report()['total_cost']:.4f}.",
                data={
                    "answer_length": len(draft["answer"]),
                    "cost_after": memory.cost_tracker.get_report()["total_cost"],
                },
            )
            yield _sse_event("step", step6.model_dump())
            await asyncio.sleep(0)

            # ── Step 7: Critique + Gap-Fill ──
            if plan["enable_critique"] and memory.cost_tracker.can_afford():
                gaps = critique_answer(req.question, draft["answer"], memory.cost_tracker)

                if gaps:
                    gap_queries = [
                        {"id": i + 100, "query": g["search_query"], "purpose": g["gap"], "depends_on": []}
                        for i, g in enumerate(gaps[: plan["max_gap_queries"]])
                    ]
                    gap_findings, gap_stored, gap_filtered, gap_deduped = await _do_research(
                        gap_queries, req.question, memory, seen_urls,
                    )
                    gap_evictions = len(memory._eviction_batch)
                    memory.flush_evictions()

                    step7 = StepLog(
                        step=7, name="Critique + Gap-Fill",
                        description=f"Found {len(gaps)} gaps → researched {len(gap_queries)} "
                                    f"(parallel). +{gap_stored} findings."
                                    + (f" Deduped {gap_deduped}." if gap_deduped else "")
                                    + (f" Compressed {gap_evictions}." if gap_evictions else ""),
                        data={"gaps": gaps, "researched": gap_findings, "gap_stored": gap_stored},
                    )
                else:
                    step7 = StepLog(
                        step=7, name="Critique + Gap-Fill",
                        description="Critique found no significant gaps — draft is solid.",
                        data={"gaps": []},
                    )
            else:
                reason = "budget insufficient" if not memory.cost_tracker.can_afford() else "orchestrator disabled"
                step7 = StepLog(
                    step=7, name="Critique + Gap-Fill",
                    description=f"Skipped ({reason}).",
                    data={"skipped": True, "reason": reason},
                )
            yield _sse_event("step", step7.model_dump())
            await asyncio.sleep(0)

            # ── Step 8: Final Synthesis ──
            final = synthesize_answer(req.question, memory)

            step8 = StepLog(
                step=8, name="Final Synthesis",
                description=f"Final answer: {len(final['answer'])} chars. "
                            f"Total: ${memory.cost_tracker.get_report()['total_cost']:.4f} "
                            f"({memory.cost_tracker.get_report()['budget_used_pct']:.0f}% of budget).",
                data={
                    "answer_length": len(final["answer"]),
                    "cost_report": memory.cost_tracker.get_report(),
                },
            )
            yield _sse_event("step", step8.model_dump())
            await asyncio.sleep(0)

            # ── Final: send complete response ──
            all_steps = [step1, step2, step3, step4, step5, step6, step7, step8]
            yield _sse_event("done", {
                "answer": final["answer"],
                "sub_queries": sub_queries,
                "session_id": memory.session_id,
                "session_report": final["session_report"],
                "steps": [s.model_dump() for s in all_steps],
            })

        except Exception as e:
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/session/{session_id}/report")
async def get_session_report(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id].get_session_report()


@app.post("/debug")
async def debug(request: dict):
    print(f"[DEBUG] Received: {request}")
    return {"received": request}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
