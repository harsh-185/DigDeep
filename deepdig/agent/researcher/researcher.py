import asyncio
from concurrent.futures import ThreadPoolExecutor

from tavily import TavilyClient

from agent.config import settings
from agent.utils.compressor import compress_text
from agent.utils.token_counter import count_tokens, truncate_to_tokens

# Thread pool for parallel Tavily calls (Tavily client is sync)
_executor = ThreadPoolExecutor(max_workers=8)


def _get_tavily_client() -> TavilyClient:
    return TavilyClient(api_key=settings.TAVILY_API_KEY)


def research_sub_query(query: str, max_result_tokens: int = 800) -> list[dict]:
    """
    Execute a web search and return cleaned, token-budgeted results.
    """
    try:
        tavily = _get_tavily_client()
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=False,
        )
    except Exception as e:
        return [{"content": f"Search failed: {str(e)}", "source": "error"}]

    findings = []
    tokens_remaining = max_result_tokens

    for result in response.get("results", []):
        content = result.get("content", "")
        source = result.get("url", "unknown")

        # Compress before budgeting — keep facts, drop filler
        content = compress_text(content, max(tokens_remaining, 150))

        content_tokens = count_tokens(content)
        if content_tokens > tokens_remaining:
            content = truncate_to_tokens(content, tokens_remaining)
            tokens_remaining = 0
        else:
            tokens_remaining -= content_tokens

        findings.append(
            {
                "content": content,
                "source": source,
                "token_count": count_tokens(content),
            }
        )

        if tokens_remaining <= 0:
            break

    return findings


async def research_sub_queries_parallel(
    sub_queries: list[dict],
    seen_urls: set[str] | None = None,
) -> dict:
    """
    Run independent sub-queries in parallel, sequential ones in order.
    Returns {results: [...], dedup_count: int} where each result has
    sub_query info + findings with duplicates removed.
    """
    if seen_urls is None:
        seen_urls = set()

    # Group sub-queries by dependency level for parallel execution
    # Level 0: no dependencies -> run in parallel
    # Level 1: depends on level 0 -> run after level 0 completes
    levels: dict[int, list[dict]] = {}
    sq_by_id: dict[int, dict] = {sq["id"]: sq for sq in sub_queries}

    def _get_level(sq: dict, memo: dict = {}) -> int:
        sid = sq["id"]
        if sid in memo:
            return memo[sid]
        deps = sq.get("depends_on", [])
        if not deps:
            memo[sid] = 0
            return 0
        level = 1 + max(_get_level(sq_by_id[d], memo) for d in deps if d in sq_by_id)
        memo[sid] = level
        return level

    for sq in sub_queries:
        level = _get_level(sq, {})
        levels.setdefault(level, []).append(sq)

    all_results = []
    completed_ids: set[int] = set()
    dedup_count = 0
    loop = asyncio.get_event_loop()

    # Execute level by level
    for level_num in sorted(levels.keys()):
        level_sqs = levels[level_num]

        # Check dependencies are met
        ready = []
        for sq in level_sqs:
            deps = sq.get("depends_on", [])
            if all(d in completed_ids for d in deps):
                ready.append(sq)

        # Run all ready sub-queries in parallel using thread pool
        async def _research_one(sq):
            return await loop.run_in_executor(
                _executor, research_sub_query, sq["query"]
            )

        tasks = [_research_one(sq) for sq in ready]
        findings_per_sq = await asyncio.gather(*tasks)

        for sq, findings in zip(ready, findings_per_sq):
            # Deduplicate by URL within this sub-query only
            local_seen = set()
            deduped = []
            for f in findings:
                url = f["source"]
                if url in local_seen:
                    dedup_count += 1
                    continue
                # Track globally for stats, but don't skip cross-query duplicates
                # Different sub-queries may extract different context from the same URL
                if url in seen_urls:
                    dedup_count += 1
                local_seen.add(url)
                seen_urls.add(url)
                deduped.append(f)

            all_results.append({
                "sub_query": sq,
                "findings": deduped,
            })
            completed_ids.add(sq["id"])

    return {"results": all_results, "dedup_count": dedup_count}
