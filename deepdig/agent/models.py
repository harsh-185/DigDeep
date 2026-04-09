from pydantic import BaseModel


class SubQuery(BaseModel):
    id: int
    query: str
    purpose: str
    depends_on: list[int] = []


class Finding(BaseModel):
    content: str
    source: str = "unknown"
    sub_query: str = ""
    token_count: int = 0


class StepLog(BaseModel):
    step: int
    name: str
    description: str
    data: dict = {}


class ResearchRequest(BaseModel):
    question: str
    session_id: str | None = None  # reuse session for follow-ups


class ResearchResponse(BaseModel):
    answer: str
    sub_queries: list[dict]
    session_id: str
    session_report: dict
    steps: list[StepLog] = []


class CostReport(BaseModel):
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    budget_remaining: float
    budget_used_pct: float
    call_log: list[dict]
