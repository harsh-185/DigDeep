# n8n Workflow Setup

## Quick Start (Local)

```bash
# Terminal 1: Start the agent
cd deepdig
source venv/bin/activate
uvicorn agent.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start n8n
npx n8n
```

## Import Workflow

1. Open http://localhost:5678
2. Go to **Workflows** -> **Import from File**
3. Select `workflows/deepdig_main.json`
4. **Activate the workflow** (toggle in top-right corner)

**Important**: The workflow must be active for the production webhook URL to work. Test URLs (`/webhook-test/...`) only work when you click "Listen for test event" in the n8n editor.

## Workflow: DeepDig Main Research

**Production URL**: `POST http://localhost:5678/webhook/research`

**Flow**:
```
Webhook -> Research Iteration 1 -> Quality Gate -> Return Answer
                                       |
                                  (answer < 100 chars)
                                       |
                                       v
                                Follow-up Research -> Return Answer
```

1. Receives `{ "question": "..." }` via webhook
2. Calls FastAPI agent at `POST http://127.0.0.1:8000/research`
3. Quality gate checks answer length (> 100 chars = pass)
4. If insufficient: runs follow-up research with same session_id
5. Returns answer + steps + cost report

## Testing

```bash
# Production webhook (requires workflow to be active)
curl -X POST http://localhost:5678/webhook/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest developments in quantum computing?"}'

# Run demo through n8n
python -m demo.run_demo --n8n
```

## Network Notes

- n8n resolves `localhost` to IPv6 (`::1`) which may fail. The workflow uses `127.0.0.1` explicitly.
- The agent must bind to `0.0.0.0` (not just `127.0.0.1`) for n8n to reach it: `uvicorn agent.main:app --host 0.0.0.0 --port 8000`

## Node Details

| Node | Type | Purpose |
|---|---|---|
| Webhook | Webhook v2 | Receives POST with question, responds with last node output |
| Research Iteration 1 | HTTP Request v4.2 | Calls agent `/research` endpoint |
| Quality Gate | If v2 | Checks `answer.length < 100` |
| Follow-up Research | HTTP Request v4.2 | Gap-filling call with session_id |
| Return Answer | NoOp v1 | Pass-through for response |
