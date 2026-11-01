# Empório da Música — Autonomous Customer Service Agent

An autonomous customer service agent for a musical instrument store in
Campo Grande, MS, Brazil. Built with FastAPI, Pandas, pdfplumber and Gemini Flash.

## Architecture

POST /chat receives {message, session_id}. The agent sends it to Gemini
with 4 registered tools. If Gemini requests a tool call, the dispatcher runs
the Python function and returns the JSON to Gemini. Loop until Gemini answers
in text (max 5 rounds).

## Tools

- get_product_info(product_name) — price, stock, status, specs
- get_active_promotions() — currently active promotions
- get_order_status(order_id, customer_email) — order status, privacy-gated
- get_store_policies() — full text of the store policy manual

## Design decisions

No vector database. The policy PDF is 8 pages. Gemini's context window is
far larger, so the full document is extracted once with lru_cache and returned
as a single string. No chunking, no embeddings, no external index.

Manual function-calling loop. start_chat(enable_automatic_function_calling=False).
Reasons: auditability of tool arguments, structured error handling, hard cap
of max_tool_rounds=5, and support for parallel tool requests.

Privacy-safe order lookup. get_order_status requires order_id + email, joins
orders.csv to customers.csv via customer_id, and returns the exact same
generic message for missing order and email mismatch. Prevents ID enumeration.
Never returns customer PII.

Anti-hallucination. Prompt forbids inventing prices, stock, tracking codes
or delivery dates. If a tool fails or returns empty, the model says it could
not retrieve the information.

## Requirements

Python 3.11+, a Google AI Studio API key with access to a Gemini Flash model.

## Setup

    git clone <your-repo-url>
    cd desafio_artefact
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    export GOOGLE_API_KEY="your-key-here"
    export GEMINI_MODEL="gemini-3.5-flash-lite"

## Run

    uvicorn agent:app --reload --port 8000

    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "Quanto custa o Takamine GD20?", "session_id": "demo-1"}'

Swagger at http://localhost:8000/docs

## Endpoints

- GET /health — liveness
- POST /chat — {message, session_id} to {response, session_id}
- DELETE /session/{session_id} — clears conversation

## Evaluation

    export GOOGLE_API_KEY="..."
    python3 evaluation/run_scenarios.py --delay 4

Covers: out-of-stock rule, price lookup, PIX non-stacking, order lookup
success, order lookup with wrong email, return policy, privacy refusal.

## Known limitations

- Sessions stored in memory (single instance only). Redis needed for multi-instance.
- No auth on /chat. Add webhook signature validation in production.
- Policy PDF cached per process; updates require a restart.
- No structured logging or metrics yet.

## With more time

Redis-backed sessions, structured tool-call logging, retry + backoff on
Gemini calls, LLM-as-judge regression tests, Docker + CI, human hand-off
on complaints or high-value refunds.

## Use of AI coding assistants

Developed with the help of Claude (Anthropic) and Gemini (Google). It was used for
architecture review, FastAPI boilerplate, SDK signature confirmation, and
drafting evaluation scenarios. Every tool was executed against the real CSVs
and every business rule was cross-checked against the policy PDF.
