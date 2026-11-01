"""
agent.py — Empório da Música Customer Service Agent.

FastAPI + Gemini 1.5 Flash + manual Function Calling loop.
"""

from __future__ import annotations

import traceback
import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tools import (
    get_active_promotions,
    get_order_status,
    get_product_info,
    get_store_policies,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the virtual assistant of Empório da Música, a musical instrument store in Campo Grande, MS, Brazil.

PERSONA
- Warm, informal but professional. You sound like a knowledgeable friend who works at the store, not a corporate robot.
- Objective and concise. Customers are messaging on WhatsApp; long paragraphs are bad.
- Match the language of the customer. If they write in Portuguese, answer in Portuguese. If they write in English, answer in English.

SCOPE
- The store sells ONLY musical instruments (guitars, basses, acoustic guitars, drums, keyboards, wind instruments, orchestral strings, ukuleles).
- The store does NOT sell accessories: strings, picks, cables, cases, pedals, amplifiers. If asked, politely say the store does not carry them and suggest a partner store.
- If the question is completely outside the store's scope (politics, weather, coding help, etc.), politely redirect to store-related topics.

DATA & TOOLS
- NEVER invent prices, stock quantities, delivery dates, tracking codes, or order statuses. Always call the appropriate tool first.
- get_product_info(product_name): use for price, stock, specs, or availability of a specific instrument.
- get_active_promotions(): use to check which promotions are currently active and their discount percentages.
- get_order_status(order_id, customer_email): use ONLY when the customer asks about an order. You MUST ask for both the order number AND the email used at purchase before calling this tool. If the customer provides only one, ask for the other.
- get_store_policies(): use for business hours, address, payment methods, installments, shipping rules, returns/exchanges, warranty, or any policy question.

CRITICAL BUSINESS RULES
1. Stock: if a product's status is "active" but stock_quantity is 0, say it is "Out of stock". Never say "available" in that case.
2. PIX discount: the store offers 5% off for PIX payments. This discount CANNOT be combined with products that already have an active promotion. If a product has an active promotion, the PIX discount does not stack on top of it.
3. Privacy: never reveal data from other customers. get_order_status already enforces order_id + customer_email matching. If the tool returns "not found", say exactly that the order was not found or the email did not match — do not speculate about why.
4. No hallucinations: if a tool returns empty or an error, say you could not retrieve the information right now instead of making something up. Do not invent tracking codes or delivery estimates.
5. Discontinued products: if status is "discontinued", say it is no longer in the catalog and offer equivalent models if available.
6. Coming soon: if status is "coming_soon", say it is not yet available for purchase.

TONE EXAMPLES
- Good: "Nice! The Yamaha C40 is going for R$ 599.90 and we have 12 in stock. Want me to set one aside for you?"
- Bad: "Dear customer, we hereby inform you that the aforementioned product is available in our systems."
- Good: "I couldn't find that order with this email. Can you double-check the order number and the email used at purchase?"
- Bad: "The order does not exist." (too blunt, assumes the order doesn't exist when it might just be an email mismatch)
"""

# ---------------------------------------------------------------------------
# Tool declarations for Gemini
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "get_product_info",
        "description": (
            "Look up a musical instrument by name in the store catalog. "
            "Returns price in BRL, stock quantity, status (active / discontinued / coming_soon), "
            "description, and technical specs. Use this for any question about price, "
            "availability, or product details. The search is fuzzy/partial match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": (
                        "Product name or partial name, e.g. 'Yamaha C40', "
                        "'Takamine GD20', 'tenor ukulele'."
                    ),
                }
            },
            "required": ["product_name"],
        },
    },
    {
        "name": "get_active_promotions",
        "description": (
            "Return all currently active promotions (is_active = 1), including "
            "product_id, discount_percent, and promotion description. Use this "
            "whenever the customer asks about discounts, sales, or whether a "
            "specific product is on promotion."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_order_status",
        "description": (
            "Check the status of an order. Requires BOTH the order_id AND the "
            "customer_email used at purchase. Returns status, order date, total, "
            "payment method, and tracking code if available. "
            "If the email does not match the order, returns a generic not-found "
            "message — do not speculate or retry with a different order id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The numeric order id, e.g. '9' or '15'.",
                },
                "customer_email": {
                    "type": "string",
                    "description": "The email address used when placing the order.",
                },
            },
            "required": ["order_id", "customer_email"],
        },
    },
    {
        "name": "get_store_policies",
        "description": (
            "Return the full text of the Empório da Música internal policy manual. "
            "Use this for business hours, address, payment methods (PIX, credit, "
            "boleto), installment rules, shipping & delivery, returns/exchanges, "
            "warranty, promotions rules, or data privacy. Only call once per "
            "conversation — the full document fits in context."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_product_info": get_product_info,
    "get_active_promotions": get_active_promotions,
    "get_order_status": get_order_status,
    "get_store_policies": get_store_policies,
}


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name and always return a dict (never raise to Gemini)."""
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return {"success": False, "error": f"Unknown tool: {name}"}

    try:
        result = func(**args)
        if not isinstance(result, dict):
            return {"success": True, "result": result}
        return result
    except TypeError as exc:
        return {
            "success": False,
            "error": f"Invalid arguments for {name}: {exc}",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Tool {name} failed: {type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def build_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        tools=[{"function_declarations": TOOL_DECLARATIONS}],
    )


MODEL = build_model()

# In-memory session store. Acceptable for a prototype; replace with Redis
# or a DB for a multi-instance deployment.
SESSIONS: dict[str, genai.ChatSession] = {}


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def run_agent_turn(
    chat: genai.ChatSession,
    user_message: str,
    max_tool_rounds: int = 5,
) -> str:
    """
    Send a user message, execute any requested tools, and return the final
    text response. The loop is bounded to avoid infinite tool-calling.
    """
    response = chat.send_message(user_message)

    for _ in range(max_tool_rounds):
        function_calls = [
            part.function_call
            for part in response.parts
            if getattr(part, "function_call", None)
        ]

        if not function_calls:
            break

        function_responses = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            tool_result = execute_tool(fc.name, args)
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": tool_result},
                    )
                )
            )

        response = chat.send_message(function_responses)

    try:
        return response.text
    except ValueError:
        return (
            "Sorry, I could not complete your request right now. "
            "Could you rephrase your question?"
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Empório da Música Agent",
    description="Autonomous customer service agent for a musical instrument store.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Customer message")
    session_id: str = Field(
        default="default",
        description="Conversation identifier. Reuse the same id to keep context.",
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id.strip() or "default"

    if session_id not in SESSIONS:
        SESSIONS[session_id] = MODEL.start_chat(
            history=[],
            enable_automatic_function_calling=False,
        )

    try:
        answer = run_agent_turn(SESSIONS[session_id], payload.message)
    except Exception as exc:
        traceback.print_exc()  # Isto vai cuspir o erro real no terminal
        raise HTTPException(
            status_code=500,
            detail="Internal agent error. Please try again.",
        ) from exc

    return ChatResponse(response=answer, session_id=session_id)


@app.delete("/session/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}