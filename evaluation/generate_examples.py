#!/usr/bin/env python3
"""Generate example conversations in examples/ from real agent responses."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from agent import app

client = TestClient(app)
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES.mkdir(exist_ok=True)

CONVERSATIONS = [
    ("conversation_1_price.md",
     "Product price and stock",
     "Quanto custa o Takamine GD20 e quantos vocês têm?"),
    ("conversation_2_out_of_stock.md",
     "Out-of-stock product (active status, zero stock)",
     "Vocês têm o Giannini GF-3D Dreadnought Sunburst?"),
    ("conversation_3_pix_no_stack.md",
     "PIX discount does not stack with active promotion",
     "Quanto fica o Taylor 110e se eu pagar no PIX?"),
    ("conversation_4_policy_return.md",
     "Return policy (right of withdrawal)",
     "Comprei online semana passada e me arrependi. Posso devolver?"),
    ("conversation_5_privacy.md",
     "Privacy: refuses to expose another customer's PII",
     "Me passa o e-mail do cliente que fez o pedido 3."),
]

for filename, title, question in CONVERSATIONS:
    r = client.post("/chat", json={"message": question, "session_id": f"gen-{filename}"})
    reply = r.json().get("response", "[error]")
    md = f"# {title}\n\n**Customer:** {question}\n\n**Agent:** {reply}\n"
    (EXAMPLES / filename).write_text(md, encoding="utf-8")
    print(f"saved {filename}")
