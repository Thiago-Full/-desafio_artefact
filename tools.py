"""
tools.py — Data access tools for the Empório da Música agent.
Includes: Product search, active promotions, order status with privacy checks,
and a simple full-document extraction (RAG) for the store policies.
"""

from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
import pdfplumber

# --- Paths Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PRODUCTS_CSV = DATA_DIR / "desafio_tecnico_ai_eng - products.csv"
PROMOTIONS_CSV = DATA_DIR / "desafio_tecnico_ai_eng - promotions.csv"
ORDERS_CSV = DATA_DIR / "desafio_tecnico_ai_eng - orders.csv"
CUSTOMERS_CSV = DATA_DIR / "desafio_tecnico_ai_eng - customers.csv"
CATEGORIES_CSV = DATA_DIR / "desafio_tecnico_ai_eng - categories.csv"
ORDER_ITEMS_CSV = DATA_DIR / "desafio_tecnico_ai_eng - order_items.csv"
POLICY_PDF = DATA_DIR / "políticas_da_loja.pdf"


# ---------------------------------------------------------------------------
# Tool 1: Product Search
# ---------------------------------------------------------------------------

def get_product_info(product_name: str) -> dict:
    """
    Searches for a product by name and returns its availability and price.
    Business Rule: If status is 'active' but stock is 0, it must return 'Out of stock'.
    """
    try:
        df = pd.read_csv(PRODUCTS_CSV)
        matches = df[df["name"].str.contains(product_name, case=False, na=False)]

        if matches.empty:
            return {"status": "error", "message": "Product not found in the catalog."}

        product = matches.iloc[0]

        status = str(product["status"]).strip()
        stock = int(product["stock_quantity"])

        if status == "active" and stock > 0:
            availability = "Available"
        elif status == "active" and stock <= 0:
            availability = "Out of stock"
        elif status == "discontinued":
            availability = "Discontinued"
        elif status == "coming_soon":
            availability = "Coming soon"
        else:
            availability = status

        return {
            "product_id": int(product["product_id"]),
            "product_name": str(product["name"]),
            "price_brl": float(product["price_brl"]),
            "stock_quantity": stock,
            "status": status,
            "availability": availability,
            "category_id": int(product["category_id"]),
            "specs": str(product["specs"]),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Tool 2: Active Promotions
# ---------------------------------------------------------------------------

def get_active_promotions() -> list:
    """
    Retrieves a list of all currently active promotions.
    """
    try:
        df = pd.read_csv(PROMOTIONS_CSV)
        active_promos = df[df["is_active"] == 1]

        promotions = []
        for _, row in active_promos.iterrows():
            promotions.append({
                "product_id": int(row["product_id"]),
                "discount_percent": float(row["discount_percent"]),
                "description": str(row["description"]),
            })
        return promotions
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


# ---------------------------------------------------------------------------
# Tool 3: Order Status (with Privacy checks)
# ---------------------------------------------------------------------------

_NOT_FOUND_RESPONSE = {
    "found": False,
    "message": "Order not found or customer email does not match.",
}


def get_order_status(order_id: str, customer_email: str) -> dict:
    """
    Retrieve order status only when BOTH order_id and customer_email match.
    Privacy rules: Same response for missing order/email mismatch. Never invent info.
    """
    order_id_clean = str(order_id or "").strip()
    email_clean = str(customer_email or "").strip().lower()

    if not order_id_clean or not email_clean:
        return _NOT_FOUND_RESPONSE.copy()

    try:
        orders = pd.read_csv(ORDERS_CSV, dtype=str, keep_default_na=False)
        customers = pd.read_csv(CUSTOMERS_CSV, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        return {"found": False, "message": "Order data is unavailable."}

    customers["_email_clean"] = customers["email"].str.strip().str.lower()
    matching_customers = customers[customers["_email_clean"] == email_clean]
    if matching_customers.empty:
        return _NOT_FOUND_RESPONSE.copy()

    customer_ids = set(matching_customers["customer_id"].astype(str).str.strip())

    orders["_order_id_clean"] = orders["order_id"].astype(str).str.strip()
    orders["_customer_id_clean"] = orders["customer_id"].astype(str).str.strip()

    matched = orders[
        (orders["_order_id_clean"] == order_id_clean)
        & (orders["_customer_id_clean"].isin(customer_ids))
    ]

    if matched.empty:
        return _NOT_FOUND_RESPONSE.copy()
    if len(matched) > 1:
        return {
            "found": False,
            "message": "Multiple orders match. Please contact support.",
        }

    row = matched.iloc[0]

    result = {
        "found": True,
        "order_id": str(row["order_id"]).strip(),
        "status": str(row["status"]).strip(),
        "order_date": str(row["order_date"]).strip(),
        "total_brl": float(row["total_brl"]) if str(row["total_brl"]).strip() else None,
        "payment_method": str(row["payment_method"]).strip(),
    }

    for field in ("tracking_code", "estimated_delivery"):
        value = str(row.get(field, "")).strip()
        if value:
            result[field] = value

    return result


# ---------------------------------------------------------------------------
# Tool 4: Simple RAG - Store Policies
# ---------------------------------------------------------------------------

def _clean_policy_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@lru_cache(maxsize=1)
def _load_policy_text() -> str:
    if not POLICY_PDF.exists():
        raise FileNotFoundError(f"Policy PDF not found at: {POLICY_PDF}")

    pages: list[str] = []
    with pdfplumber.open(POLICY_PDF) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_policy_text(raw)
            pages.append(f"--- Page {idx} ---\n{cleaned}")
    return "\n\n".join(pages)


def get_store_policies() -> dict:
    """
    Return the full text of the Empório da Música internal policy manual.
    The whole document is returned in a single call — the LLM is expected
    to locate the relevant rule itself.
    """
    try:
        text = _load_policy_text()
    except FileNotFoundError:
        return {"success": False, "error": "Policy document is currently unavailable."}
    except Exception as exc:
        return {"success": False, "error": f"Failed to read policy document: {type(exc).__name__}"}

    if not text.strip():
        return {"success": False, "error": "Policy document is empty or unreadable."}

    return {
        "success": True,
        "source": "políticas_da_loja.pdf",
        "page_count": len(text.split("--- Page ")) - 1,
        "char_count": len(text),
        "content": text,
    }