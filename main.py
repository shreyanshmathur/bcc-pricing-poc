import os
import base64
import json
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq import Groq

app = FastAPI(title="BCC AI Pricing Engine")

# Resolve index.html relative to this file so it works whether run locally
# with uvicorn or bundled inside a Netlify/Lambda function package.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

session_log: List[Dict[str, Any]] = []
_item_counter = 0
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

VISION_SYSTEM_PROMPT = """You are a senior buying specialist at Bombay Closet Cleanse (BCC), a premium pre-loved fashion platform in India. You have deep expertise in the Indian resale fashion market and clothing quality assessment.

BCC context:
- Average selling price: ₹500
- Intake cost: under ₹50 per item
- Model: 2x store credit vs cash payout
- Platforms we benchmark against: OLX, Carousell India, Instagram thrift accounts, Vinted, Depop India

High-desirability brands and categories for Indian consumers:
- Fast fashion: Zara, H&M, Mango, Marks & Spencer, Gap, Uniqlo, Pull&Bear, Bershka
- Indian contemporary: AND, W, Biba, FabIndia, Global Desi, Aurelia, Anokhi, Manyavar
- Sportswear: Nike, Adidas, Puma, Reebok, Under Armour, New Balance
- Premium/designer: Tommy Hilfiger, Calvin Klein, Ralph Lauren, Lacoste, Levi's, Wrangler, Diesel
- Trending: Vintage denim, Y2K silhouettes, Bollywood-adjacent styles, handloom, block-print, ikat, Kalamkari

Pricing guidelines (base reference):
- Condition 1/5 (heavily damaged): ₹99–₹149
- Condition 2/5 (worn): ₹150–₹299
- Condition 3/5 (fair): ₹299–₹499
- Condition 4/5 (good): ₹399–₹699
- Condition 5/5 (mint): ₹499–₹999
- Add 40–80% for recognised premium/designer brands
- Add 20–40% for high-demand categories or rarity signals

You MUST respond with ONLY a valid JSON object using these exact keys — no additional text, no markdown, no code blocks:
{
  "brand": "string — brand name if identifiable from tags, logo, stitching, or silhouette; otherwise 'Unbranded'",
  "category": "string — one of: Tops, Dresses, Denim, Trousers, Outerwear, Ethnic Wear, Activewear, Accessories, Footwear, Other",
  "condition_score": 1-5 integer,
  "condition_notes": "string — describe fabric quality, visible wear, fading, pilling, missing buttons, stains, structural integrity",
  "rarity_signals": ["array of specific signals, e.g. 'Limited edition colourway', 'Y2K silhouette', 'Vintage wash denim'"],
  "pricing_score": 1-10 integer,
  "price_low": integer in INR rounded to nearest 50,
  "price_high": integer in INR rounded to nearest 50,
  "confidence": "low" or "medium" or "high"
}

If the image is NOT a clothing or fashion item, respond with exactly:
{"error": "This image does not appear to show a clothing or fashion item and cannot be priced."}

Never include markdown, code blocks, or any text outside the JSON object."""

RATIONALE_SYSTEM_PROMPT = """You are a pricing analyst at Bombay Closet Cleanse (BCC), India's curated pre-loved fashion platform. Write brief, clear pricing rationale for non-technical buying staff. Be conversational, specific, and avoid jargon. Reference comparable prices on Indian resale platforms where possible."""


@app.get("/")
async def root():
    try:
        with open(os.path.join(_BASE_DIR, "index.html"), encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found. Ensure it is in the same directory as main.py.</h1>", status_code=500)


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    global _item_counter

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set. Please set the environment variable and restart the server."
        )

    # Validate content type
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    ct = (file.content_type or "").lower()
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ct not in allowed_types and ext not in {"jpg", "jpeg", "png", "webp"}:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WebP images are supported."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f} MB — maximum allowed size is 10 MB."
        )

    if ct == "image/png" or ext == "png":
        mime = "image/png"
    elif ct == "image/webp" or ext == "webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    b64 = base64.b64encode(contents).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    client = Groq(api_key=api_key)

    # Vision call
    try:
        vision_resp = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        },
                        {
                            "type": "text",
                            "text": "Analyse this clothing item and return your pricing assessment as a JSON object."
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.2
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Vision model request failed: {str(exc)}"
        )

    raw = ""
    try:
        raw = vision_resp.choices[0].message.content or ""
        data = json.loads(raw)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"Could not parse model response as JSON. Received: {raw[:300]}"
        )

    if "error" in data:
        raise HTTPException(status_code=422, detail=data["error"])

    required = [
        "brand", "category", "condition_score", "condition_notes",
        "rarity_signals", "pricing_score", "price_low", "price_high", "confidence"
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Model response missing required fields: {', '.join(missing)}"
        )

    try:
        data["condition_score"] = max(1, min(5, int(data["condition_score"])))
        data["pricing_score"] = max(1, min(10, int(data["pricing_score"])))
        data["price_low"] = int(data["price_low"])
        data["price_high"] = int(data["price_high"])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail=f"Non-integer numeric field: {exc}")

    if not isinstance(data["rarity_signals"], list):
        data["rarity_signals"] = [str(data["rarity_signals"])] if data["rarity_signals"] else []

    if data["price_low"] > data["price_high"]:
        data["price_low"], data["price_high"] = data["price_high"], data["price_low"]

    # Market rationale call
    rationale = ""
    try:
        signals_str = ", ".join(data["rarity_signals"]) if data["rarity_signals"] else "none noted"
        rationale_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RATIONALE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Write 2–3 conversational sentences explaining why "
                        f"₹{data['price_low']:,}–₹{data['price_high']:,} is the right price "
                        f"for this item at BCC.\n\n"
                        f"Item details:\n"
                        f"- Brand: {data['brand']}\n"
                        f"- Category: {data['category']}\n"
                        f"- Condition: {data['condition_score']}/5 — {data['condition_notes']}\n"
                        f"- Signals: {signals_str}\n"
                        f"- Pricing score: {data['pricing_score']}/10\n"
                        f"- Confidence: {data['confidence']}\n\n"
                        f"Reference comparable Indian resale platform prices where relevant. "
                        f"Be direct and practical — a buying staff member will use this to decide "
                        f"whether to accept or adjust the price."
                    )
                }
            ],
            max_tokens=280,
            temperature=0.7
        )
        rationale = rationale_resp.choices[0].message.content.strip()
    except Exception:
        rationale = (
            f"This {data['brand']} {data['category'].lower()} in {data['condition_score']}/5 "
            f"condition is recommended at ₹{data['price_low']:,}–₹{data['price_high']:,}, "
            f"consistent with BCC's positioning in the Indian pre-loved market. "
            f"At this price point the item should sell within 2–4 weeks and deliver a "
            f"healthy margin over intake cost."
        )

    _item_counter += 1
    entry: Dict[str, Any] = {
        "item_no": _item_counter,
        "timestamp": datetime.now().strftime("%H:%M"),
        "brand": data["brand"],
        "category": data["category"],
        "score": data["pricing_score"],
        "price_low": data["price_low"],
        "price_high": data["price_high"],
        "final_price": None,
        "action": "pending"
    }
    session_log.append(entry)

    return JSONResponse(content={
        "item_no": _item_counter,
        "brand": data["brand"],
        "category": data["category"],
        "condition_score": data["condition_score"],
        "condition_notes": data["condition_notes"],
        "rarity_signals": data["rarity_signals"],
        "pricing_score": data["pricing_score"],
        "price_low": data["price_low"],
        "price_high": data["price_high"],
        "confidence": data["confidence"],
        "market_rationale": rationale
    })


@app.post("/log-decision")
async def log_decision(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    item_no = body.get("item_no")
    action = body.get("action")
    final_price = body.get("final_price")

    if not isinstance(item_no, int) or action not in ("accepted", "overridden"):
        raise HTTPException(
            status_code=400,
            detail="item_no (int) and action ('accepted' or 'overridden') are required."
        )

    for entry in session_log:
        if entry["item_no"] == item_no:
            entry["action"] = action
            entry["final_price"] = final_price
            break

    return JSONResponse(content={"ok": True})


@app.get("/session-log")
async def get_session_log():
    return JSONResponse(content={"log": session_log[-10:]})


# ── Netlify / AWS Lambda handler ─────────────────────────────────────────────
# Mangum wraps the ASGI app so Netlify Functions (Lambda-compatible runtime)
# can invoke it. Importing this file with uvicorn ignores this export.
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None
