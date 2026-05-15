/**
 * BCC AI Pricing Engine — Netlify Function (v2, JavaScript)
 * Zero npm dependencies. Calls Groq HTTP API directly.
 *
 * Pipeline (per analyse request):
 *  1. Vision  — llama-4-scout  → identifies ALL clothing items in photos, prices each individually
 *  2. Search  — compound-beta  → live market search for EACH item in parallel
 *  3. Rationale — llama-3.3-70b → per-item rationale grounded in real market data (parallel)
 *
 * Key rotation: round-robin across GROQ_API_KEY and GROQ_API_KEY_2.
 */

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

let _sessionLog  = [];
// Seeded from timestamp to reduce item_no collisions after a cold-start reset (POC tradeoff)
let _itemCounter = Date.now() % 100000;
let _keyIndex    = 0;

// ── API key rotation ──────────────────────────────────────────────────────────

function pickApiKey() {
  const keys = [
    process.env.GROQ_API_KEY,
    process.env.GROQ_API_KEY_2,
  ].filter(Boolean);
  if (!keys.length) return null;
  const key = keys[_keyIndex % keys.length];
  _keyIndex++;
  return key;
}

// ── Prompts ───────────────────────────────────────────────────────────────────

const VISION_PROMPT = `You are a senior buying specialist at Bombay Closet Cleanse (BCC), a premium pre-loved fashion platform in India. You have deep expertise in the Indian resale fashion market and clothing quality assessment.

BCC context:
- Average selling price: Rs500 | Intake cost: under Rs50 | Model: 2x store credit vs cash
- Benchmark platforms: OLX, Carousell India, Instagram thrift accounts

High-desirability brands for Indian consumers:
- Fast fashion: Zara, H&M, Mango, Marks & Spencer, Gap, Uniqlo, Pull&Bear
- Indian contemporary: AND, W, Biba, FabIndia, Global Desi, Aurelia, Anokhi, Manyavar
- Sportswear: Nike, Adidas, Puma, Reebok, Under Armour, New Balance
- Premium/designer: Tommy Hilfiger, Calvin Klein, Ralph Lauren, Lacoste, Levi's, Wrangler
- Trending: Vintage denim, Y2K silhouettes, handloom, block-print, ikat

Pricing guidelines (per item):
- Condition 1/5: Rs99-Rs149 | 2/5: Rs150-Rs299 | 3/5: Rs299-Rs499
- Condition 4/5: Rs399-Rs699 | 5/5: Rs499-Rs999
- Add 40-80% for premium/designer brands; 20-40% for rarity signals

CRITICAL: Identify and price EVERY separate clothing/fashion piece visible in the image individually.
BCC buys each piece separately, so a suit must be broken into jacket + trousers (+ waistcoat if present).

Splitting rules:
- Suit → jacket + trousers + waistcoat (if 3-piece) = 2 or 3 separate items
- Outfit with top + bottom → each is a separate item
- Accessories (tie, belt, shoes, bag) → each is its own item
- Only combine if items are genuinely sold as a single inseparable unit (e.g. a dress is one item)

Photos may show items flat, on a hanger, or worn by a person — assess the garments, not the person.

Respond ONLY with valid JSON — no markdown, no text outside JSON:
{
  "items": [
    {
      "item_description": "concise label e.g. 'Navy blue suit jacket' or 'Red paisley tie'",
      "brand": "brand name or 'Unbranded'",
      "category": "one of: Tops, Dresses, Denim, Trousers, Outerwear, Ethnic Wear, Activewear, Accessories, Footwear, Other",
      "condition_score": 1-5 integer,
      "condition_notes": "fabric quality, wear, fading, pilling, structural integrity",
      "rarity_signals": ["specific signals"],
      "pricing_score": 1-10 integer,
      "price_low": integer INR rounded to nearest 50,
      "price_high": integer INR rounded to nearest 50,
      "confidence": "low" or "medium" or "high"
    }
  ]
}

If the image contains NO clothing at all (landscape, food, animal, blank wall):
{"error": "This image does not appear to show a clothing item."}`;

const RATIONALE_PROMPT = `You are a pricing analyst at Bombay Closet Cleanse (BCC), India's pre-loved fashion platform. Write 2-3 conversational sentences for non-technical buying staff. Be specific, reference Indian resale platforms, and use any real market data provided. No jargon.`;

// ── Helpers ───────────────────────────────────────────────────────────────────

function corsHeaders() {
  return {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Accept',
  };
}

function ok(data)        { return new Response(JSON.stringify(data), { status: 200, headers: corsHeaders() }); }
function err(status, msg){ return new Response(JSON.stringify({ detail: msg }), { status, headers: corsHeaders() }); }

async function groqChat(model, messages, extra, apiKey) {
  return groqChatWithSignal(model, messages, extra, apiKey, null);
}

/** Same as groqChat but accepts an AbortSignal for timeout/cancellation. */
async function groqChatWithSignal(model, messages, extra, apiKey, signal) {
  const resp = await fetch(GROQ_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages, ...extra }),
    ...(signal ? { signal } : {}),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Groq ${resp.status}: ${txt.slice(0, 300)}`);
  }
  const json = await resp.json();
  const content = json?.choices?.[0]?.message?.content;
  if (content == null)
    throw new Error(`Groq returned no content (choices: ${JSON.stringify(json?.choices ?? [])})`);
  return content;
}

/** Live market price search — returns null on failure (best-effort).
 *  Hard timeout: 25 s so a slow web-search never blocks the whole pipeline. */
async function fetchMarketPrices(brand, category, itemDescription, apiKey) {
  if (!brand || brand.toLowerCase() === 'unbranded') return null;
  try {
    const query = (itemDescription
      ? `${brand} ${itemDescription}`
      : `${brand} ${category}`).replace(/\s+/g, ' ').trim();

    const TIMEOUT_MS = 25_000;
    const controller  = new AbortController();
    const timer       = setTimeout(() => controller.abort(), TIMEOUT_MS);

    const result = await groqChatWithSignal(
      'compound-beta',
      [
        {
          role: 'system',
          content: 'You are a pricing researcher for Indian fashion resale. Use web search. 2-3 sentences, specific INR amounts, India market only.',
        },
        {
          role: 'user',
          content:
            `Search: (1) Current retail/MRP price of ${query} sold new in India (INR). ` +
            `(2) Current second-hand resale price of ${query} on OLX India, Carousell India, or Indian Instagram thrift accounts (INR). ` +
            `Give specific price ranges. Be brief.`,
        },
      ],
      { max_tokens: 300, temperature: 0.1 },
      apiKey,
      controller.signal
    ).finally(() => clearTimeout(timer));

    return result ? result.trim() : null;
  } catch (_) {
    return null;   // timeout, network error, or model error — degrade gracefully
  }
}

/** Generate rationale for one item. Returns fallback string on failure. */
async function generateRationale(item, marketPrices, apiKey) {
  try {
    const signals       = (item.rarity_signals || []).join(', ') || 'none noted';
    const marketContext = marketPrices ? `\nLive market research:\n${marketPrices}\n` : '';
    const text = await groqChat(
      'llama-3.3-70b-versatile',
      [
        { role: 'system', content: RATIONALE_PROMPT },
        { role: 'user', content:
            `Explain why Rs${item.price_low}-Rs${item.price_high} is the right BCC intake price for this item.${marketContext}\n` +
            `Item: ${item.item_description || item.category} | Brand: ${item.brand} | Condition: ${item.condition_score}/5 — ${item.condition_notes} | ` +
            `Signals: ${signals} | Score: ${item.pricing_score}/10\n` +
            `Reference Indian resale platforms and live market data if provided.`
        },
      ],
      { max_tokens: 240, temperature: 0.7 },
      apiKey
    );
    return text.trim();
  } catch {
    return `This ${item.brand} ${(item.item_description || item.category).toLowerCase()} in ${item.condition_score}/5 condition is recommended at Rs${item.price_low}–Rs${item.price_high}.`;
  }
}

/** Sanitise and clamp numeric fields on a raw item object. */
function sanitiseItem(raw) {
  const item = { ...raw };
  item.condition_score = Math.max(1, Math.min(5, parseInt(item.condition_score, 10) || 3));
  item.pricing_score   = Math.max(1, Math.min(10, parseInt(item.pricing_score, 10) || 5));
  item.price_low       = parseInt(item.price_low,  10) || 299;
  item.price_high      = parseInt(item.price_high, 10) || 499;
  if (!Array.isArray(item.rarity_signals))
    item.rarity_signals = item.rarity_signals ? [String(item.rarity_signals)] : [];
  if (item.price_low > item.price_high)
    [item.price_low, item.price_high] = [item.price_high, item.price_low];
  item.item_description = item.item_description || item.category;
  // Fill optional fields that the model occasionally omits
  item.confidence       = item.confidence      || 'medium';
  item.condition_notes  = item.condition_notes || 'No notes provided.';
  return item;
}

// ── Main handler ──────────────────────────────────────────────────────────────

export default async function handler(req) {
  const url    = new URL(req.url);
  const path   = url.pathname;
  const method = req.method.toUpperCase();

  if (method === 'OPTIONS') return new Response('', { status: 200, headers: corsHeaders() });

  // ── POST /analyse ──────────────────────────────────────────────────────────
  if (path.endsWith('/analyse') && method === 'POST') {
    const apiKey = pickApiKey();
    if (!apiKey) return err(500, 'GROQ_API_KEY is not configured on the server.');

    let formData;
    try { formData = await req.formData(); }
    catch (e) { return err(400, `Could not parse form data: ${e.message}`); }

    const rawFiles = formData.getAll('file').filter(f => f && typeof f !== 'string');
    if (!rawFiles.length) return err(400, 'No file found in request. Upload a JPEG or PNG image.');

    const files = rawFiles.slice(0, 3);

    let totalBytes = 0;
    for (const f of files) {
      if (f.size > 10 * 1024 * 1024)
        return err(413, `One file is ${(f.size / 1024 / 1024).toFixed(1)} MB — max 10 MB per photo.`);
      totalBytes += f.size;
    }
    if (totalBytes > 25 * 1024 * 1024)
      return err(413, `Total upload is ${(totalBytes / 1024 / 1024).toFixed(1)} MB — max 25 MB combined.`);

    // Server-side MIME allowlist (client type header can be spoofed)
    const ALLOWED_MIME = new Set(['image/jpeg','image/jpg','image/png','image/webp']);
    for (const f of files) {
      if (!ALLOWED_MIME.has((f.type || '').toLowerCase()))
        return err(415, `File type "${f.type}" is not supported. Please upload JPEG, PNG, or WebP images.`);
    }

    // Build image blocks
    const imageBlocks = await Promise.all(files.map(async file => {
      const buf  = await file.arrayBuffer();
      const b64  = Buffer.from(buf).toString('base64');
      const mime = file.type || 'image/jpeg';
      return { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } };
    }));

    // ── Step 1: Vision — identify ALL items ─────────────────────────────────
    let visionRaw;
    try {
      const photoNote = files.length > 1
        ? `Analyse these ${files.length} photos (all of the same outfit/item). Identify every separate clothing piece visible and return the full items array.`
        : 'Analyse this image. Identify every separate clothing piece visible and return the full items array.';

      visionRaw = await groqChat(
        'meta-llama/llama-4-scout-17b-16e-instruct',
        [
          { role: 'system', content: VISION_PROMPT },
          { role: 'user', content: [...imageBlocks, { type: 'text', text: photoNote }] },
        ],
        { response_format: { type: 'json_object' }, max_tokens: 2048, temperature: 0.2 },
        apiKey
      );
    } catch (e) { return err(502, `Vision model request failed: ${e.message}`); }

    let vdata;
    try { vdata = JSON.parse(visionRaw); }
    catch { return err(502, `Could not parse model response: ${String(visionRaw).slice(0, 200)}`); }

    if (vdata.error) return err(422, vdata.error);

    // Normalise: accept both new {items:[]} format and legacy single-item format
    let rawItems = [];
    if (Array.isArray(vdata.items) && vdata.items.length > 0) {
      rawItems = vdata.items;
    } else if (vdata.brand) {
      // Legacy single-item fallback
      rawItems = [vdata];
    } else {
      return err(502, 'Model returned no items. Please try again with a clearer photo.');
    }

    const ITEM_REQUIRED = ['brand','category','condition_score','condition_notes',
                           'rarity_signals','pricing_score','price_low','price_high','confidence'];

    const items = rawItems.slice(0, 6).map(sanitiseItem);  // cap at 6 items

    // ── Step 2: Market searches — all items in parallel ──────────────────────
    const marketPricesArr = await Promise.all(
      items.map(item => fetchMarketPrices(item.brand, item.category, item.item_description, pickApiKey()))
    );

    // ── Step 3: Rationales — all items in parallel ───────────────────────────
    const rationalesArr = await Promise.all(
      items.map((item, i) => generateRationale(item, marketPricesArr[i], pickApiKey()))
    );

    // ── Build response + update session log ──────────────────────────────────
    const now = new Date();
    const ts  = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;

    const responseItems = items.map((item, i) => {
      _itemCounter++;
      const itemNo = _itemCounter;
      _sessionLog.push({
        item_no: itemNo, timestamp: ts,
        brand: item.brand,
        description: item.item_description,
        category: item.category,
        score: item.pricing_score, photo_count: files.length,
        price_low: item.price_low, price_high: item.price_high,
        final_price: null, action: 'pending',
      });
      return {
        item_no: itemNo,
        item_description: item.item_description,
        ...Object.fromEntries(ITEM_REQUIRED.map(k => [k, item[k]])),
        market_prices: marketPricesArr[i],
        market_rationale: rationalesArr[i],
      };
    });

    return ok({
      photo_count: files.length,
      item_count: responseItems.length,
      items: responseItems,
    });
  }

  // ── POST /log-decision ─────────────────────────────────────────────────────
  if (path.endsWith('/log-decision') && method === 'POST') {
    let body;
    try { body = await req.json(); } catch { return err(400, 'Invalid JSON body.'); }
    const { item_no, action, final_price } = body;
    if (!Number.isInteger(item_no) || !['accepted','overridden'].includes(action))
      return err(400, "item_no (int) and action ('accepted'|'overridden') are required.");
    const entry = _sessionLog.find(e => e.item_no === item_no);
    if (entry) { entry.action = action; entry.final_price = final_price; }
    return ok({ ok: true });
  }

  // ── GET /session-log ───────────────────────────────────────────────────────
  if (path.endsWith('/session-log') && method === 'GET') {
    return ok({ log: _sessionLog.slice(-20) });
  }

  return err(404, `Route not found: ${method} ${path}`);
}

export const config = {
  path: ['/analyse', '/log-decision', '/session-log'],
};
