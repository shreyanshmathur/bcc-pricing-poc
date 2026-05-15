/**
 * BCC AI Pricing Engine — Netlify Function (v2, JavaScript)
 * Zero npm dependencies — uses built-in fetch + Buffer + req.formData().
 * Calls Groq HTTP API directly; no groq-sdk package required.
 *
 * Pipeline:
 *  1. Vision  — llama-4-scout  → brand, category, condition, initial price
 *  2. Search  — compound-beta  → live retail MRP + Indian resale prices (web search)
 *  3. Rationale — llama-3.3-70b → human-readable justification using real market data
 */

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

// In-memory state — persists across warm Lambda invocations
let _sessionLog = [];
let _itemCounter = 0;

// ── Prompts ───────────────────────────────────────────────────────────────────

const VISION_PROMPT = `You are a senior buying specialist at Bombay Closet Cleanse (BCC), a premium pre-loved fashion platform in India. You have deep expertise in the Indian resale fashion market and clothing quality assessment.

BCC context:
- Average selling price: Rs500 | Intake cost: under Rs50 | Model: 2x store credit vs cash
- Benchmark platforms: OLX, Carousell India, Instagram thrift accounts, Vinted, Depop India

High-desirability brands for Indian consumers:
- Fast fashion: Zara, H&M, Mango, Marks & Spencer, Gap, Uniqlo, Pull&Bear
- Indian contemporary: AND, W, Biba, FabIndia, Global Desi, Aurelia, Anokhi, Manyavar
- Sportswear: Nike, Adidas, Puma, Reebok, Under Armour, New Balance
- Premium/designer: Tommy Hilfiger, Calvin Klein, Ralph Lauren, Lacoste, Levi's, Wrangler
- Trending: Vintage denim, Y2K silhouettes, Bollywood-adjacent styles, handloom, block-print, ikat

Pricing guidelines:
- Condition 1/5: Rs99-Rs149 | 2/5: Rs150-Rs299 | 3/5: Rs299-Rs499
- Condition 4/5: Rs399-Rs699 | 5/5: Rs499-Rs999
- Add 40-80% for premium/designer brands; 20-40% for rarity signals

Respond ONLY with a valid JSON object — no markdown, no text outside JSON:
{
  "brand": "brand name or 'Unbranded'",
  "category": "one of: Tops, Dresses, Denim, Trousers, Outerwear, Ethnic Wear, Activewear, Accessories, Footwear, Other",
  "condition_score": 1-5 integer,
  "condition_notes": "describe fabric quality, wear, fading, pilling, structural integrity",
  "rarity_signals": ["array of specific signals"],
  "pricing_score": 1-10 integer,
  "price_low": integer INR rounded to nearest 50,
  "price_high": integer INR rounded to nearest 50,
  "confidence": "low" or "medium" or "high"
}

If NOT a clothing/fashion item: {"error": "This image does not appear to show a clothing item."}`;

const RATIONALE_PROMPT = `You are a pricing analyst at Bombay Closet Cleanse (BCC), India's pre-loved fashion platform. Write 2-3 conversational sentences of market rationale for non-technical buying staff. Be specific, reference Indian resale platforms where relevant, and incorporate any real market price data provided. No jargon.`;

// ── Helpers ───────────────────────────────────────────────────────────────────

function corsHeaders() {
  return {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Accept',
  };
}

function ok(data) {
  return new Response(JSON.stringify(data), { status: 200, headers: corsHeaders() });
}

function err(status, msg) {
  return new Response(JSON.stringify({ detail: msg }), { status, headers: corsHeaders() });
}

async function groqChat(model, messages, extra, apiKey) {
  const resp = await fetch(GROQ_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages, ...extra }),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Groq ${resp.status}: ${txt.slice(0, 300)}`);
  }
  const json = await resp.json();
  return json.choices[0].message.content;
}

/**
 * Fetch live retail MRP + Indian resale prices using Groq compound-beta (web search).
 * Returns a brief 2-3 sentence summary, or null on any failure (graceful degradation).
 */
async function fetchMarketPrices(brand, category, apiKey) {
  // Unbranded items have no useful market benchmark — skip
  if (!brand || brand.toLowerCase() === 'unbranded') return null;

  try {
    const result = await groqChat(
      'compound-beta',
      [
        {
          role: 'system',
          content: 'You are a pricing researcher for Indian fashion resale. Use web search to find current prices. Be concise — answer in 2-3 sentences with specific INR amounts. Focus on India market only.',
        },
        {
          role: 'user',
          content:
            `Search online and tell me: ` +
            `(1) What is the current retail/MRP price of ${brand} ${category} sold new in India (INR)? ` +
            `(2) What are people currently selling second-hand ${brand} ${category} for on OLX India, Carousell India, or Indian Instagram thrift accounts (INR)? ` +
            `Give specific price ranges. Be brief.`,
        },
      ],
      { max_tokens: 350, temperature: 0.1 },
      apiKey
    );
    return result ? result.trim() : null;
  } catch (_) {
    // Web search is best-effort — never fail the whole pipeline
    return null;
  }
}

// ── Main handler ──────────────────────────────────────────────────────────────

export default async function handler(req) {
  const url = new URL(req.url);
  const path = url.pathname;
  const method = req.method.toUpperCase();

  if (method === 'OPTIONS') {
    return new Response('', { status: 200, headers: corsHeaders() });
  }

  // ── POST /analyse ──────────────────────────────────────────────────────────
  if (path.endsWith('/analyse') && method === 'POST') {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) return err(500, 'GROQ_API_KEY is not configured on the server.');

    // Parse multipart form data — native in Node 18+ / Netlify v2
    let formData;
    try {
      formData = await req.formData();
    } catch (e) {
      return err(400, `Could not parse form data: ${e.message}`);
    }

    const file = formData.get('file');
    if (!file || typeof file === 'string') {
      return err(400, 'No file found in request. Upload a JPEG or PNG image.');
    }

    const arrayBuf = await file.arrayBuffer();
    const bytes = arrayBuf.byteLength;
    if (bytes > 10 * 1024 * 1024) {
      return err(413, `File is ${(bytes / 1024 / 1024).toFixed(1)} MB — max is 10 MB.`);
    }

    const b64 = Buffer.from(arrayBuf).toString('base64');
    const mime = file.type || 'image/jpeg';
    const dataUrl = `data:${mime};base64,${b64}`;

    // ── Step 1: Vision analysis ──────────────────────────────────────────────
    let visionRaw;
    try {
      visionRaw = await groqChat(
        'meta-llama/llama-4-scout-17b-16e-instruct',
        [
          { role: 'system', content: VISION_PROMPT },
          { role: 'user', content: [
            { type: 'image_url', image_url: { url: dataUrl } },
            { type: 'text', text: 'Analyse this clothing item and return your pricing assessment as a JSON object.' },
          ]},
        ],
        { response_format: { type: 'json_object' }, max_tokens: 1024, temperature: 0.2 },
        apiKey
      );
    } catch (e) {
      return err(502, `Vision model request failed: ${e.message}`);
    }

    let vdata;
    try {
      vdata = JSON.parse(visionRaw);
    } catch {
      return err(502, `Could not parse model response: ${String(visionRaw).slice(0, 200)}`);
    }

    if (vdata.error) return err(422, vdata.error);

    const REQUIRED = ['brand','category','condition_score','condition_notes',
                      'rarity_signals','pricing_score','price_low','price_high','confidence'];
    const missing = REQUIRED.filter(k => !(k in vdata));
    if (missing.length) return err(502, `Model response missing fields: ${missing.join(', ')}`);

    vdata.condition_score = Math.max(1, Math.min(5, parseInt(vdata.condition_score) || 3));
    vdata.pricing_score   = Math.max(1, Math.min(10, parseInt(vdata.pricing_score) || 5));
    vdata.price_low       = parseInt(vdata.price_low)  || 299;
    vdata.price_high      = parseInt(vdata.price_high) || 499;
    if (!Array.isArray(vdata.rarity_signals)) {
      vdata.rarity_signals = vdata.rarity_signals ? [String(vdata.rarity_signals)] : [];
    }
    if (vdata.price_low > vdata.price_high) {
      [vdata.price_low, vdata.price_high] = [vdata.price_high, vdata.price_low];
    }

    // ── Step 2: Live market price search (web) ───────────────────────────────
    const marketPrices = await fetchMarketPrices(vdata.brand, vdata.category, apiKey);

    // ── Step 3: Rationale — grounded in real market data ────────────────────
    let rationale = '';
    try {
      const signals = vdata.rarity_signals.join(', ') || 'none noted';
      const marketContext = marketPrices
        ? `\nLive market research:\n${marketPrices}\n`
        : '';
      rationale = await groqChat(
        'llama-3.3-70b-versatile',
        [
          { role: 'system', content: RATIONALE_PROMPT },
          { role: 'user', content:
              `Write 2-3 sentences explaining why Rs${vdata.price_low}-Rs${vdata.price_high} is the right BCC intake price for this item.${marketContext}\n` +
              `Brand: ${vdata.brand} | Category: ${vdata.category} | ` +
              `Condition: ${vdata.condition_score}/5 — ${vdata.condition_notes} | ` +
              `Signals: ${signals} | Score: ${vdata.pricing_score}/10\n` +
              `Reference Indian resale platforms and the live market data if provided. Be direct and practical.`
          },
        ],
        { max_tokens: 280, temperature: 0.7 },
        apiKey
      );
      rationale = rationale.trim();
    } catch {
      rationale = `This ${vdata.brand} ${vdata.category.toLowerCase()} in ${vdata.condition_score}/5 condition ` +
        `is recommended at Rs${vdata.price_low}–Rs${vdata.price_high}, consistent with BCC's Indian pre-loved market positioning.`;
    }

    _itemCounter++;
    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
    _sessionLog.push({
      item_no: _itemCounter, timestamp: ts,
      brand: vdata.brand, category: vdata.category,
      score: vdata.pricing_score,
      price_low: vdata.price_low, price_high: vdata.price_high,
      final_price: null, action: 'pending',
    });

    return ok({
      item_no: _itemCounter,
      ...Object.fromEntries(REQUIRED.map(k => [k, vdata[k]])),
      market_rationale: rationale,
      market_prices: marketPrices,   // null when brand is unbranded or search failed
    });
  }

  // ── POST /log-decision ─────────────────────────────────────────────────────
  if (path.endsWith('/log-decision') && method === 'POST') {
    let body;
    try { body = await req.json(); } catch { return err(400, 'Invalid JSON body.'); }
    const { item_no, action, final_price } = body;
    if (!Number.isInteger(item_no) || !['accepted','overridden'].includes(action)) {
      return err(400, "item_no (int) and action ('accepted'|'overridden') are required.");
    }
    const entry = _sessionLog.find(e => e.item_no === item_no);
    if (entry) { entry.action = action; entry.final_price = final_price; }
    return ok({ ok: true });
  }

  // ── GET /session-log ───────────────────────────────────────────────────────
  if (path.endsWith('/session-log') && method === 'GET') {
    return ok({ log: _sessionLog.slice(-10) });
  }

  return err(404, `Route not found: ${method} ${path}`);
}

// Tell Netlify which URL paths this function should intercept (v2 routing)
export const config = {
  path: ['/analyse', '/log-decision', '/session-log'],
};
