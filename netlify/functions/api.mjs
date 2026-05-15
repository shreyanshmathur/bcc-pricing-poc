/**
 * BCC AI Pricing Engine — Netlify Function (v2, JavaScript)
 * Zero npm dependencies. Calls Groq HTTP API directly.
 *
 * Pipeline (per analyse request):
 *  1. Vision  — llama-4-scout  → brand, category, condition, initial price (up to 3 images)
 *  2. Search  — compound-beta  → live retail MRP + Indian resale prices (web search)
 *  3. Rationale — llama-3.3-70b → human-readable justification grounded in market data
 *
 * Key rotation: alternates between GROQ_API_KEY and GROQ_API_KEY_2 to avoid rate limits.
 */

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

let _sessionLog  = [];
let _itemCounter = 0;
let _keyIndex    = 0;          // round-robin across API keys

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

If multiple photos are provided they are all of the same item — use them together for a more accurate assessment.

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

The photo may show the item laid flat, on a hanger, or being worn by a person — all are valid. Assess the clothing visible in the image regardless of how it is presented. When worn by a person, focus on the garment(s), not the person.

Only return the error JSON if the image contains NO clothing at all (e.g. a landscape, food, animal, blank wall):
{"error": "This image does not appear to show a clothing item."}`;

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

function ok(data)        { return new Response(JSON.stringify(data), { status: 200, headers: corsHeaders() }); }
function err(status, msg){ return new Response(JSON.stringify({ detail: msg }), { status, headers: corsHeaders() }); }

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

/** Live market price search via compound-beta web search. Gracefully returns null on failure. */
async function fetchMarketPrices(brand, category, apiKey) {
  if (!brand || brand.toLowerCase() === 'unbranded') return null;
  try {
    const result = await groqChat(
      'compound-beta',
      [
        {
          role: 'system',
          content: 'You are a pricing researcher for Indian fashion resale. Use web search. Answer in 2-3 sentences with specific INR amounts. India market only.',
        },
        {
          role: 'user',
          content:
            `Search and find: (1) Current retail/MRP price of ${brand} ${category} sold new in India (INR). ` +
            `(2) Current second-hand resale prices for ${brand} ${category} on OLX India, Carousell India, or Indian Instagram thrift accounts (INR). ` +
            `Give specific price ranges. Be brief.`,
        },
      ],
      { max_tokens: 350, temperature: 0.1 },
      apiKey
    );
    return result ? result.trim() : null;
  } catch (_) {
    return null;
  }
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

    // Accept up to 3 files (all sent with field name "file")
    const rawFiles = formData.getAll('file').filter(f => f && typeof f !== 'string');
    if (!rawFiles.length) return err(400, 'No file found in request. Upload a JPEG or PNG image.');

    const files = rawFiles.slice(0, 3);

    // Per-file size check (10 MB each) + total sanity check
    let totalBytes = 0;
    for (const f of files) {
      if (f.size > 10 * 1024 * 1024)
        return err(413, `One file is ${(f.size / 1024 / 1024).toFixed(1)} MB — max 10 MB per photo.`);
      totalBytes += f.size;
    }
    if (totalBytes > 25 * 1024 * 1024)
      return err(413, `Total upload is ${(totalBytes / 1024 / 1024).toFixed(1)} MB — max 25 MB combined.`);

    // Build image_url content blocks for all uploaded photos
    const imageBlocks = await Promise.all(files.map(async file => {
      const buf  = await file.arrayBuffer();
      const b64  = Buffer.from(buf).toString('base64');
      const mime = file.type || 'image/jpeg';
      return { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } };
    }));

    // ── Step 1: Vision ───────────────────────────────────────────────────────
    let visionRaw;
    try {
      const photoNote = files.length > 1
        ? `Analyse these ${files.length} photos of the same clothing item and return your pricing assessment as a JSON object.`
        : 'Analyse this clothing item and return your pricing assessment as a JSON object.';

      visionRaw = await groqChat(
        'meta-llama/llama-4-scout-17b-16e-instruct',
        [
          { role: 'system', content: VISION_PROMPT },
          { role: 'user', content: [...imageBlocks, { type: 'text', text: photoNote }] },
        ],
        { response_format: { type: 'json_object' }, max_tokens: 1024, temperature: 0.2 },
        apiKey
      );
    } catch (e) { return err(502, `Vision model request failed: ${e.message}`); }

    let vdata;
    try { vdata = JSON.parse(visionRaw); }
    catch { return err(502, `Could not parse model response: ${String(visionRaw).slice(0, 200)}`); }

    if (vdata.error) return err(422, vdata.error);

    const REQUIRED = ['brand','category','condition_score','condition_notes',
                      'rarity_signals','pricing_score','price_low','price_high','confidence'];
    const missing = REQUIRED.filter(k => !(k in vdata));
    if (missing.length) return err(502, `Model response missing fields: ${missing.join(', ')}`);

    vdata.condition_score = Math.max(1, Math.min(5, parseInt(vdata.condition_score) || 3));
    vdata.pricing_score   = Math.max(1, Math.min(10, parseInt(vdata.pricing_score) || 5));
    vdata.price_low       = parseInt(vdata.price_low)  || 299;
    vdata.price_high      = parseInt(vdata.price_high) || 499;
    if (!Array.isArray(vdata.rarity_signals))
      vdata.rarity_signals = vdata.rarity_signals ? [String(vdata.rarity_signals)] : [];
    if (vdata.price_low > vdata.price_high)
      [vdata.price_low, vdata.price_high] = [vdata.price_high, vdata.price_low];

    // ── Step 2: Live web market prices ───────────────────────────────────────
    const marketPrices = await fetchMarketPrices(vdata.brand, vdata.category, pickApiKey());

    // ── Step 3: Rationale grounded in market data ────────────────────────────
    let rationale = '';
    try {
      const signals       = vdata.rarity_signals.join(', ') || 'none noted';
      const marketContext = marketPrices ? `\nLive market research:\n${marketPrices}\n` : '';
      rationale = await groqChat(
        'llama-3.3-70b-versatile',
        [
          { role: 'system', content: RATIONALE_PROMPT },
          { role: 'user', content:
              `Write 2-3 sentences explaining why Rs${vdata.price_low}-Rs${vdata.price_high} is the right BCC intake price.${marketContext}\n` +
              `Brand: ${vdata.brand} | Category: ${vdata.category} | Condition: ${vdata.condition_score}/5 — ${vdata.condition_notes} | ` +
              `Signals: ${signals} | Score: ${vdata.pricing_score}/10\n` +
              `Reference Indian resale platforms and the live market data if provided. Be direct and practical.`
          },
        ],
        { max_tokens: 280, temperature: 0.7 },
        pickApiKey()
      );
      rationale = rationale.trim();
    } catch {
      rationale = `This ${vdata.brand} ${vdata.category.toLowerCase()} in ${vdata.condition_score}/5 condition ` +
        `is recommended at Rs${vdata.price_low}–Rs${vdata.price_high}, consistent with BCC's Indian pre-loved market positioning.`;
    }

    _itemCounter++;
    const now = new Date();
    const ts  = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
    _sessionLog.push({
      item_no: _itemCounter, timestamp: ts,
      brand: vdata.brand, category: vdata.category,
      score: vdata.pricing_score, photo_count: files.length,
      price_low: vdata.price_low, price_high: vdata.price_high,
      final_price: null, action: 'pending',
    });

    return ok({
      item_no: _itemCounter,
      photo_count: files.length,
      ...Object.fromEntries(REQUIRED.map(k => [k, vdata[k]])),
      market_rationale: rationale,
      market_prices: marketPrices,
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
    return ok({ log: _sessionLog.slice(-10) });
  }

  return err(404, `Route not found: ${method} ${path}`);
}

export const config = {
  path: ['/analyse', '/log-decision', '/session-log'],
};
