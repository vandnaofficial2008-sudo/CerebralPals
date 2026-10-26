// Netlify Function: proxies coaching requests to the Google Gemini API (Google AI Studio key).
// Needs the environment variable GEMINI_API_KEY (set in the Netlify dashboard).
const MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

// Best-effort rate limit (per warm function instance): 10 requests / 10 minutes per IP
const hits = new Map();
function limited(ip) {
  const now = Date.now(), win = 10 * 60 * 1000;
  const list = (hits.get(ip) || []).filter(t => now - t < win);
  list.push(now); hits.set(ip, list);
  return list.length > 10;
}

function buildPrompt(b) {
  const scores = b.rows.map(r => `- ${r.skill} (${r.game}): ${r.score}/100. ${r.detail}`).join('\n');
  return 'You are a warm, encouraging brain-training coach talking to a college student who just played short brain games. '
    + `Scores (0 to 100 per skill):\n${scores}\nTotal Brain Points: ${b.points} of ${b.maxPoints}.\n\n`
    + 'Reply with only a JSON object: {"headline": string (max 8 words), "summary": string (2 to 3 sentences, plain language), '
    + '"weak_points": [{"skill": string, "why": string (1 sentence), "practice": string (1 to 2 sentences, concrete)}] (at most 2, the lowest scores), '
    + '"plan": [string, string, string] (three habits of 5 minutes or less)}. '
    + 'Be specific to these numbers, never diagnose anything, and remind them this is a fun snapshot, not a clinical test, only if it fits naturally.';
}

function validate(b) {
  if (!b || !Array.isArray(b.rows) || b.rows.length < 1 || b.rows.length > 8) return false;
  b.points = Number(b.points) || 0; b.maxPoints = Number(b.maxPoints) || 0;
  b.rows = b.rows.map(r => ({
    skill: String(r.skill || '').slice(0, 40), game: String(r.game || '').slice(0, 40),
    score: Math.max(0, Math.min(100, Number(r.score) || 0)), detail: String(r.detail || '').slice(0, 120)
  }));
  return true;
}

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json' } });

export default async (req, context) => {
  if (req.method !== 'POST') return json({ error: 'method_not_allowed' }, 405);
  if (!process.env.GEMINI_API_KEY) return json({ error: 'missing_key' }, 500);
  if (limited(context.ip || 'unknown')) return json({ error: 'rate_limited' }, 429);

  let body;
  try { body = await req.json(); } catch { return json({ error: 'bad_request' }, 400); }
  if (!validate(body)) return json({ error: 'bad_request' }, 400);

  try {
    const models = [MODEL, process.env.GEMINI_FALLBACK_MODEL].filter(Boolean);
    const payload = JSON.stringify({
      contents: [{ parts: [{ text: buildPrompt(body) }] }],
      generationConfig: {
        responseMimeType: 'application/json',
        maxOutputTokens: 1000,
        thinkingConfig: { thinkingBudget: 0 }
      }
    });

    // Gemini is sometimes busy (503) or slow. Try up to 3 times, each with a time limit,
    // so the function always answers before Netlify's 30 second cut-off.
    const started = Date.now();
    let lastStatus = 0;
    for (let attempt = 0; attempt < 3; attempt++) {
      const model = models[Math.min(attempt > 0 ? 1 : 0, models.length - 1)];
      if (Date.now() - started > 18000) break;
      try {
        const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-goog-api-key': process.env.GEMINI_API_KEY },
          body: payload,
          signal: AbortSignal.timeout(8000)
        });
        if (r.ok) {
          const data = await r.json();
          const text = (data.candidates?.[0]?.content?.parts || []).map(p => p.text || '').join('');
          const clean = text.replace(/```json|```/g, '').trim();
          return json(JSON.parse(clean.slice(clean.indexOf('{'), clean.lastIndexOf('}') + 1)));
        }
        lastStatus = r.status;
        console.error('Gemini API', r.status, model, (await r.text()).slice(0, 300));
        if (![429, 500, 503, 504].includes(r.status)) break; // not worth retrying
      } catch (e) {
        lastStatus = 0;
        console.error('Gemini request failed or timed out', model, e.name);
      }
      await new Promise(res => setTimeout(res, 1000));
    }
    return json({ error: lastStatus === 429 ? 'rate_limited' : 'upstream' }, lastStatus === 429 ? 429 : 502);
  } catch (e) {
    console.error(e);
    return json({ error: 'upstream' }, 502);
  }
};

export const config = { path: '/api/coach' };
