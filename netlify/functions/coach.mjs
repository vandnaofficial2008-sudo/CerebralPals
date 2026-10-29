// Netlify Function: sends game scores to an AI model and returns coaching JSON.
// Uses NVIDIA Nemotron models on Nebius Token Factory when NEBIUS_API_KEY is set, otherwise NVIDIA's own API (NVIDIA_API_KEY).
// Optional: NEBIUS_MODEL / NVIDIA_MODEL to choose a model.
const PROVIDERS = {
  nebius: {
    name: 'Nebius',
    key: () => process.env.NEBIUS_API_KEY,
    base: 'https://api.tokenfactory.nebius.com/v1',
    model: () => process.env.NEBIUS_MODEL,
    // Nemotron models hosted on Nebius Token Factory (ids can differ in capitalisation, so the live model list is also checked)
    known: ['nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B', 'nvidia/nemotron-3-super-120b-a12b', 'nvidia/Nemotron-Nano-V2-12b'],
    match: /nemotron/i,
    extra: { chat_template_kwargs: { enable_thinking: false } }
  },
  nvidia: {
    name: 'NVIDIA',
    key: () => process.env.NVIDIA_API_KEY,
    base: 'https://integrate.api.nvidia.com/v1',
    model: () => process.env.NVIDIA_MODEL,
    known: ['meta/llama-3.1-8b-instruct', 'deepseek-ai/deepseek-v3.1', 'moonshotai/kimi-k2.6', 'mistralai/mistral-7b-instruct-v0.3'],
    match: /instruct/i,
    extra: { chat_template_kwargs: { enable_thinking: false } }
  }
};
const pickProvider = () => (PROVIDERS.nebius.key() ? PROVIDERS.nebius : PROVIDERS.nvidia.key() ? PROVIDERS.nvidia : null);

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
  const prov = pickProvider();
  if (!prov) return json({ error: 'missing_key' }, 500);
  if (limited(context.ip || 'unknown')) return json({ error: 'rate_limited' }, 429);

  let body;
  try { body = await req.json(); } catch { return json({ error: 'bad_request' }, 400); }
  if (!validate(body)) return json({ error: 'bad_request' }, 400);

  try {
    const key = prov.key();
    const prompt = buildPrompt(body);
    const started = Date.now();
    const models = await candidateModels(prov, key);
    let lastStatus = 0;

    // Walk through candidate models. A retired or missing model (404/410) just moves on to the next one.
    for (let i = 0; i < models.length && i < 8; i++) {
      if (Date.now() - started > 19000) break;
      const model = models[i];
      try {
        // Reasoning models can spend all their tokens thinking, so ask for little thinking and allow plenty of tokens.
        const call = extra => fetch(prov.base + '/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + key },
          body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: 3000, temperature: 0.6, stream: false, ...extra }),
          signal: AbortSignal.timeout(Math.max(3000, Math.min(i === 0 ? 24000 : 9000, 27000 - (Date.now() - started))))
        });
        let r = await call(prov.extra);
        if (r.status === 400) r = await call({}); // model does not know that setting: retry plainly
        if (r.ok) {
          const data = await r.json();
          const text = data.choices?.[0]?.message?.content || '';
          const clean = text.replace(/```json|```/g, '').trim();
          if (clean.indexOf('{') < 0) throw new Error('Empty answer (finish_reason: ' + data.choices?.[0]?.finish_reason + ')');
          console.log(prov.name + ' answered with ' + model);
          return json(JSON.parse(clean.slice(clean.indexOf('{'), clean.lastIndexOf('}') + 1)));
        }
        lastStatus = r.status;
        console.error(prov.name + ' API', r.status, model, (await r.text()).slice(0, 200));
        if (r.status === 401 || r.status === 403) break; // key problem: other models will fail too
      } catch (e) {
        lastStatus = 0;
        console.error(prov.name + ' request failed, timed out or gave unreadable JSON', model, e.name, e.message);
      }
    }
    return json({ error: lastStatus === 429 ? 'rate_limited' : 'upstream' }, lastStatus === 429 ? 429 : 502);
  } catch (e) {
    console.error(e);
    return json({ error: 'upstream' }, 502);
  }
};

// Your chosen model first (if set), then well-known ones, then whatever instruct models the provider lists.
const modelCache = {};
async function candidateModels(prov, key) {
  const first = [prov.model()].filter(Boolean);
  if (!modelCache[prov.name]) {
    modelCache[prov.name] = [];
    try {
      const r = await fetch(prov.base + '/models', { headers: { Authorization: 'Bearer ' + key }, signal: AbortSignal.timeout(4000) });
      if (r.ok) {
        const ids = ((await r.json()).data || []).map(m => m.id);
        const ok = ids.filter(id => prov.match.test(id) && !/embed|rerank|guard|vision|-vl|omni|video|audio|reward|parse|safety|clip|code|math|translate|nemoretriever/i.test(id));
        const size = id => { const m = id.match(/(\d+(?:\.\d+)?)b/i); return m ? parseFloat(m[1]) : 50; };
        modelCache[prov.name] = ok.sort((x, y) => Math.abs(size(x) - 30) - Math.abs(size(y) - 30)).slice(0, 6);
      }
    } catch (e) { console.error('Could not list models', e.name); }
  }
  return [...new Set([...first, ...prov.known, ...modelCache[prov.name]])];
}

export const config = { path: '/api/coach' };
