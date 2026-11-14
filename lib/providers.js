// Image-edit providers, runtime agnostic (Node and Cloudflare Workers). Keys are passed in via `env`,
// never read from globals, so the same code runs in server.mjs, bench.mjs and worker/index.js.

function toB64(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}
function fromB64(b64) {
  const s = atob(b64), u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return u8;
}

export const DEFAULT_PROMPT =
  'Edit this photo so the person has their eyes gently closed, as if peacefully resting. ' +
  'Change only the eyelids. Keep everything else exactly identical: same face, identity, skin, ' +
  'expression, head pose, framing, lighting, colors, background and image size. Photorealistic.';

// id -> config. `opts` are provider specific knobs.
export const MODELS = {
  'gemini-3.1-flash-lite-image': { provider: 'gemini', label: 'Nano Banana 2 Lite', vendor: 'Google' },
  'gemini-3.1-flash-image': { provider: 'gemini', label: 'Nano Banana 2', vendor: 'Google' },
  'gemini-nano-banana-2.1': { provider: 'gemini', label: 'Nano Banana 2.1', vendor: 'Google', opts: { thinking: 'minimal' } },
  'gemini-3-pro-image': { provider: 'gemini', label: 'Nano Banana Pro', vendor: 'Google', opts: { thinking: null } },
  'gemini-2.5-flash-image': { provider: 'gemini', label: 'Nano Banana (2.5)', vendor: 'Google', opts: { thinking: null } },
  'gpt-image-2.6-flare@low': { provider: 'openai', model: 'gpt-image-2.6-flare', label: 'GPT Image 2.6 Flare (low)', vendor: 'OpenAI', opts: { quality: 'low', complexity: 'low' } },
  'gpt-image-2.6-flare@medium': { provider: 'openai', model: 'gpt-image-2.6-flare', label: 'GPT Image 2.6 Flare (medium)', vendor: 'OpenAI', opts: { quality: 'medium', complexity: 'low' } },
  'gpt-image-2.6-flare@low-816': { provider: 'openai', model: 'gpt-image-2.6-flare', size: '816x816', label: 'GPT Image 2.6 Flare (low, 816²)', vendor: 'OpenAI', opts: { quality: 'low', complexity: 'low' } },
  'gpt-image-2.6-flare@low-defaultcx': { provider: 'openai', model: 'gpt-image-2.6-flare', label: 'GPT Image 2.6 Flare (low, default complexity)', vendor: 'OpenAI', opts: { quality: 'low' } },
  'gpt-image-2.5-flare@low': { provider: 'openai', model: 'gpt-image-2.5-flare', label: 'GPT Image 2.5 Flare (low)', vendor: 'OpenAI', opts: { quality: 'low', complexity: 'low' } },
  'gpt-image-2.6-sunburst@low': { provider: 'openai', model: 'gpt-image-2.6-sunburst', label: 'GPT Image 2.6 Sunburst (low)', vendor: 'OpenAI', opts: { quality: 'low', complexity: 'low' } },
};

export function availableModels(env, allow = null) {
  return Object.entries(MODELS)
    .filter(([id, m]) => (!allow || allow.includes(id)) && (m.provider === 'openai' ? env.OPENAI_API_KEY : env.GEMINI_API_KEY))
    .map(([id, m]) => ({ id, label: m.label, vendor: m.vendor }));
}

async function readError(res) {
  const t = await res.text();
  try { const j = JSON.parse(t); return j.error?.message || t; } catch { return t.slice(0, 500); }
}

// ---------- OpenAI: POST /v1/images/edits (multipart) ----------
async function openaiEdit(cfg, jpeg, prompt, signal, env) {
  const send = async (extra) => {
    const fd = new FormData();
    fd.append('model', cfg.model);
    fd.append('prompt', prompt);
    fd.append('image[]', new Blob([jpeg], { type: 'image/jpeg' }), 'face.jpg');
    fd.append('size', cfg.size || '1024x1024');
    fd.append('output_format', 'jpeg');
    fd.append('output_compression', '90');
    fd.append('n', '1');
    for (const [k, v] of Object.entries(extra)) if (v != null) fd.append(k, v);
    return fetch('https://api.openai.com/v1/images/edits', {
      method: 'POST', headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` }, body: fd, signal,
    });
  };
  let res = await send(cfg.opts || {});
  // If an optional knob is rejected (e.g. `complexity`), retry without it once.
  if (res.status === 400) {
    const msg = await readError(res);
    const bad = Object.keys(cfg.opts || {}).find((k) => msg.toLowerCase().includes(k));
    if (!bad) throw new Error(`openai 400: ${msg}`);
    const { [bad]: _drop, ...rest } = cfg.opts;
    cfg.opts = rest; // remember for subsequent calls
    console.warn(`[openai] dropped unsupported param "${bad}" for ${cfg.model}`);
    res = await send(rest);
  }
  if (!res.ok) throw new Error(`openai ${res.status}: ${await readError(res)}`);
  const j = await res.json();
  const b64 = j.data?.[0]?.b64_json;
  if (!b64) throw new Error('openai: no image in response');
  return { data: fromB64(b64), mime: 'image/jpeg', usage: j.usage };
}

// ---------- Gemini: Interactions API, falling back to generateContent ----------
async function geminiEdit(id, cfg, jpeg, prompt, signal, env) {
  const key = env.GEMINI_API_KEY;
  const b64 = toB64(jpeg);
  const thinking = cfg.opts && 'thinking' in cfg.opts ? cfg.opts.thinking : 'minimal';

  if (cfg.api !== 'generateContent') {
    const body = {
      model: id,
      input: [
        { type: 'text', text: prompt },
        { type: 'image', mime_type: 'image/jpeg', data: b64 },
      ],
      response_format: { type: 'image', aspect_ratio: '1:1', image_size: '1K' },
    };
    if (thinking) body.generation_config = { thinking_level: thinking };
    const res = await fetch('https://generativelanguage.googleapis.com/v1beta/interactions', {
      method: 'POST',
      headers: { 'x-goog-api-key': key, 'Content-Type': 'application/json', 'Api-Revision': '2026-05-20' },
      body: JSON.stringify(body), signal,
    });
    if (res.ok) {
      const j = await res.json();
      const img = j.output_image || j.outputs?.find?.((o) => o.type === 'image') || findImage(j);
      if (img?.data) return { data: fromB64(img.data), mime: img.mime_type || 'image/png', usage: j.usage };
      throw new Error('gemini interactions: no image in response: ' + JSON.stringify(j).slice(0, 300));
    }
    const msg = await readError(res);
    if (res.status !== 404 && res.status !== 400) throw new Error(`gemini ${res.status}: ${msg}`);
    console.warn(`[gemini] interactions failed for ${id} (${res.status}: ${msg.slice(0, 120)}); using generateContent`);
    cfg.api = 'generateContent';
  }

  const body = {
    contents: [{ role: 'user', parts: [{ text: prompt }, { inline_data: { mime_type: 'image/jpeg', data: b64 } }] }],
    generationConfig: { responseModalities: ['IMAGE'], imageConfig: { aspectRatio: '1:1' } },
  };
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${id}:generateContent`, {
    method: 'POST', headers: { 'x-goog-api-key': key, 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
  });
  if (!res.ok) throw new Error(`gemini ${res.status}: ${await readError(res)}`);
  const j = await res.json();
  const img = findImage(j);
  if (!img) throw new Error('gemini: no image in response: ' + JSON.stringify(j).slice(0, 300));
  return { data: fromB64(img.data), mime: img.mime_type || img.mimeType || 'image/png', usage: j.usageMetadata };
}

// Depth-first search for the first {data, mime_type|mimeType} image blob.
function findImage(o) {
  if (!o || typeof o !== 'object') return null;
  const blob = o.inlineData || o.inline_data;
  if (blob?.data) return blob;
  if (typeof o.data === 'string' && o.data.length > 1000 && /image\//.test(o.mime_type || o.mimeType || '')) return o;
  for (const v of Object.values(o)) { const r = findImage(v); if (r) return r; }
  return null;
}

// env holds OPENAI_API_KEY / GEMINI_API_KEY: process.env in Node, the Worker's env (secrets) on Cloudflare.
export async function editImage(id, jpeg, prompt, env, { timeoutMs = 120000 } = {}) {
  const cfg = MODELS[id];
  if (!cfg) throw new Error(`unknown model ${id}`);
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  const t0 = performance.now();
  try {
    const out = cfg.provider === 'openai'
      ? await openaiEdit(cfg, jpeg, prompt, ac.signal, env)
      : await geminiEdit(id, cfg, jpeg, prompt, ac.signal, env);
    return { ...out, ms: Math.round(performance.now() - t0) };
  } finally {
    clearTimeout(timer);
  }
}
