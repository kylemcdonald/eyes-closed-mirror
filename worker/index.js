// Cloudflare Worker: serves the static site (dist/) and proxies /api/edit to OpenAI.
// The OpenAI key is a Worker secret (`wrangler secret put OPENAI_API_KEY`); it never reaches the browser.
// Because this endpoint spends money, it is deliberately narrow:
//   - fixed server-side prompt (clients can't use it as a general image editor)
//   - allowlisted models only
//   - JPEG bodies under 3 MB only
//   - per-IP rate limit (Workers rate limiting binding)
//   - global daily cap (Durable Object counter, DAILY_LIMIT)
import { editImage, availableModels, DEFAULT_PROMPT } from '../lib/providers.js';

const ALLOWED_MODELS = ['gpt-image-2.6-flare@low-816', 'gpt-image-2.6-flare@low'];
const MAX_BYTES = 3 * 1024 * 1024;

const json = (body, status = 200, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...headers } });

export default {
  async fetch(req, env) {
    const url = new URL(req.url);

    if (url.pathname === '/api/models') {
      return json({ models: availableModels(env, ALLOWED_MODELS), prompt: DEFAULT_PROMPT });
    }

    if (url.pathname === '/api/edit') {
      if (req.method !== 'POST') return json({ error: 'POST only' }, 405);
      if (!env.OPENAI_API_KEY) return json({ error: 'server not configured' }, 503);

      const ip = req.headers.get('cf-connecting-ip') || 'unknown';
      if (env.EDIT_LIMITER) {
        const { success } = await env.EDIT_LIMITER.limit({ key: ip });
        if (!success) return json({ error: 'too many requests, try again in a minute' }, 429, { 'Retry-After': '60' });
      }

      const len = Number(req.headers.get('content-length') || 0);
      if (len > MAX_BYTES) return json({ error: 'image too large' }, 413);
      const body = new Uint8Array(await req.arrayBuffer());
      if (body.length > MAX_BYTES) return json({ error: 'image too large' }, 413);
      if (body.length < 1000 || body[0] !== 0xff || body[1] !== 0xd8) return json({ error: 'expected a JPEG' }, 415);

      // global daily budget
      const counter = env.USAGE.get(env.USAGE.idFromName('global'));
      const { allowed, count, limit } = await (await counter.fetch('https://usage/take', {
        method: 'POST', body: JSON.stringify({ limit: Number(env.DAILY_LIMIT || 300) }),
      })).json();
      if (!allowed) return json({ error: `daily limit reached (${limit}); back tomorrow` }, 429);

      const q = url.searchParams.get('model');
      const model = ALLOWED_MODELS.includes(q) ? q : ALLOWED_MODELS[0];
      try {
        const out = await editImage(model, body, DEFAULT_PROMPT, env, { timeoutMs: 60000 });
        return new Response(out.data, {
          headers: {
            'Content-Type': out.mime, 'Cache-Control': 'no-store',
            'X-Api-Ms': String(out.ms), 'X-Usage-Today': String(count),
          },
        });
      } catch (e) {
        console.error('edit failed', model, e.message);
        // failed edits don't spend the daily budget
        await counter.fetch('https://usage/refund', { method: 'POST', body: '{}' }).catch(() => {});
        // upstream key/billing problems: tell clients to back off for a while (no details leaked)
        if (/openai (401|403|429)|api key|quota|billing/i.test(e.message)) {
          return json({ error: 'service temporarily unavailable' }, 503, { 'Retry-After': '300' });
        }
        return json({ error: 'image edit failed' }, 502);
      }
    }

    return env.ASSETS.fetch(req);
  },
};

// One instance (idFromName('global')) counts edits per UTC day.
export class UsageCounter {
  constructor(state) { this.state = state; }
  async fetch(req) {
    const { limit } = await req.json();
    const day = new Date().toISOString().slice(0, 10);
    const rec = (await this.state.storage.get('usage')) || { day, count: 0 };
    if (rec.day !== day) { rec.day = day; rec.count = 0; }
    if (new URL(req.url).pathname === '/refund') {
      rec.count = Math.max(0, rec.count - 1);
      await this.state.storage.put('usage', rec);
      return Response.json({ count: rec.count });
    }
    if (rec.count >= limit) return Response.json({ allowed: false, count: rec.count, limit });
    rec.count++;
    await this.state.storage.put('usage', rec);
    return Response.json({ allowed: true, count: rec.count, limit });
  }
}
