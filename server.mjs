// Eyes Closed Mirror: static server + image-edit proxy (keeps API keys server-side).
// Usage: node server.mjs [--port 8417] [--https]
//   --https generates a self-signed cert in .certs/ so the camera works from other devices
//   (browsers only allow getUserMedia on https:// or http://localhost).
import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { editImage, availableModels, DEFAULT_PROMPT } from './providers.mjs';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const port = Number(args[args.indexOf('--port') + 1]) || Number(process.env.PORT) || 8417;
const useHttps = args.includes('--https');

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.webp': 'image/webp', '.log': 'text/plain; charset=utf-8',
};

// URL -> file. "/" and pages come from public/, benchmark data from bench/.
function resolve(urlPath) {
  const p = decodeURIComponent(urlPath.split('?')[0]);
  const base = p.startsWith('/bench/') ? root : path.join(root, 'public');
  const file = path.normalize(path.join(base, p === '/' ? 'index.html' : p));
  if (!file.startsWith(base)) return null;
  // clean URLs like Cloudflare's assets: /about -> about.html
  if (!path.extname(file) && fs.existsSync(file + '.html')) return file + '.html';
  return file;
}

async function readBody(req, limit = 15e6) {
  const chunks = []; let n = 0;
  for await (const c of req) { n += c.length; if (n > limit) throw new Error('body too large'); chunks.push(c); }
  return Buffer.concat(chunks);
}

async function handler(req, res) {
  const url = new URL(req.url, 'http://x');
  try {
    if (url.pathname === '/api/models') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ models: availableModels(), prompt: DEFAULT_PROMPT }));
    }
    if (url.pathname === '/api/edit' && req.method === 'POST') {
      const model = url.searchParams.get('model') || 'gpt-image-2.6-flare@low-816';
      const prompt = req.headers['x-prompt'] ? decodeURIComponent(req.headers['x-prompt']) : DEFAULT_PROMPT;
      const jpeg = await readBody(req);
      const t0 = Date.now();
      const out = await editImage(model, jpeg, prompt);
      console.log(`[edit] ${model} in=${(jpeg.length / 1024) | 0}KB api=${out.ms}ms total=${Date.now() - t0}ms`);
      res.writeHead(200, {
        'Content-Type': out.mime, 'X-Api-Ms': String(out.ms), 'Access-Control-Expose-Headers': 'X-Api-Ms', 'Cache-Control': 'no-store',
      });
      return res.end(out.data);
    }
    const file = resolve(url.pathname);
    if (!file || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    fs.createReadStream(file).pipe(res);
  } catch (e) {
    console.error('[error]', url.pathname, e.message);
    if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: e.message }));
  }
}

function selfSigned() {
  const dir = path.join(root, '.certs');
  const key = path.join(dir, 'key.pem'), cert = path.join(dir, 'cert.pem');
  if (!fs.existsSync(cert)) {
    fs.mkdirSync(dir, { recursive: true });
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '825', '-subj', '/CN=eyes-closed-mirror',
      '-keyout', key, '-out', cert], { stdio: 'ignore' });
  }
  return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
}

const server = useHttps ? https.createServer(selfSigned(), handler) : http.createServer(handler);
server.listen(port, '0.0.0.0', () => {
  const s = useHttps ? 'https' : 'http';
  console.log(`Eyes Closed Mirror on ${s}://localhost:${port}/  (models: ${availableModels().map((m) => m.id).join(', ') || 'none - add keys to .env'})`);
});
