// Assemble dist/ for Cloudflare: the static pages plus the benchmark data the explainer reads.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(root, 'dist');
fs.rmSync(dist, { recursive: true, force: true });
fs.cpSync(path.join(root, 'public'), dist, { recursive: true });
for (const p of ['results.json', 'faces', 'out']) {
  fs.cpSync(path.join(root, 'bench', p), path.join(dist, 'bench', p), { recursive: true });
}
console.log('built', dist);
