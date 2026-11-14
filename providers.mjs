// Node wrapper around lib/providers.js for server.mjs and bench.mjs.
// Keys come from ./.env or ../.env (first non-empty value wins), or the process env.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as core from './lib/providers.js';

const here = path.dirname(fileURLToPath(import.meta.url));

export function loadEnv() {
  for (const file of [path.join(here, '.env'), path.join(here, '..', '.env')]) {
    if (!fs.existsSync(file)) continue;
    for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
      const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
      if (!m) continue;
      const v = m[2].replace(/\s+#.*$/, '').replace(/^["']|["']$/g, '');
      if (v && !process.env[m[1]]) process.env[m[1]] = v;
    }
  }
}
loadEnv();

export const { MODELS, DEFAULT_PROMPT } = core;
export const availableModels = () => core.availableModels(process.env);
export const editImage = (id, jpeg, prompt = DEFAULT_PROMPT, opts) => core.editImage(id, jpeg, prompt, process.env, opts);
