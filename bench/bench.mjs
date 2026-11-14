// Latency benchmark for "close the eyes" image edits.
// Usage: node bench/bench.mjs [runsPerFace=3] [modelFilter]
// Writes bench/results.json and bench/out/<model>/<face>-<run>.jpg
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MODELS, DEFAULT_PROMPT, editImage } from '../providers.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const runs = Number(process.argv[2] || 3);
const filter = process.argv[3] ? new RegExp(process.argv[3]) : null;
const faces = fs.readdirSync(path.join(here, 'faces')).filter((f) => /\.(jpe?g|png)$/i.test(f)).sort();
const resultsFile = path.join(here, 'results.json');
const prev = fs.existsSync(resultsFile) ? JSON.parse(fs.readFileSync(resultsFile, 'utf8')) : { models: {} };
const results = { date: new Date().toISOString(), prompt: DEFAULT_PROMPT, faces, models: { ...prev.models } };
const save = () => fs.writeFileSync(resultsFile, JSON.stringify(results, null, 2));

const ids = Object.keys(MODELS).filter((id) => !filter || filter.test(id));
const safe = (id) => id.replace(/[^a-z0-9.\-]+/gi, '_');

async function benchModel(id) {
  const m = MODELS[id];
  const entry = { label: m.label, vendor: m.vendor, provider: m.provider, trials: [], status: 'ok' };
  results.models[id] = entry;
  const keyOk = m.provider === 'openai' ? process.env.OPENAI_API_KEY : process.env.GEMINI_API_KEY;
  if (!keyOk) { entry.status = 'no-key'; save(); return; }
  fs.mkdirSync(path.join(here, 'out', safe(id)), { recursive: true });
  for (let r = 0; r < runs; r++) {
    for (const face of faces) {
      const jpeg = fs.readFileSync(path.join(here, 'faces', face));
      const trial = { face, run: r, t: new Date().toISOString() };
      try {
        const out = await editImage(id, jpeg);
        const ext = out.mime.includes('png') ? 'png' : 'jpg';
        const rel = `out/${safe(id)}/${path.parse(face).name}-${r}.${ext}`;
        fs.writeFileSync(path.join(here, rel), out.data);
        Object.assign(trial, { ms: out.ms, file: rel, bytes: out.data.length });
        console.log(`${id.padEnd(30)} ${face} #${r}  ${out.ms} ms`);
      } catch (e) {
        trial.error = String(e.message || e).slice(0, 400);
        console.log(`${id.padEnd(30)} ${face} #${r}  ERROR ${trial.error.slice(0, 120)}`);
        // Quota / permission problems won't fix themselves: stop this model early.
        if (/\b(429|403|401)\b.*(quota|billing|permission|organization|verify|limit: 0|Free Tier)/i.test(trial.error)) {
          entry.trials.push(trial);
          entry.status = 'blocked';
          entry.reason = trial.error;
          save();
          return;
        }
      }
      entry.trials.push(trial);
      save();
    }
  }
}

// Different models run concurrently; trials within a model run one at a time.
await Promise.all(ids.map(benchModel));

for (const [id, e] of Object.entries(results.models)) {
  const ms = e.trials.filter((t) => t.ms).map((t) => t.ms).sort((a, b) => a - b);
  const q = (p) => ms[Math.min(ms.length - 1, Math.floor(p * ms.length))];
  e.stats = ms.length ? { n: ms.length, min: ms[0], p50: q(0.5), p90: q(0.9), max: ms.at(-1), errors: e.trials.length - ms.length } : null;
  console.log(id.padEnd(30), e.status, e.stats ? `p50 ${e.stats.p50}  p90 ${e.stats.p90}  min ${e.stats.min}` : '');
}
save();
