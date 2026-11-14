// Generate synthetic eyes-open test portraits (so no real person's photo is sent to the APIs).
// Usage: node bench/make-faces.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import '../providers.mjs'; // loads .env

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'faces');
const people = [
  'a woman in her 30s with dark curly hair, warm indoor light',
  'an older man with a grey beard and glasses, overcast daylight',
  'a young man with short black hair, under fluorescent gallery lighting',
  'a teenage girl with freckles and red hair, soft window light',
];

async function gen(desc) {
  const prompt = `Candid webcam photo, head and shoulders, of ${desc}, looking straight into the camera ` +
    `with eyes wide open, neutral expression, plain background, slightly soft focus like a laptop webcam. Square crop, face centered.`;
  const res = await fetch('https://api.openai.com/v1/images/generations', {
    method: 'POST',
    headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'gpt-image-2.6-flare', prompt, size: '1024x1024', quality: 'medium', output_format: 'jpeg' }),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const j = await res.json();
  return Buffer.from(j.data[0].b64_json, 'base64');
}

await Promise.all(people.map(async (desc, i) => {
  const file = path.join(dir, `synthetic-${i + 1}.jpg`);
  if (fs.existsSync(file)) return;
  fs.writeFileSync(file, await gen(desc));
  console.log('wrote', file);
}));
