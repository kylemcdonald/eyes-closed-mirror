# Eyes Closed Mirror

A browser mirror that shows you yourself with your eyes closed. When a new face arrives, one frame is sent to a fast
image-edit model (GPT Image 2.6 Flare, "close the eyes", ~5 s). The closed eyelids are then grafted back onto the live
video at 60 fps: MediaPipe Face Landmarker + pyramidal Lucas-Kanade optical flow for tracking, and a WebGL2 port of the
[FaceSubstitution](https://github.com/arturoc/FaceSubstitution) clone shader for blending.

The explainer page (`/`) has the model latency benchmark, quality metrics and how the tracking/blending works.

## Run locally

```sh
cp .env.example .env         # add OPENAI_API_KEY (GEMINI_API_KEY optional, for benchmarking)
node server.mjs              # http://localhost:8417/          explainer
                             # http://localhost:8417/mirror.html   the mirror
                             # http://localhost:8417/mirror.html?demo=synthetic-1   no camera needed
node server.mjs --https      # self-signed https so phones/other machines can use their camera
node bench/make-faces.mjs    # (re)generate synthetic test portraits
node bench/bench.mjs 3 [re]  # latency benchmark, optional model-id regex
```

Mirror controls are hidden by default: **double-tap** (or press `h`) to show them, `?ui=1` to start with them shown.
Keys: space = capture now, r = reset, d = cycle debug view, f = fullscreen.
Tracking benchmark: `mirror.html?demo=synthetic-3&bench=300&motion=fast` (results in the console).

## Deploy (Cloudflare Workers)

```sh
wrangler secret put OPENAI_API_KEY   # stored encrypted by Cloudflare, never in the repo or the browser
npm run deploy                       # builds dist/ and deploys worker + static assets
```

`worker/index.js` serves the static site and a deliberately narrow `/api/edit`: fixed prompt, allowlisted models,
JPEG-only bodies under 3 MB, per-IP rate limit (6/min) and a global daily cap (`DAILY_LIMIT` in `wrangler.jsonc`,
counted by a Durable Object).

## Layout

- `lib/providers.js`: model table and the OpenAI / Gemini edit calls (runtime agnostic; keys passed in)
- `providers.mjs`, `server.mjs`: local Node server (loads `.env`)
- `worker/index.js`, `wrangler.jsonc`: Cloudflare deployment
- `public/mirror.js`: capture state machine, mesh warp, masked blur, clone composite
- `public/flow.js`: pyramidal Lucas-Kanade + fused optical-flow/detector tracker
- `public/face.js`: MediaPipe loader and the "stable" landmark sets (eyelids excluded)
- `bench/`: latency benchmark, synthetic test faces, results shown on the explainer
