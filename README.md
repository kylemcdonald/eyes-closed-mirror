# Eyes Closed Mirror

A browser mirror that shows you yourself with your eyes closed. When a new face arrives, one frame is sent to a fast
image-edit model ("close the eyes"). The closed eyelids are then grafted back onto the live video at 60 fps with a
WebGL2 port of the FaceSubstitution clone shader, driven by MediaPipe Face Landmarker.

```sh
node server.mjs              # http://localhost:8417/  explainer + benchmark results
                             # http://localhost:8417/mirror.html  the mirror
                             # http://localhost:8417/mirror.html?demo=synthetic-1  no camera needed
node server.mjs --https      # self-signed https so phones/other machines can use their camera
node bench/make-faces.mjs    # (re)generate synthetic test portraits
node bench/bench.mjs 3 [re]  # latency benchmark, optional model-id regex
```

Keys are read from `.env` or `../.env`: `OPENAI_API_KEY`, `GEMINI_API_KEY`. The browser never sees them.

- `providers.mjs`: model table and the OpenAI / Gemini edit calls
- `public/mirror.js`: capture state machine, mesh warp, masked blur, clone composite
- `public/face.js`: MediaPipe loader and the "stable" landmark sets (eyelids excluded)
- `bench/results.json`, `bench/out/`: benchmark data shown on the explainer

Mirror keys: space = capture now, r = reset, d = cycle debug view, h = hide UI, f = fullscreen.
