// Eyes Closed Mirror
// 1. Track the live face with MediaPipe FaceLandmarker.
// 2. When a new face is steady with eyes open, crop it and send it to an image-edit model ("close the eyes").
// 3. Find landmarks in the returned image, triangulate the *stable* eye-region landmarks (not the lids).
// 4. Every frame: warp the closed-eye image onto the live face with that mesh, then blend it with the
//    FaceSubstitution clone trick: out = src + blur(live) - blur(src), with both blurs masked to the mesh.
import { createLandmarker, STABLE, WEIGHTS, EYE_R, EYE_L, eyesClosedScore, stableDrift } from './face.js';
import Delaunator from 'https://cdn.jsdelivr.net/npm/delaunator@5.0.1/+esm';
import { FusedTracker, setWindow } from './flow.js';
// phones: smaller optical-flow windows to keep per-frame tracking cheap
if (matchMedia('(pointer: coarse)').matches) setWindow(6);

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const demo = params.get('demo'); // e.g. ?demo=synthetic-1 uses a test face instead of the camera
const settings = {
  model: params.get('model') || localStorage.getItem('ecm.model') || 'gpt-image-2.6-flare@low-816',
  view: 'final', blur: 20, auto: !params.has('bench'), mirror: true,
  hud: params.get('ui') === '1', // settings + HUD hidden by default; double-tap or press H to show
  // Tracking. 'image' = MediaPipe IMAGE mode every frame (no internal smoothing, so no lag; noisy on its own).
  // 'video' = VIDEO mode (MediaPipe smooths landmarks internally, which lags fast motion).
  // flow = lock points to the image with Lucas-Kanade optical flow, detector only corrects drift.
  detector: params.get('detector') || 'image',
  flow: params.get('flow') !== '0',
  smooth: +(params.get('smooth') ?? 0),
};

// ---------------------------------------------------------------- input source
// Every new camera frame is copied ONCE into `frameSource` (a canvas). Tracking, the face crop and the
// displayed texture all read that same snapshot, so the overlay is always drawn on the exact frame its
// landmarks came from. (Reading the <video> element separately for tracking and for display lets it advance
// to a newer frame in between, which makes the eyes trail the face.) The cost is that the displayed video
// is delayed by the tracking time, which the HUD reports.
let frameSource, frameCtx, video, W, H, demoImg;
let newVideoFrame = false, rvfcHandle = null;
async function setupSource() {
  frameSource = document.createElement('canvas');
  if (demo) {
    demoImg = new Image();
    demoImg.src = `/bench/faces/${demo}.jpg`;
    await demoImg.decode();
    W = 1280; H = 720;
  } else {
    video = $('#video');
    await openCamera(localStorage.getItem('ecm.camera') || null);
  }
  frameSource.width = W; frameSource.height = H;
  frameCtx = frameSource.getContext('2d', { alpha: false, willReadFrequently: true }); // CPU-side, for optical flow
}

// Open a camera by deviceId (null = browser default). 1280x720 is plenty for tracking and keeps a Brio
// from delivering 4K frames we'd only downsample.
async function openCamera(deviceId) {
  video.srcObject?.getTracks().forEach((t) => t.stop());
  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { ...size, deviceId: { exact: deviceId } } : { ...size, facingMode: 'user' }, audio: false,
    });
  } catch (e) {
    if (!deviceId) throw e;
    console.warn('saved camera unavailable, falling back to default', e);
    localStorage.removeItem('ecm.camera');
    stream = await navigator.mediaDevices.getUserMedia({ video: { ...size, facingMode: 'user' }, audio: false });
  }
  video.srcObject = stream;
  await video.play();
  W = video.videoWidth; H = video.videoHeight;
  if ('requestVideoFrameCallback' in video) {
    if (rvfcHandle != null) video.cancelVideoFrameCallback(rvfcHandle);
    const onFrame = () => { newVideoFrame = true; rvfcHandle = video.requestVideoFrameCallback(onFrame); };
    rvfcHandle = video.requestVideoFrameCallback(onFrame);
  }
  return stream.getVideoTracks()[0];
}

async function switchCamera(deviceId) {
  const oldW = W, oldH = H;
  const track = await openCamera(deviceId);
  localStorage.setItem('ecm.camera', track.getSettings().deviceId || deviceId);
  if (W !== oldW || H !== oldH) resizeTargets();
  resetFilters(); newVideoFrame = false; lastResult = null;
  await refreshCameraList();
}

// Device labels are only exposed after camera permission, so this runs after the first getUserMedia.
async function refreshCameraList() {
  const sel = $('#camera');
  if (demo) { sel.innerHTML = '<option>demo image</option>'; sel.disabled = true; return; }
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const current = video.srcObject?.getVideoTracks()[0]?.getSettings().deviceId;
  sel.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || `Camera ${i + 1}`}</option>`).join('');
  if (current) sel.value = current;
}
// Returns true when a new frame was snapshotted into frameSource.
let lastVideoTime = -1;
function grabFrame(t) {
  if (demo) { drawDemo(t); return true; }
  // phones rotate: the stream switches between landscape and portrait sizes
  if (video.videoWidth && (video.videoWidth !== W || video.videoHeight !== H)) {
    W = video.videoWidth; H = video.videoHeight;
    resizeTargets(); resetFilters();
  }
  const fresh = 'requestVideoFrameCallback' in video ? newVideoFrame : video.currentTime !== lastVideoTime;
  if (!fresh) return false;
  newVideoFrame = false; lastVideoTime = video.currentTime;
  frameCtx.drawImage(video, 0, 0, W, H);
  return true;
}
// Demo: the still test face drifts and turns slightly so the tracking + warp are actually exercised.
// motion=fast is a quick head shake: up to ~35 px/frame at 30 fps, +-7 deg roll, +-8% scale.
const fastMotion = params.get('motion') === 'fast';
function demoTransform(t) {
  const τ = t / 1000, TAU = 2 * Math.PI;
  if (fastMotion) return {
    cx: W / 2 + 140 * Math.sin(TAU * 0.7 * τ) + 40 * Math.sin(TAU * 1.9 * τ),
    cy: H / 2 + 50 * Math.sin(TAU * 0.9 * τ + 1),
    rot: 0.12 * Math.sin(TAU * 0.5 * τ),
    s: (H / demoImg.height) * (1 + 0.08 * Math.sin(TAU * 0.4 * τ)),
  };
  if (params.get('motion') === 'still') return { cx: W / 2, cy: H / 2, rot: 0.03, s: H / demoImg.height };
  return {
    cx: W / 2 + Math.sin(t * 0.0007) * 60, cy: H / 2 + Math.sin(t * 0.0011) * 20,
    rot: Math.sin(t * 0.0009) * 0.07, s: (H / demoImg.height) * (1 + Math.sin(t * 0.0005) * 0.06),
  };
}
// image px -> frame px under the demo transform (ground truth for the tracking benchmark)
function demoMap(T, x, y) {
  const lx = (x - demoImg.width / 2) * T.s, ly = (y - demoImg.height / 2) * T.s;
  const c = Math.cos(T.rot), s = Math.sin(T.rot);
  return [T.cx + c * lx - s * ly, T.cy + s * lx + c * ly];
}
let noiseTiles = null;
function drawDemo(t) {
  const c = frameCtx, T = demoTransform(t);
  c.fillStyle = '#ddd'; c.fillRect(0, 0, W, H);
  c.save();
  c.translate(T.cx, T.cy); c.rotate(T.rot);
  c.drawImage(demoImg, -demoImg.width * T.s / 2, -demoImg.height * T.s / 2, demoImg.width * T.s, demoImg.height * T.s);
  c.restore();
  // webcam-like sensor noise, different every frame
  if (!noiseTiles) noiseTiles = [0, 1, 2, 3].map(() => {
    const n = document.createElement('canvas'); n.width = 256; n.height = 256;
    const nc = n.getContext('2d'), d = nc.createImageData(256, 256);
    for (let i = 0; i < d.data.length; i += 4) { const v = Math.random() * 255; d.data[i] = d.data[i + 1] = d.data[i + 2] = v; d.data[i + 3] = 255; }
    nc.putImageData(d, 0, 0); return n;
  });
  c.globalAlpha = 0.07;
  const tile = noiseTiles[Math.floor(Math.random() * 4)], ox = Math.random() * 256, oy = Math.random() * 256;
  c.fillStyle = c.createPattern(tile, 'repeat');
  c.translate(-ox, -oy); c.fillRect(ox, oy, W, H); c.setTransform(1, 0, 0, 1, 0, 0);
  c.globalAlpha = 1;
  if (params.has('flicker')) { // auto-exposure-like global brightness wander
    const f = 0.1 * Math.sin(t / 1000 * 2 * Math.PI * 1.3) + 0.05 * Math.sin(t / 1000 * 2 * Math.PI * 4.1);
    c.fillStyle = f > 0 ? `rgba(255,255,255,${f})` : `rgba(0,0,0,${-f})`; c.fillRect(0, 0, W, H);
  }
}

// ---------------------------------------------------------------- WebGL2
const canvas = $('#out');
const gl = canvas.getContext('webgl2', { premultipliedAlpha: false, preserveDrawingBuffer: !!demo });

const VS_QUAD = `#version 300 es
in vec2 aPos; out vec2 vUv; uniform int uScreen; uniform bool uMirror;
void main() {
  vUv = aPos * 0.5 + 0.5;
  if (uScreen == 1) { vUv.y = 1.0 - vUv.y; if (uMirror) vUv.x = 1.0 - vUv.x; }
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;
// Mesh: positions are live landmarks, texcoords are landmarks in the generated image.
const VS_WARP = `#version 300 es
in vec2 aPos; in vec2 aTex; in float aW; out vec2 vTex; out float vW;
void main() { vTex = aTex; vW = aW; gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0); }`;
// alpha encodes three things in one 8-bit channel:
//   0          outside the mesh
//   0.3        "eye hole": fully swapped, but excluded from the clone blur statistics
//   0.5 .. 1   inside the mesh with blend weight 0..1
const FS_WARP = `#version 300 es
precision highp float; in vec2 vTex; in float vW; uniform sampler2D uSrc; out vec4 o;
void main() { o = vec4(texture(uSrc, vTex).rgb, 0.5 + 0.5 * vW); }`;
// Eye openings (live) are where the two images legitimately differ: open eye vs closed lid. If they were
// included in blur(live) - blur(src), the dark iris / white sclera would tint the swapped lid.
const VS_HOLE = `#version 300 es
in vec2 aPos; void main() { gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0); }`;
const FS_HOLE = `#version 300 es
precision highp float; out vec4 o; void main() { o = vec4(0.0, 0.0, 0.0, 0.3); }`;
// Separable blur restricted to the mesh (normalized convolution, the smooth cousin of MaskBlur.frag).
// pass 0 reads colour from uTex and the mask from uWarp.a; pass 1 reads both from pass 0's output.
const FS_BLUR = `#version 300 es
precision highp float; in vec2 vUv; out vec4 o;
uniform sampler2D uTex, uWarp; uniform vec2 uStep; uniform int uPass; uniform float uSigma;
void main() {
  vec3 sum = vec3(0.0); float wsum = 0.0, norm = 0.0;
  for (int i = -12; i <= 12; i++) {
    float g = exp(-float(i * i) / (2.0 * uSigma * uSigma));
    vec2 uv = vUv + float(i) * uStep;
    vec4 t = texture(uTex, uv);
    float m = uPass == 0 ? step(0.45, texture(uWarp, uv).a) : t.a;
    sum += t.rgb * m * g; wsum += m * g; norm += g;
  }
  o = wsum > 1e-4 ? vec4(sum / wsum, wsum / norm) : vec4(0.0);
}`;
const FS_COMP = `#version 300 es
precision highp float; in vec2 vUv; out vec4 o;
uniform sampler2D uCam, uWarp, uSrcBlur, uDstBlur; uniform float uAmount; uniform int uView;
void main() {
  vec3 cam = texture(uCam, vUv).rgb;
  vec4 w = texture(uWarp, vUv);
  float covered = step(0.25, w.a);
  float weight = w.a < 0.45 ? covered : clamp((w.a - 0.5) * 2.0, 0.0, 1.0);
  weight = smoothstep(0.0, 1.0, weight);
  vec4 sb = texture(uSrcBlur, vUv), db = texture(uDstBlur, vUv);
  vec3 offset = (sb.a > 0.01 && db.a > 0.01) ? db.rgb - sb.rgb : vec3(0.0);
  vec3 cloned = clamp(w.rgb + offset, 0.0, 1.0);
  if (uView == 1) { o = vec4(mix(cam, w.rgb, covered), 1.0); return; }           // raw warp, no blending
  if (uView == 2) {                                                               // mask: pink = blend weight, cyan = eye hole
    vec3 c = mix(cam * 0.3, vec3(1.0, 0.2, 0.4), weight * 0.8 + covered * 0.2);
    o = vec4(w.a > 0.25 && w.a < 0.45 ? vec3(0.2, 0.9, 1.0) : c, 1.0); return; }
  if (uView == 3) { o = vec4(mix(cam, w.rgb, weight * uAmount), 1.0); return; } // feather only, no clone
  o = vec4(mix(cam, cloned, weight * uAmount), 1.0);
}`;

function compile(vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name] = gl.getUniformLocation(p, name); }
  return { p, u, a: (name) => gl.getAttribLocation(p, name) };
}
function texture(w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (w) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  return t;
}
function fbo(w, h) {
  const tex = texture(w, h), fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  return { tex, fb, w, h };
}

let holeVao, holeBuf; const holeData = new Float32Array(2 * 16 * 3 * 2);
let P, quadVao, warpVao, warpPosBuf, warpTexBuf, warpIdxBuf, warpCount = 0;
let camTex, srcTex, warpFbo, blurA, blurB, srcBlur, dstBlur;
function setupGL() {
  P = { warp: compile(VS_WARP, FS_WARP), blur: compile(VS_QUAD, FS_BLUR), comp: compile(VS_QUAD, FS_COMP), hole: compile(VS_HOLE, FS_HOLE) };
  quadVao = gl.createVertexArray(); gl.bindVertexArray(quadVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  for (const prog of [P.blur, P.comp]) { const l = prog.a('aPos'); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0); }

  warpVao = gl.createVertexArray(); gl.bindVertexArray(warpVao);
  const attr = (name, size, data) => {
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    const l = P.warp.a(name); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, 0, 0); return b;
  };
  warpPosBuf = attr('aPos', 2, new Float32Array(STABLE.length * 2));
  warpTexBuf = attr('aTex', 2, new Float32Array(STABLE.length * 2));
  attr('aW', 1, new Float32Array(WEIGHTS));
  warpIdxBuf = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, warpIdxBuf);
  holeVao = gl.createVertexArray(); gl.bindVertexArray(holeVao);
  holeBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, holeBuf); gl.bufferData(gl.ARRAY_BUFFER, holeData, gl.DYNAMIC_DRAW);
  { const l = P.hole.a('aPos'); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0); }
  gl.bindVertexArray(null);

  camTex = texture(); srcTex = texture();
  resizeTargets();
}

// (Re)allocate everything sized to the camera; called again when switching to a camera with another resolution.
function resizeTargets() {
  for (const f of [warpFbo, blurA, blurB, srcBlur, dstBlur]) if (f) { gl.deleteFramebuffer(f.fb); gl.deleteTexture(f.tex); }
  canvas.width = W; canvas.height = H;
  overlay.width = W; overlay.height = H;
  if (frameSource) { frameSource.width = W; frameSource.height = H; }
  warpFbo = fbo(W, H);
  const bw = W >> 1, bh = H >> 1; // low frequencies only: half resolution is plenty
  blurA = fbo(bw, bh); blurB = fbo(bw, bh); srcBlur = fbo(bw, bh); dstBlur = fbo(bw, bh);
}

function bindTex(unit, tex, prog, name) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(prog.u[name], unit); }
function maskedBlur(inputTex, out) {
  const p = P.blur; gl.useProgram(p.p); gl.bindVertexArray(quadVao);
  gl.uniform1f(p.u.uSigma, 6.0);
  const stepPx = settings.blur / 6; // spread the 25 taps so the kernel covers roughly +-2*blur px
  gl.bindFramebuffer(gl.FRAMEBUFFER, blurA.fb); gl.viewport(0, 0, blurA.w, blurA.h);
  bindTex(0, inputTex, p, 'uTex'); bindTex(1, warpFbo.tex, p, 'uWarp');
  gl.uniform2f(p.u.uStep, stepPx / W, 0); gl.uniform1i(p.u.uPass, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, out.fb);
  bindTex(0, blurA.tex, p, 'uTex');
  gl.uniform2f(p.u.uStep, 0, stepPx / H); gl.uniform1i(p.u.uPass, 1);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

const VIEWS = ['final', 'warp', 'mask', 'feather', 'mesh'];
// Fan-triangulate each live eye contour, dilated around its centroid to include lashes and the lid crease.
function updateHoles(lm) {
  let k = 0;
  for (const eye of [EYE_R, EYE_L]) {
    let cx = 0, cy = 0;
    for (const i of eye) { cx += lm[i].x; cy += lm[i].y; }
    cx /= eye.length; cy /= eye.length;
    const w = Math.hypot(lm[eye[0]].x - lm[eye[8]].x, lm[eye[0]].y - lm[eye[8]].y);
    const pt = (i) => {
      const dx = lm[i].x - cx, dy = lm[i].y - cy;
      return [cx + dx * 1.25, cy + dy * 1.25 + (dy < 0 ? -0.18 : 0.06) * w]; // extra room above for the crease
    };
    for (let j = 0; j < eye.length; j++) {
      const a = pt(eye[j]), b = pt(eye[(j + 1) % eye.length]);
      holeData.set([cx, cy, a[0], a[1], b[0], b[1]], k); k += 6;
    }
  }
}

function render(livePts, freshFrame) {
  if (freshFrame) {
    gl.bindTexture(gl.TEXTURE_2D, camTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frameSource);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, warpFbo.fb); gl.viewport(0, 0, W, H);
  gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  const active = livePts && warpCount && state.amount > 0;
  if (active) {
    gl.useProgram(P.warp.p); gl.bindVertexArray(warpVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, warpPosBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, livePts);
    bindTex(0, srcTex, P.warp, 'uSrc');
    gl.drawElements(gl.TRIANGLES, warpCount, gl.UNSIGNED_SHORT, 0);
    if (state.holes) {
      gl.useProgram(P.hole.p); gl.bindVertexArray(holeVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, holeBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, holeData);
      gl.colorMask(false, false, false, true);
      gl.drawArrays(gl.TRIANGLES, 0, holeData.length / 2);
      gl.colorMask(true, true, true, true);
    }
    maskedBlur(warpFbo.tex, srcBlur);
    maskedBlur(camTex, dstBlur);
  }

  const p = P.comp; gl.useProgram(p.p); gl.bindVertexArray(quadVao);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H);
  gl.uniform1i(p.u.uScreen, 1); gl.uniform1i(p.u.uMirror, settings.mirror ? 1 : 0);
  bindTex(0, camTex, p, 'uCam'); bindTex(1, warpFbo.tex, p, 'uWarp');
  bindTex(2, srcBlur.tex, p, 'uSrcBlur'); bindTex(3, dstBlur.tex, p, 'uDstBlur');
  gl.uniform1f(p.u.uAmount, active ? state.amount : 0);
  const v = VIEWS.indexOf(settings.view);
  gl.uniform1i(p.u.uView, v === 4 ? 0 : v);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// ---------------------------------------------------------------- smoothing (One Euro filter, in pixels)
// Any smoothing trades jitter for lag. settings.smooth = 0 passes landmarks through untouched; the default is
// light: ~2.6 Hz cutoff at rest (kills sub-pixel jitter) that opens up quickly with speed, so a moving head
// is followed with almost no lag.
class OneEuro {
  constructor() { Object.assign(this, { x: null, dx: 0, t: 0, dCutoff: 1 }); }
  filter(x, t) {
    const s = settings.smooth;
    if (this.x == null || s <= 0) { this.x = x; this.t = t; this.dx = 0; return x; }
    const minCutoff = 0.4 / s, beta = 0.1 * (1 - s) + 0.004;
    const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
    const a = (c) => 1 / (1 + 1 / (2 * Math.PI * c * dt));
    const dx = (x - this.x) / dt; this.dx += a(this.dCutoff) * (dx - this.dx);
    this.x += a(minCutoff + beta * Math.abs(this.dx)) * (x - this.x);
    return this.x;
  }
}
const filters = STABLE.map(() => [new OneEuro(), new OneEuro()]);
const livePts = new Float32Array(STABLE.length * 2);
const detPx = new Float32Array(STABLE.length * 2);
const tracker = new FusedTracker(STABLE.length);
if (params.has('alpha')) tracker.alpha = +params.get('alpha');
if (params.has('fmode')) tracker.mode = params.get('fmode');
if (params.has('hw')) setWindow(+params.get('hw'));
if (params.has('sigma')) tracker.sigmaFrac = +params.get('sigma');
// Detected landmarks -> optical-flow-fused points -> optional One Euro -> normalized mesh positions.
function trackLive(lm, t) {
  if (lm) STABLE.forEach((idx, i) => { detPx[i * 2] = lm[idx].x * W; detPx[i * 2 + 1] = lm[idx].y * H; });
  const px = tracker.update(frameCtx, W, H, lm ? detPx : null, settings.flow);
  if (!px) return null;
  for (let i = 0; i < STABLE.length; i++) {
    livePts[i * 2] = filters[i][0].filter(px[i * 2], t) / W;
    livePts[i * 2 + 1] = filters[i][1].filter(px[i * 2 + 1], t) / H;
  }
  return livePts;
}
const resetFilters = () => { filters.forEach((f) => { f[0].x = null; f[1].x = null; }); tracker.reset(); };

// ---------------------------------------------------------------- capture + generation state machine
const state = {
  phase: 'loading', // loading | idle | watching | generating | ready | error
  amount: 0, target: 0, steadySince: 0, lastSeen: 0, genId: 0, last: null, message: '', thumb: null,
};
let videoLM, imageLM;

function faceMetrics(lm) {
  const iod = Math.hypot((lm[33].x - lm[263].x) * W, (lm[33].y - lm[263].y) * H);
  const midX = (lm[33].x + lm[263].x) / 2;
  const yaw = ((lm[1].x - midX) * W) / iod; // nose tip offset from eye midpoint
  return { iod, yaw };
}

async function capture(result) {
  const lm = result.faceLandmarks[0];
  const id = ++state.genId;
  // square crop around the face, 1.8x the landmark bbox
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of lm) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  const size = Math.max((x1 - x0) * W, (y1 - y0) * H) * 1.8;
  const cx = ((x0 + x1) / 2) * W, cy = ((y0 + y1) / 2) * H;
  const crop = { x: cx - size / 2, y: cy - size / 2, s: size };
  const N = 1024, c = document.createElement('canvas'); c.width = c.height = N;
  const ctx = c.getContext('2d'); ctx.fillStyle = '#808080'; ctx.fillRect(0, 0, N, N);
  ctx.drawImage(frameSource, crop.x, crop.y, crop.s, crop.s, 0, 0, N, N);
  const capturePts = STABLE.map((i) => [(lm[i].x * W - crop.x) / crop.s, (lm[i].y * H - crop.y) / crop.s]);

  state.phase = 'generating'; state.message = `sending to ${settings.model}…`;
  const t0 = performance.now();
  let blob, apiMs = null;
  try {
    if (demo && params.get('gen') !== 'api') {
      // Demo uses the cached benchmark output for this face (no API call); the crop is irrelevant.
      const r = await fetch(`/bench/out/gpt-image-2.6-flare_low/${demo}-0.jpg`);
      if (!r.ok) throw new Error('no cached output; run the benchmark or use &gen=api');
      blob = await r.blob();
      // capture points must refer to the *original* test image, so detect on it directly
      const r0 = imageLM.detect(demoImg);
      const lm0 = r0.faceLandmarks[0];
      capturePts.splice(0, capturePts.length, ...STABLE.map((i) => [lm0[i].x, lm0[i].y]));
    } else {
      const jpeg = await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.9));
      const r = await fetch(`/api/edit?model=${encodeURIComponent(settings.model)}`, { method: 'POST', body: jpeg });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
      apiMs = Number(r.headers.get('X-Api-Ms')) || null;
      blob = await r.blob();
    }
  } catch (e) {
    if (id !== state.genId) return;
    state.phase = 'error'; state.message = `edit failed: ${e.message}`; console.error(e);
    setTimeout(() => { if (state.phase === 'error') state.phase = 'idle'; }, 4000);
    return;
  }
  if (id !== state.genId) return; // person left while we were waiting
  const bmp = await createImageBitmap(blob);
  const det = imageLM.detect(bmp);
  const genLm = det.faceLandmarks?.[0];
  // Prefer landmarks found in the generated image; fall back to the capture landmarks (same framing).
  const srcPts = genLm ? STABLE.map((i) => [genLm[i].x, genLm[i].y]) : capturePts;
  const closed = eyesClosedScore(det);
  let drift = null;
  if (genLm) {
    const fake = []; STABLE.forEach((idx, i) => { fake[idx] = { x: capturePts[i][0], y: capturePts[i][1] }; });
    const g = []; STABLE.forEach((idx) => { g[idx] = genLm[idx]; });
    drift = stableDrift(fake, g);
  }

  gl.bindTexture(gl.TEXTURE_2D, srcTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
  gl.bindBuffer(gl.ARRAY_BUFFER, warpTexBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(srcPts.flat()));
  const tri = new Delaunator(srcPts.flat()).triangles;
  gl.bindVertexArray(warpVao); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, warpIdxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(tri), gl.STATIC_DRAW); gl.bindVertexArray(null);
  warpCount = tri.length; state.triangles = Array.from(tri);

  const total = Math.round(performance.now() - t0);
  state.last = { model: settings.model, total, apiMs, closed, drift, detected: !!genLm };
  state.thumb = URL.createObjectURL(blob);
  $('#thumb').src = state.thumb;
  state.phase = 'ready'; state.target = 1; state.message = '';
}

function reset() {
  state.genId++; state.target = 0; state.phase = 'idle'; state.steadySince = 0; resetFilters();
}

function update(result, t) {
  const lm = result?.faceLandmarks?.[0];
  if (lm) state.lastSeen = t;
  const gone = t - state.lastSeen > 2000;
  if (gone && state.phase !== 'idle' && state.phase !== 'loading') reset();
  if (!lm) {
    state.steadySince = 0;
    // brief detector dropouts (motion blur, a hand) are bridged by optical flow alone for up to ~0.3 s
    if (tracker.pts && t - state.lastSeen < 300) return trackLive(null, t);
    tracker.reset();
    return null;
  }

  if (state.phase === 'idle' || state.phase === 'watching') {
    const { iod, yaw } = faceMetrics(lm);
    const blink = eyesClosedScore(result) ?? 0;
    const good = iod > 40 && Math.abs(yaw) < 0.18 && blink < 0.45;
    if (!settings.auto) state.phase = 'idle';
    else if (!good) { state.phase = 'watching'; state.steadySince = 0; state.message = blink >= 0.45 ? 'eyes open please' : 'look at the mirror'; }
    else {
      state.phase = 'watching'; state.message = 'hold still';
      if (!state.steadySince) state.steadySince = t;
      if (t - state.steadySince > 700) capture(result);
    }
  }
  updateHoles(lm); state.holes = true;
  return trackLive(lm, t);
}

// ---------------------------------------------------------------- HUD + mesh overlay
const overlay = $('#overlay'), octx = overlay.getContext('2d');
function drawOverlay(pts) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (settings.view !== 'mesh' || !pts) return;
  const tx = (x) => (settings.mirror ? 1 - x : x) * W;
  octx.lineWidth = 1;
  if (state.triangles) {
    octx.strokeStyle = 'rgba(80,255,180,0.7)'; octx.beginPath();
    for (let i = 0; i < state.triangles.length; i += 3) {
      const [a, b, c] = state.triangles.slice(i, i + 3);
      octx.moveTo(tx(pts[a * 2]), pts[a * 2 + 1] * H); octx.lineTo(tx(pts[b * 2]), pts[b * 2 + 1] * H);
      octx.lineTo(tx(pts[c * 2]), pts[c * 2 + 1] * H); octx.closePath();
    }
    octx.stroke();
  }
  STABLE.forEach((_, i) => {
    octx.fillStyle = WEIGHTS[i] ? '#ff3d7f' : '#3db8ff';
    octx.fillRect(tx(pts[i * 2]) - 2, pts[i * 2 + 1] * H - 2, 4, 4);
  });
}
function drawHud(fps) {
  const hud = $('#hud');
  hud.hidden = !settings.hud;
  $('#thumb').hidden = !(settings.hud && state.thumb && settings.view !== 'final');
  if (!settings.hud) return;
  const l = state.last;
  hud.innerHTML = `<b>${state.phase}</b>${state.message ? ' · ' + state.message : ''}<br>` +
    (l ? `${l.model} · ${(l.total / 1000).toFixed(1)}s round trip${l.apiMs ? ` (${(l.apiMs / 1000).toFixed(1)}s API)` : ''}` +
      ` · eyes-closed ${l.closed == null ? '?' : l.closed.toFixed(2)} · drift ${l.drift == null ? 'n/a' : (l.drift * 100).toFixed(1) + '% IOD'}<br>` : '') +
    `${fps.toFixed(0)} fps · tracking ${trackMs.toFixed(0)} ms/frame (display delayed by this)` +
    (settings.flow && tracker.stats.lkMs != null ? ` · optical flow ${tracker.stats.lkMs.toFixed(1)} ms` : '') +
    ` · view: ${settings.view}`;
}

// ---------------------------------------------------------------- UI
function toggleUI(show = !settings.hud) { settings.hud = show; $('#panel').hidden = !show; }
// double-tap anywhere (touch or mouse) toggles the settings
let lastTap = 0;
function onTap(e) {
  if (e.target.closest('#panel')) return;
  const now = performance.now();
  if (now - lastTap < 350) { toggleUI(); lastTap = 0; } else lastTap = now;
}
// keep a phone/tablet mirror from dimming and locking
let wakeLock = null;
async function keepAwake() {
  try { if (!wakeLock && document.visibilityState === 'visible') { wakeLock = await navigator.wakeLock?.request('screen'); wakeLock?.addEventListener('release', () => { wakeLock = null; }); } } catch {}
}
async function setupUI() {
  toggleUI(settings.hud);
  addEventListener('pointerup', onTap);
  keepAwake();
  document.addEventListener('visibilitychange', keepAwake);
  const sel = $('#model');
  try {
    const { models } = await (await fetch('/api/models')).json();
    sel.innerHTML = models.map((m) => `<option value="${m.id}">${m.label}</option>`).join('');
    if (!models.find((m) => m.id === settings.model) && models[0]) settings.model = models[0].id;
  } catch { sel.innerHTML = `<option>${settings.model}</option>`; }
  sel.value = settings.model;
  sel.onchange = () => { settings.model = sel.value; localStorage.setItem('ecm.model', sel.value); };
  $('#view').onchange = (e) => { settings.view = e.target.value; };
  $('#blur').oninput = (e) => { settings.blur = +e.target.value; };
  $('#smooth').oninput = (e) => { settings.smooth = +e.target.value; };
  $('#smooth').value = settings.smooth;
  $('#tracking').value = settings.flow ? 'flow' : settings.detector;
  $('#tracking').onchange = (e) => {
    const v = e.target.value;
    settings.flow = v === 'flow'; settings.detector = v === 'video' ? 'video' : 'image';
    resetFilters();
  };
  await refreshCameraList();
  $('#camera').onchange = async (e) => {
    try { await switchCamera(e.target.value); } catch (err) { state.message = `camera error: ${err.message}`; console.error(err); }
  };
  if (!demo) navigator.mediaDevices.addEventListener('devicechange', refreshCameraList);
  $('#auto').onchange = (e) => { settings.auto = e.target.checked; };
  $('#mirror').onchange = (e) => { settings.mirror = e.target.checked; };
  $('#capture').onclick = () => { if (lastResult?.faceLandmarks?.length) capture(lastResult); };
  $('#reset').onclick = reset;
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key === ' ') { e.preventDefault(); $('#capture').click(); }
    else if (e.key === 'r') reset();
    else if (e.key === 'h') toggleUI();
    else if (e.key === 'd') { settings.view = VIEWS[(VIEWS.indexOf(settings.view) + 1) % VIEWS.length]; $('#view').value = settings.view; }
    else if (e.key === 'f') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  });
}

// ---------------------------------------------------------------- main loop
// ---------------------------------------------------------------- tracking benchmark (?demo=..&bench=300&motion=fast)
// Steps the demo one simulated 30 fps frame per loop and compares tracked points with the known ground truth.
//   lag:    error projected on velocity, in frames (1.0 = drawn where the face was one frame ago)
//   wobble: RMS / p99 of the frame-to-frame change of the error, px (how much the overlay slides on the face)
//   err:    RMS error after removing each point's constant offset, px
const bench = params.has('bench') ? { k: 0, warm: 20, N: +params.get('bench') || 300, L0: null, prevTruth: null, sets: {} } : null;
function benchStep(lm, t) {
  const T = demoTransform(t);
  const truth = new Float32Array(STABLE.length * 2);
  STABLE.forEach((idx, i) => { const [x, y] = demoMap(T, bench.L0[i][0], bench.L0[i][1]); truth[2 * i] = x; truth[2 * i + 1] = y; });
  const est = {
    detector: lm ? STABLE.flatMap((idx) => [lm[idx].x * W, lm[idx].y * H]) : null,
    output: lm || tracker.pts ? Array.from(livePts, (v, i) => v * (i % 2 ? H : W)) : null,
  };
  if (bench.prevTruth && bench.k > bench.warm) {
    for (const [name, e] of Object.entries(est)) {
      const S = (bench.sets[name] ??= { ev: 0, vv: 0, frames: [], prevE: null, d: [] });
      if (!e) { S.prevE = null; continue; }
      const err = new Float32Array(truth.length);
      for (let i = 0; i < truth.length; i++) {
        err[i] = e[i] - truth[i];
        const v = truth[i] - bench.prevTruth[i];
        S.ev += err[i] * v; S.vv += v * v;
      }
      if (S.prevE) for (let i = 0; i < truth.length; i += 2) S.d.push(Math.hypot(err[i] - S.prevE[i], err[i + 1] - S.prevE[i + 1]));
      S.prevE = err; S.frames.push(err);
    }
  }
  bench.prevTruth = truth;
  if (++bench.k === bench.warm + bench.N) {
    const out = { tstats: tracker.stats, motion: params.get('motion') || 'slow', detector: settings.detector, flow: settings.flow, smooth: settings.smooth, frames: bench.N, trackMs: +trackMs.toFixed(1) };
    for (const [name, S] of Object.entries(bench.sets)) {
      const n = truth.length, mean = new Float64Array(n);
      for (const f of S.frames) for (let i = 0; i < n; i++) mean[i] += f[i] / S.frames.length;
      let se = 0, cnt = 0;
      for (const f of S.frames) for (let i = 0; i < n; i += 2) { se += (f[i] - mean[i]) ** 2 + (f[i + 1] - mean[i + 1]) ** 2; cnt++; }
      const d = S.d.sort((a, b) => a - b);
      out[name] = {
        lag: +(-S.ev / S.vv).toFixed(3),
        wobbleRms: +Math.sqrt(d.reduce((a, b) => a + b * b, 0) / d.length).toFixed(3),
        wobbleP99: +d[Math.floor(d.length * 0.99)].toFixed(3),
        errRms: +Math.sqrt(se / cnt).toFixed(3),
        missed: bench.N - S.frames.length,
      };
    }
    window.__bench = out; console.log('BENCH', JSON.stringify(out));
  }
}

let lastResult = null, lastT = performance.now(), fps = 0, trackMs = 0;
function loop(t) {
  if (bench) {
    if (window.__bench) return; // done
    t = bench.k * 1000 / 30;     // deterministic simulated 30 fps, independent of how fast the page runs
  }
  let pts = null;
  const fresh = grabFrame(t);
  if (fresh) {
    const t0 = performance.now();
    // synchronous: landmarks belong to exactly this snapshot
    lastResult = settings.detector === 'video' ? videoLM.detectForVideo(frameSource, t) : imageLM.detect(frameSource);
    pts = update(lastResult, t);
    trackMs += (performance.now() - t0 - trackMs) * 0.1;
    if (bench) benchStep(lastResult?.faceLandmarks?.[0], t);
  } else if (lastResult?.faceLandmarks?.length) pts = livePts;
  // fade in/out
  const dt = (t - lastT) / 1000; lastT = t;
  fps += (1 / Math.max(dt, 1e-3) - fps) * 0.05;
  state.amount += Math.sign(state.target - state.amount) * Math.min(Math.abs(state.target - state.amount), dt / 0.8);
  if (state.amount === 0 && state.target === 0 && state.phase === 'idle') warpCount = 0;
  render(pts, fresh);
  drawOverlay(pts);
  drawHud(fps);
  requestAnimationFrame(loop);
}

(async function main() {
  try {
    $('#hud').textContent = 'loading face tracker…';
    await setupSource();
    setupGL();
    [videoLM, imageLM] = await Promise.all([createLandmarker('VIDEO'), createLandmarker('IMAGE')]);
    await setupUI();
    state.phase = 'idle';
    if (bench) { const r = imageLM.detect(demoImg).faceLandmarks[0]; bench.L0 = STABLE.map((i) => [r[i].x * demoImg.width, r[i].y * demoImg.height]); }
    window.__ecm = { state, settings, tracker }; // for debugging / automated screenshots
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    $('#hud').hidden = false;
    $('#hud').textContent = `error: ${e.message}`;
  }
})();
