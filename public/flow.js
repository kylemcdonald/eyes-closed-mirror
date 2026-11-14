// Frame-to-frame point tracking: pyramidal Lucas-Kanade (Bouguet) on a grayscale ROI, plus a fused tracker
// that combines optical flow (zero lag, very low jitter, but drifts) with per-frame landmark detection
// (no drift, but noisy) as a complementary filter.

const LEVELS = 4;     // pyramid levels: 1, 1/2, 1/4, 1/8
export let HW = 8;    // LK window half-size -> 17x17
const MAX_ITER = 12;

// ---------------------------------------------------------------- grayscale pyramid of a region of the frame
export function buildPyramid(rgba, w, h, x0, y0, reuse) {
  const levels = [];
  let lw = w, lh = h;
  let g = reuse?.levels[0]?.data.length === w * h ? reuse.levels[0].data : new Float32Array(w * h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) g[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  levels.push({ data: g, w: lw, h: lh });
  for (let L = 1; L < LEVELS; L++) {
    const src = levels[L - 1];
    const nw = src.w >> 1, nh = src.h >> 1;
    const prevBuf = reuse?.levels[L]?.data;
    const d = prevBuf && prevBuf.length === nw * nh ? prevBuf : new Float32Array(nw * nh);
    const s = src.data, sw = src.w;
    for (let y = 0; y < nh; y++) {
      const r0 = 2 * y * sw, r1 = r0 + sw;
      for (let x = 0; x < nw; x++) {
        const c = 2 * x;
        d[y * nw + x] = 0.25 * (s[r0 + c] + s[r0 + c + 1] + s[r1 + c] + s[r1 + c + 1]);
      }
    }
    levels.push({ data: d, w: nw, h: nh });
  }
  return { levels, x0, y0, w, h };
}

function sample(l, x, y) {
  const w = l.w, h = l.h;
  if (x < 0) x = 0; else if (x > w - 1.001) x = w - 1.001;
  if (y < 0) y = 0; else if (y > h - 1.001) y = h - 1.001;
  const ix = x | 0, iy = y | 0, fx = x - ix, fy = y - iy;
  const i = iy * w + ix, d = l.data;
  const a = d[i] + (d[i + 1] - d[i]) * fx;
  const b = d[i + w] + (d[i + w + 1] - d[i + w]) * fx;
  return a + (b - a) * fy;
}

let N_WIN, T, GX, GY, J;
export function setWindow(hw) { HW = hw; N_WIN = (2 * HW + 1) ** 2; T = new Float32Array(N_WIN); GX = new Float32Array(N_WIN); GY = new Float32Array(N_WIN); J = new Float32Array(N_WIN); }
setWindow(HW);

// Track n points (frame pixel coords in `pts`) from pyramid `prev` to `cur`.
// guess: optional initial displacement per point. Writes positions to `out`, and per point
// quality (min eigenvalue of the structure tensor per pixel, 0 = failed) and residual (mean |dI|).
export function lucasKanade(prev, cur, pts, n, guess, out, quality, residual) {
  for (let i = 0; i < n; i++) {
    const px = pts[2 * i], py = pts[2 * i + 1];
    let gx = guess ? guess[2 * i] / (1 << (LEVELS - 1)) : 0;
    let gy = guess ? guess[2 * i + 1] / (1 << (LEVELS - 1)) : 0;
    let ok = true, minEig = 0, res = 0;
    for (let L = LEVELS - 1; L >= 0; L--) {
      const s = 1 / (1 << L);
      const P = prev.levels[L], C = cur.levels[L];
      const ux = (px - prev.x0) * s, uy = (py - prev.y0) * s;
      const ox = (prev.x0 - cur.x0) * s, oy = (prev.y0 - cur.y0) * s;
      let g11 = 0, g12 = 0, g22 = 0, k = 0;
      for (let wy = -HW; wy <= HW; wy++) {
        for (let wx = -HW; wx <= HW; wx++, k++) {
          const x = ux + wx, y = uy + wy;
          T[k] = sample(P, x, y);
          const ix = 0.5 * (sample(P, x + 1, y) - sample(P, x - 1, y));
          const iy = 0.5 * (sample(P, x, y + 1) - sample(P, x, y - 1));
          GX[k] = ix; GY[k] = iy;
          g11 += ix * ix; g12 += ix * iy; g22 += iy * iy;
        }
      }
      const det = g11 * g22 - g12 * g12;
      if (det < 1e-6) { ok = false; break; }
      let meanT = 0;
      for (let j = 0; j < N_WIN; j++) meanT += T[j];
      meanT /= N_WIN;
      if (L === 0) {
        const tr = g11 + g22;
        minEig = (tr - Math.sqrt(Math.max(0, tr * tr - 4 * det))) / 2 / N_WIN;
      }
      let vx = 0, vy = 0;
      for (let it = 0; it < MAX_ITER; it++) {
        let bx = 0, by = 0, e = 0, meanJ = 0;
        const cx = ux + gx + vx + ox, cy = uy + gy + vy + oy;
        k = 0;
        for (let wy = -HW; wy <= HW; wy++) {
          for (let wx = -HW; wx <= HW; wx++, k++) { J[k] = sample(C, cx + wx, cy + wy); meanJ += J[k]; }
        }
        // brightness offset compensation: camera auto-exposure shouldn't read as motion
        const off = meanT - meanJ / N_WIN;
        for (k = 0; k < N_WIN; k++) {
          const dI = T[k] - J[k] - off;
          bx += dI * GX[k]; by += dI * GY[k]; e += Math.abs(dI);
        }
        res = e / N_WIN;
        const ex = (g22 * bx - g12 * by) / det, ey = (g11 * by - g12 * bx) / det;
        vx += ex; vy += ey;
        if (ex * ex + ey * ey < 1e-4) break;
      }
      if (L > 0) { gx = 2 * (gx + vx); gy = 2 * (gy + vy); } else { gx += vx; gy += vy; }
    }
    out[2 * i] = px + gx; out[2 * i + 1] = py + gy;
    quality[i] = ok ? minEig : 0;
    residual[i] = ok ? res : 1e9;
  }
}

// Weighted least-squares 2D affine fit src -> dst. Returns [a, b, c, d, tx, ty]: x' = a x + b y + tx, y' = c x + d y + ty.
function fitAffine(src, dst, w, n) {
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0, su = 0, sv = 0, sxu = 0, syu = 0, sxv = 0, syv = 0;
  for (let i = 0; i < n; i++) {
    const wi = w[i]; if (!wi) continue;
    const x = src[2 * i], y = src[2 * i + 1], u = dst[2 * i], v = dst[2 * i + 1];
    sw += wi; sx += wi * x; sy += wi * y; sxx += wi * x * x; sxy += wi * x * y; syy += wi * y * y;
    su += wi * u; sv += wi * v; sxu += wi * x * u; syu += wi * y * u; sxv += wi * x * v; syv += wi * y * v;
  }
  if (sw < 3) return null;
  // centre for conditioning
  const mx = sx / sw, my = sy / sw, mu = su / sw, mv = sv / sw;
  const cxx = sxx / sw - mx * mx, cxy = sxy / sw - mx * my, cyy = syy / sw - my * my;
  const cxu = sxu / sw - mx * mu, cyu = syu / sw - my * mu, cxv = sxv / sw - mx * mv, cyv = syv / sw - my * mv;
  const det = cxx * cyy - cxy * cxy;
  if (Math.abs(det) < 1e-9) return null;
  const a = (cxu * cyy - cyu * cxy) / det, b = (cyu * cxx - cxu * cxy) / det;
  const c = (cxv * cyy - cyv * cxy) / det, d = (cyv * cxx - cxv * cxy) / det;
  return [a, b, c, d, mu - a * mx - b * my, mv - c * mx - d * my];
}

// ---------------------------------------------------------------- fused tracker
export class FusedTracker {
  constructor(n) {
    this.n = n;
    this.pts = null;                       // current fused estimate (frame px)
    this.vel = new Float32Array(2 * n);    // last per-point displacement, used as LK initial guess
    this.flow = new Float32Array(2 * n);
    this.q = new Float32Array(n); this.r = new Float32Array(n); this.w = new Float32Array(n);
    this.pyr = null; this.spare = null;
    this.alpha = 0.1;    // detector weight per frame for small (noise-level) disagreements
    this.sigmaFrac = 0.012; // disagreement scale (fraction of eye-band width) above which the detector takes over
    this.snapPx = 12;    // if detection and prediction disagree by more than this (median), re-sync
    this.mode = 'affine'; // 'affine': all points follow one robust affine fit of the flow; 'points': per-point flow
    this.stats = {};
  }
  reset() { this.pts = null; this.pyr = null; }

  // ctx: 2D context holding the current frame. det: Float32Array of detected points (frame px) or null.
  update(ctx, W, H, det, useFlow = true) {
    const n = this.n;
    if (!useFlow) { this.pts = det ? det.slice() : null; this.pyr = null; return this.pts; }
    const ref = this.pts || det;
    if (!ref) return null;
    // ROI: bbox of current points + margin for motion and the LK window at the coarsest level.
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = ref[2 * i], y = ref[2 * i + 1];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    const M = 64 + 0.25 * (x1 - x0);
    const A = 1 << (LEVELS - 1); // align ROI to the coarsest pixel grid so pyramids line up between frames
    x0 = Math.max(0, Math.floor((x0 - M) / A) * A); y0 = Math.max(0, Math.floor((y0 - M) / A) * A);
    x1 = Math.min(Math.floor(W / A) * A, Math.ceil((x1 + M) / A) * A); y1 = Math.min(Math.floor(H / A) * A, Math.ceil((y1 + M) / A) * A);
    const rw = x1 - x0, rh = y1 - y0;
    if (rw < 32 || rh < 32) { this.reset(); return det; }
    const t0 = performance.now();
    const img = ctx.getImageData(x0, y0, rw, rh);
    const cur = buildPyramid(img.data, rw, rh, x0, y0, this.spare);
    const t1 = performance.now();

    if (!this.pts || !this.pyr) {
      if (!det) return null;
      this.pts = det.slice(); this.vel.fill(0);
      this.spare = this.pyr; this.pyr = cur;
      return this.pts;
    }

    // 1. optical flow from the previous frame
    lucasKanade(this.pyr, cur, this.pts, n, this.vel, this.flow, this.q, this.r);
    const t2 = performance.now();
    // 2. robust affine motion model over all points (IRLS, Huber), to reject bad individual flows
    for (let i = 0; i < n; i++) this.w[i] = this.q[i] > 2 && this.r[i] < 25 ? Math.min(this.q[i], 50) : 0;
    let Aff = null;
    for (let it = 0; it < 3; it++) {
      Aff = fitAffine(this.pts, this.flow, this.w, n);
      if (!Aff) break;
      for (let i = 0; i < n; i++) {
        if (this.q[i] <= 2 || this.r[i] >= 25) { this.w[i] = 0; continue; }
        const x = this.pts[2 * i], y = this.pts[2 * i + 1];
        const ex = Aff[0] * x + Aff[1] * y + Aff[4] - this.flow[2 * i], ey = Aff[2] * x + Aff[3] * y + Aff[5] - this.flow[2 * i + 1];
        const e = Math.hypot(ex, ey);
        this.w[i] = Math.min(this.q[i], 50) * (e < 0.5 ? 1 : 0.5 / e); // texture strength x Huber(0.5 px)
      }
    }
    // prediction: own flow if consistent with the model (allows non-rigid motion like brows), else model
    const pred = this.flow;
    let outliers = 0;
    for (let i = 0; i < n; i++) {
      if (!Aff) break;
      const x = this.pts[2 * i], y = this.pts[2 * i + 1];
      const mx = Aff[0] * x + Aff[1] * y + Aff[4], my = Aff[2] * x + Aff[3] * y + Aff[5];
      const bad = this.mode === 'affine' || this.q[i] <= 2 || this.r[i] >= 25 || Math.hypot(pred[2 * i] - mx, pred[2 * i + 1] - my) > 3;
      if (bad) { pred[2 * i] = mx; pred[2 * i + 1] = my; outliers++; }
    }
    // 3. complementary filter with the detector
    let alpha = this.alpha, med = 0;
    if (det) {
      const errs = new Float32Array(n);
      for (let i = 0; i < n; i++) errs[i] = Math.hypot(det[2 * i] - pred[2 * i], det[2 * i + 1] - pred[2 * i + 1]);
      med = errs.slice().sort()[n >> 1];
      if (med > this.snapPx) alpha = 1; // lost track: re-sync to the detector
    } else alpha = 0;
    // Per point: small detector/prediction disagreements are detector noise -> heavy filtering (alpha);
    // large ones are real non-rigid motion (brows, cheeks) -> follow the detector quickly.
    const sigma = Math.max(1.5, this.sigmaFrac * (x1 - x0 - 2 * M));
    for (let i = 0; i < n; i++) {
      let a = alpha;
      if (det && alpha < 1) {
        const e = Math.hypot(det[2 * i] - pred[2 * i], det[2 * i + 1] - pred[2 * i + 1]) / sigma;
        a = alpha + (1 - alpha) * (1 - Math.exp(-0.5 * e * e));
      }
      for (let k = 2 * i; k < 2 * i + 2; k++) {
        const p = det ? pred[k] + a * (det[k] - pred[k]) : pred[k];
        this.vel[k] = p - this.pts[k];
        this.pts[k] = p;
      }
    }
    this.spare = this.pyr; this.pyr = cur;
    this.stats = { roi: `${rw}x${rh}`, readMs: t1 - t0, lkMs: t2 - t1, outliers, med, alpha };
    return this.pts;
  }
}
