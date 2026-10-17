import { CLASSES } from "./classes";
import type { AnalysisResult, Defect } from "./types";

/**
 * Browser-only stand-in for the model: a blur-difference detector with simulated MC noise.
 * It exists to exercise the UI. It is NOT the trained WeldSight model.
 */
export const DEMO = { gate: 2.5, offset: 5.2, scale: 0.8, jitter: 0.9, lo: 0.2, minArea: 18, minPeak: 0.35 };
const LN2 = Math.LN2;
const clamp = (v: number, m: number) => (v < 0 ? 0 : v > m ? m : v);

export function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h), d = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[o + clamp(x, w - 1)];
    for (let x = 0; x < w; x++) { tmp[o + x] = s / d; s += src[o + clamp(x + r + 1, w - 1)] - src[o + clamp(x - r, w - 1)]; }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[clamp(y, h - 1) * w + x];
    for (let y = 0; y < h; y++) { out[y * w + x] = s / d; s += tmp[clamp(y + r + 1, h - 1) * w + x] - tmp[clamp(y - r, h - 1) * w + x]; }
  }
  return out;
}

const H = (p: number) => { p = Math.min(1 - 1e-6, Math.max(1e-6, p)); return -(p * Math.log(p) + (1 - p) * Math.log(1 - p)); };
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
const gauss = () => { let s = 0; for (let i = 0; i < 4; i++) s += Math.random(); return (s - 2) * 1.73; };

function infer(gray: Float32Array, w: number, h: number, T: number) {
  const n = w * h, bg = boxBlur(gray, w, h, 14), sm = boxBlur(gray, w, h, 1), z = new Float32Array(n), sample: number[] = [];
  for (let i = 0; i < n; i++) z[i] = bg[i] - sm[i];
  for (let i = 0; i < n; i += 7) sample.push(z[i]);
  sample.sort((a, b) => a - b);
  const med = sample[sample.length >> 1];
  const dev = sample.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
  const sd = Math.max(0.5, 1.4826 * dev[dev.length >> 1]);
  for (let i = 0; i < n; i++) z[i] = (z[i] - med) / sd;
  const meanP = new Float32Array(n), pe = new Float32Array(n), mi = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (z[i] < DEMO.gate) continue;
    let sp = 0, se = 0;
    for (let t = 0; t < T; t++) { const p = sigmoid((z[i] + gauss() * DEMO.jitter - DEMO.offset) / DEMO.scale); sp += p; se += H(p); }
    const m = sp / T;
    meanP[i] = m; pe[i] = H(m) / LN2; mi[i] = Math.min(1, Math.max(0, (H(m) - se / T) / 0.3));
  }
  return { meanP, pe, mi };
}

interface Comp { px: number[]; bbox: [number, number, number, number] }

function components(meanP: Float32Array, w: number, h: number): Comp[] {
  const seen = new Uint8Array(w * h), comps: Comp[] = [], stack: number[] = [], lo = DEMO.lo;
  for (let s = 0; s < w * h; s++) {
    if (seen[s] || meanP[s] < lo) continue;
    const px: number[] = [];
    let x0 = w, y0 = h, x1 = 0, y1 = 0;
    seen[s] = 1; stack.push(s);
    while (stack.length) {
      const i = stack.pop()!;
      px.push(i);
      const x = i % w, y = (i / w) | 0;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (x > 0 && !seen[i - 1] && meanP[i - 1] >= lo) { seen[i - 1] = 1; stack.push(i - 1); }
      if (x < w - 1 && !seen[i + 1] && meanP[i + 1] >= lo) { seen[i + 1] = 1; stack.push(i + 1); }
      if (y > 0 && !seen[i - w] && meanP[i - w] >= lo) { seen[i - w] = 1; stack.push(i - w); }
      if (y < h - 1 && !seen[i + w] && meanP[i + w] >= lo) { seen[i + w] = 1; stack.push(i + w); }
    }
    if (px.length >= DEMO.minArea) comps.push({ px, bbox: [x0, y0, x1 - x0 + 1, y1 - y0 + 1] });
  }
  return comps;
}

/** Shape rules; elongation from the covariance of the pixel cloud, not the bbox (cracks run diagonally). */
export function classify(c: Comp, w: number, h: number): string {
  const [, y, bw, bh] = c.bbox, area = c.px.length;
  let mx = 0, my = 0;
  for (const i of c.px) { mx += i % w; my += (i / w) | 0; }
  mx /= area; my /= area;
  let sxx = 0, syy = 0, sxy = 0;
  for (const i of c.px) { const dx = (i % w) - mx, dy = ((i / w) | 0) - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  sxx /= area; syy /= area; sxy /= area;
  const tr = sxx + syy, det = sxx * syy - sxy * sxy, disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(1e-6, tr / 2 - disc);
  const elong = Math.sqrt(l1 / l2), angle = (Math.abs(0.5 * Math.atan2(2 * sxy, sxx - syy)) * 180) / Math.PI; // 0 = horizontal
  const horizontal = angle < 20, fill = area / (bw * bh);
  if (elong > 5) return horizontal ? "lack_of_penetration" : "crack";
  if (elong > 3 && horizontal) return y < h * 0.3 || y > h * 0.7 ? "undercut" : "lack_of_fusion";
  if (area < 160 && fill > 0.45) return "porosity";
  return "inclusion";
}

function meanOver(arr: Float32Array, idx: number[]) {
  let s = 0;
  for (const i of idx) s += arr[i];
  return s / idx.length;
}

/** display only: spread the per-pixel score so thin regions are visible */
function halo(map: Float32Array, w: number, h: number) {
  const b = boxBlur(map, w, h, 4);
  for (let i = 0; i < b.length; i++) b[i] = Math.min(1, Math.max(map[i], b[i] * 3));
  return b;
}

export function runDemo(gray: Float32Array, w: number, h: number, T: number): AnalysisResult {
  const { meanP, pe, mi } = infer(gray, w, h, T);
  const classes = CLASSES.map((c) => c.id), labels = new Uint8Array(w * h), defects: Defect[] = [];
  for (const c of components(meanP, w, h)) {
    let sp = 0, peak = 0;
    for (const i of c.px) { sp += meanP[i]; if (meanP[i] > peak) peak = meanP[i]; }
    if (peak < DEMO.minPeak) continue;
    const prob = sp / c.px.length, cls = classify(c, w, h), suspect = prob < 0.5;
    if (!suspect) { const k = classes.indexOf(cls) + 1; for (const i of c.px) labels[i] = k; }
    defects.push({ cls, prob, bbox: c.bbox, suspect, unc: { pe: meanOver(pe, c.px), mi: meanOver(mi, c.px) } });
  }
  return { classes, labels, maps: { pe: halo(pe, w, h), mi: halo(mi, w, h) }, defects, model: "demo (browser heuristic)", stub: true };
}
