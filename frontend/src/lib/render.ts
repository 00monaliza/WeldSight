import { classColor, hexToRgb } from "./classes";
import type { AnalysisResult, ScoreKind } from "./types";

// black -> violet -> orange -> pale yellow
const LUT = (() => {
  const stops: [number, number[]][] = [[0, [0, 0, 0]], [0.35, [90, 20, 120]], [0.7, [240, 100, 30]], [1, [255, 240, 150]]];
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 1;
    while (k < stops.length - 1 && t > stops[k][0]) k++;
    const [t0, c0] = stops[k - 1], [t1, c1] = stops[k], f = (t - t0) / (t1 - t0);
    for (let j = 0; j < 3; j++) out[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * f;
  }
  return out;
})();

export function drawOverlay(
  ctx: CanvasRenderingContext2D, w: number, h: number, res: AnalysisResult,
  score: ScoreKind, opts: { seg: boolean; unc: boolean; opacity: number },
) {
  const img = ctx.createImageData(w, h), d = img.data, a = opts.opacity, map = res.maps[score], labels = res.labels;
  if (opts.unc) {
    for (let i = 0; i < w * h; i++) {
      const u = map[i];
      if (u < 0.03) continue;
      const k = Math.min(255, (u * 255) | 0) * 3;
      d[i * 4] = LUT[k]; d[i * 4 + 1] = LUT[k + 1]; d[i * 4 + 2] = LUT[k + 2];
      d[i * 4 + 3] = Math.min(255, 255 * a * Math.min(1, u * 1.6));
    }
  }
  if (opts.seg) {
    const rgb = res.classes.map((c) => hexToRgb(classColor(c)));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x, c = labels[i];
        if (!c || !rgb[c - 1]) continue;
        const edge = x === 0 || y === 0 || x === w - 1 || y === h - 1 || labels[i - 1] !== c || labels[i + 1] !== c || labels[i - w] !== c || labels[i + w] !== c;
        const col = rgb[c - 1];
        d[i * 4] = col[0]; d[i * 4 + 1] = col[1]; d[i * 4 + 2] = col[2];
        d[i * 4 + 3] = edge ? 255 : 255 * a * 0.7;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function drawBoxes(ctx: CanvasRenderingContext2D, w: number, h: number, res: AnalysisResult | null, active: number) {
  ctx.clearRect(0, 0, w, h);
  if (!res) return;
  const lw = Math.max(1.5, w / 600), pad = 4 + lw;
  res.defects.forEach((d, i) => {
    const [x, y, bw, bh] = d.bbox, on = i === active;
    ctx.lineWidth = on ? lw * 2 : lw;
    ctx.strokeStyle = d.suspect ? "#ffffffcc" : classColor(d.cls);
    ctx.setLineDash(d.suspect ? [6, 4] : []);
    ctx.globalAlpha = on ? 1 : active >= 0 ? 0.35 : 0.85;
    ctx.strokeRect(x - pad, y - pad, bw + 2 * pad, bh + 2 * pad);
  });
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
}
