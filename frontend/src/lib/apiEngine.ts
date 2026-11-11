import { canvasToBlob, MAX_DIM } from "./imaging";
import type { AnalysisResult, Film } from "./types";

export interface Health {
  status: string;
  model: string;
  stub: boolean;
  classes: string[];
  device?: string;
}

interface PredictResponse {
  width: number;
  height: number;
  classes: string[];
  mask_png: string;
  maps?: { pe?: string; mi?: string };
  defects: { cls: string; prob: number; bbox: number[]; uncertainty?: { pe?: number; mi?: number } }[];
  model?: string;
  stub?: boolean;
}

const base = (url: string) => url.trim().replace(/\/+$/, "");
const PREDICT_TIMEOUT_MS = 120_000;
const HEALTH_TIMEOUT_MS = 5_000;

function withTimeout(signal: AbortSignal | undefined, ms: number): AbortSignal {
  const t = AbortSignal.timeout(ms);
  return signal ? AbortSignal.any([signal, t]) : t;
}

export async function fetchHealth(apiUrl: string, signal?: AbortSignal): Promise<Health> {
  const r = await fetch(`${base(apiUrl)}/api/health`, { signal: withTimeout(signal, HEALTH_TIMEOUT_MS) });
  if (!r.ok) throw new Error(`health check failed (${r.status})`);
  return r.json();
}

/** R channel of a (grey or colour) PNG, resampled to w x h */
async function pngChannel(b64: string, w: number, h: number, smooth: boolean): Promise<Uint8Array> {
  const blob = await (await fetch(`data:image/png;base64,${b64}`)).blob();
  const bmp = await createImageBitmap(blob);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true })!;
  g.imageSmoothingEnabled = smooth;
  g.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const d = g.getImageData(0, 0, w, h).data, out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4];
  return out;
}

export async function runApi(film: Film, apiUrl: string, mc: number, signal?: AbortSignal): Promise<AnalysisResult> {
  const body = new FormData();
  body.append("image", film.file ?? (await canvasToBlob(film.base)), film.file?.name ?? film.name);
  body.append("mc_samples", String(mc));
  body.append("max_dim", String(MAX_DIM));
  let resp: Response;
  try {
    resp = await fetch(`${base(apiUrl)}/api/predict`, { method: "POST", body, signal: withTimeout(signal, PREDICT_TIMEOUT_MS) });
  } catch (e) {
    if ((e as Error).name === "TimeoutError") throw new Error(`API did not answer within ${PREDICT_TIMEOUT_MS / 1000}s`);
    throw e;
  }
  if (!resp.ok) {
    let detail = "";
    try { const j = await resp.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    throw new Error(`API ${resp.status}${detail ? `: ${detail}` : ""}`);
  }
  const j: PredictResponse = await resp.json();
  // A missing uncertainty must not read as "certain": refuse the response instead of defaulting to 0.
  if (!j.mask_png || !j.maps?.pe || !j.maps?.mi || !Array.isArray(j.defects) || !Array.isArray(j.classes)) {
    throw new Error("API response is missing mask_png, maps.pe/mi, classes or defects");
  }
  if (j.defects.some((d) => typeof d.uncertainty?.pe !== "number" || typeof d.uncertainty?.mi !== "number")) {
    throw new Error("API response has a region without uncertainty.pe / uncertainty.mi");
  }
  const { w, h } = film;
  const sx = w / (j.width || w), sy = h / (j.height || h);
  const labels = await pngChannel(j.mask_png, w, h, false);
  const unit = async (b64: string) => {
    const f = new Float32Array(w * h);
    const u = await pngChannel(b64, w, h, true);
    for (let i = 0; i < f.length; i++) f[i] = u[i] / 255;
    return f;
  };
  return {
    classes: j.classes,
    labels,
    maps: { pe: await unit(j.maps.pe), mi: await unit(j.maps.mi) },
    defects: j.defects.map((d) => ({
      cls: d.cls,
      prob: d.prob,
      suspect: d.prob < 0.5,
      bbox: [d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy].map(Math.round) as [number, number, number, number],
      unc: { pe: d.uncertainty!.pe!, mi: d.uncertainty!.mi! },
    })),
    model: j.model,
    stub: j.stub,
  };
}
