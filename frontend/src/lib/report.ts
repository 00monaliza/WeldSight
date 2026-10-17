import { canvasToBlob, download } from "./imaging";
import { decide, decisionLabel } from "./decision";
import { drawBoxes, drawOverlay } from "./render";
import type { AnalysisResult, Film, Settings } from "./types";

const stem = (n: string) => n.replace(/\.\w+$/, "");

export async function savePng(film: Film, res: AnalysisResult, s: Settings) {
  const out = document.createElement("canvas"), over = document.createElement("canvas"), hi = document.createElement("canvas");
  for (const c of [out, over, hi]) { c.width = film.w; c.height = film.h; }
  drawOverlay(over.getContext("2d")!, film.w, film.h, res, s.score, { seg: s.showSeg, unc: s.showUnc, opacity: s.opacity });
  if (s.showBox) drawBoxes(hi.getContext("2d")!, film.w, film.h, res, -1);
  const g = out.getContext("2d")!;
  g.drawImage(film.base, 0, 0); g.drawImage(over, 0, 0); g.drawImage(hi, 0, 0);
  download(`${stem(film.name)}_weldsight.png`, await canvasToBlob(out));
}

export function saveJson(film: Film, res: AnalysisResult, s: Settings) {
  const d = decide(res, s.score, s.thr);
  const report = {
    film: film.name, width: film.w, height: film.h, engine: s.engine, model: res.model, mc_samples: s.mc,
    uncertainty_score: s.score, refer_threshold: s.thr, film_uncertainty: "score" in d ? +d.score.toFixed(4) : 0,
    decision: decisionLabel(d),
    note: res.stub ? "Produced by a heuristic stand-in, not the trained model." : undefined,
    defects: res.defects,
  };
  download(`${stem(film.name)}_weldsight.json`, new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
}
