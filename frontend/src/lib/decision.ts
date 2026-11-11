import { pretty } from "./classes";
import type { AnalysisResult, Decision, ScoreKind } from "./types";

export const defectUnc = (d: AnalysisResult["defects"][number], s: ScoreKind) => d.unc[s];

/** Film score = max defect uncertainty (the radiograph score used for the referral curve). */
export function filmScore(res: AnalysisResult, s: ScoreKind): number {
  return res.defects.reduce((m, d) => Math.max(m, d.unc[s]), 0);
}

export function decide(res: AnalysisResult, s: ScoreKind, thr: number): Decision {
  const score = filmScore(res, s);
  const real = res.defects.filter((d) => !d.suspect);
  if (res.defects.length && score >= thr) {
    const worst = res.defects.reduce((a, b) => (b.unc[s] > a.unc[s] ? b : a));
    return {
      state: "refer", score, title: "Send to inspector",
      sub: `Uncertainty ${score.toFixed(2)} ≥ ${thr.toFixed(2)} on ${worst.suspect ? "a sub-threshold " : "a "}${pretty(worst.cls).toLowerCase()} region. Do not rely on the model alone.`,
    };
  }
  if (real.length) {
    return { state: "flag", score, title: `${real.length} defect${real.length > 1 ? "s" : ""} flagged`, sub: "Model is confident enough to handle this film automatically. Spot-check per your QA policy." };
  }
  return { state: "clear", score, title: "No defects found", sub: "Nothing above the detection threshold, and nothing uncertain enough to refer." };
}

export function decisionLabel(d: Decision): string {
  return d.state === "refer" ? "refer_to_inspector" : d.state === "flag" ? "defects_flagged" : d.state === "clear" ? "clear" : "none";
}
