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
      state: "refer", score, title: "Передать инспектору",
      sub: `Неуверенность ${score.toFixed(2)} ≥ ${thr.toFixed(2)} на области «${pretty(worst.cls).toLowerCase()}»${worst.suspect ? " (ниже порога обнаружения)" : ""}. Полагаться только на модель нельзя.`,
    };
  }
  if (real.length) {
    return { state: "flag", score, title: `Найдено дефектов: ${real.length}`, sub: "Модель достаточно уверена, чтобы обработать снимок автоматически. Выборочная проверка — по регламенту контроля качества." };
  }
  return { state: "clear", score, title: "Дефектов не найдено", sub: "Ничего выше порога обнаружения и ничего достаточно неуверенного, чтобы передать инспектору." };
}

export function decisionLabel(d: Decision): string {
  return d.state === "refer" ? "refer_to_inspector" : d.state === "flag" ? "defects_flagged" : d.state === "clear" ? "clear" : "none";
}
