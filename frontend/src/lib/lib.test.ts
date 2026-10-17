import { describe, expect, it } from "vitest";
import { decide, filmScore } from "./decision";
import { runDemo } from "./demoEngine";
import { workingSize } from "./imaging";
import type { AnalysisResult, Defect } from "./types";

const defect = (o: Partial<Defect> & { pe: number }): Defect => ({
  cls: "crack", prob: 0.9, bbox: [0, 0, 5, 5], suspect: false, ...o, unc: { pe: o.pe, mi: o.pe / 2 },
});
const res = (defects: Defect[]): AnalysisResult => ({ classes: ["crack"], labels: new Uint8Array(1), maps: { pe: new Float32Array(1), mi: new Float32Array(1) }, defects });

describe("workingSize", () => {
  it("matches the server rule (never upscales, keeps aspect)", () => {
    expect(workingSize(2000, 400)).toEqual([1024, 205]);
    expect(workingSize(300, 200)).toEqual([300, 200]);
  });
});

describe("decision", () => {
  it("refers when the max defect uncertainty reaches the threshold", () => {
    const r = res([defect({ pe: 0.1 }), defect({ pe: 0.7, suspect: true, prob: 0.4 })]);
    expect(filmScore(r, "pe")).toBe(0.7);
    expect(decide(r, "pe", 0.55).state).toBe("refer");
    expect(decide(r, "pe", 0.9).state).toBe("flag");
  });
  it("is clear without findings", () => {
    expect(decide(res([]), "pe", 0.5).state).toBe("clear");
  });
  it("only suspects and confident -> clear", () => {
    expect(decide(res([defect({ pe: 0.1, suspect: true })]), "pe", 0.5).state).toBe("clear");
  });
});

describe("demo engine", () => {
  it("finds a dark blob on a flat noisy film and leaves the background alone", () => {
    const w = 200, h = 120, g = new Float32Array(w * h);
    let s = 1;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < g.length; i++) g[i] = 120 + (rnd() - 0.5) * 8;
    for (let y = 50; y < 60; y++) for (let x = 90; x < 100; x++) g[y * w + x] = 40;
    const out = runDemo(g, w, h, 6);
    const hits = out.defects.filter((d) => !d.suspect);
    expect(hits).toHaveLength(1);
    const [x, y] = hits[0].bbox;
    expect(x).toBeGreaterThanOrEqual(85); expect(x).toBeLessThanOrEqual(95);
    expect(y).toBeGreaterThanOrEqual(45); expect(y).toBeLessThanOrEqual(55);
    expect(out.labels[55 * w + 95]).toBeGreaterThan(0);
    expect(out.labels[10 * w + 10]).toBe(0);
  });
});
