export type ScoreKind = "pe" | "mi";
export type EngineKind = "demo" | "api";

export interface Defect {
  cls: string;
  prob: number;
  /** x, y, w, h in film pixels */
  bbox: [number, number, number, number];
  /** below the detection threshold: shown dashed, still counts for referral */
  suspect: boolean;
  unc: Record<ScoreKind, number>;
}

export interface AnalysisResult {
  /** class names; `labels` holds 0 = background, k = classes[k-1] */
  classes: string[];
  labels: Uint8Array;
  maps: Record<ScoreKind, Float32Array>;
  defects: Defect[];
  model?: string;
  stub?: boolean;
}

export interface Film {
  name: string;
  w: number;
  h: number;
  /** grayscale canvas shown in the viewer */
  base: HTMLCanvasElement;
  gray: Float32Array;
  /** original upload, sent as-is to the API so the server sees native resolution */
  file?: File;
}

export interface Settings {
  engine: EngineKind;
  apiUrl: string;
  score: ScoreKind;
  mc: number;
  thr: number;
  opacity: number;
  showSeg: boolean;
  showUnc: boolean;
  showBox: boolean;
}

export type Decision =
  | { state: "idle"; title: string; sub: string }
  | { state: "refer" | "flag" | "clear"; title: string; sub: string; score: number };
