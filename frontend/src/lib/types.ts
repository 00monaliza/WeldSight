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

/** Patch-classification mode (label-only runs, e.g. RIAWELC): one prediction per tile. */
export interface Tile {
  /** x, y, w, h in film pixels */
  bbox: [number, number, number, number];
  probs: Record<string, number>;
  /** std of the class probability over the MC Dropout passes */
  std: Record<string, number>;
  pe: number;
  mi: number;
}

/** Ground truth of a patch opened from the dataset browser */
export interface Truth {
  id: string;
  split: string;
  specimen: string;
  labels: string[];
}

export interface AnalysisResult {
  /** class names; `labels` holds 0 = background, k = classes[k-1] */
  classes: string[];
  labels: Uint8Array;
  maps: Record<ScoreKind, Float32Array>;
  defects: Defect[];
  model?: string;
  stub?: boolean;
  /** present only in patch mode */
  tiles?: Tile[];
  /** settings this result was computed with (exports must not mix them with newer settings) */
  engine?: EngineKind;
  mc?: number;
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
  /** set when the film comes from the dataset browser */
  truth?: Truth;
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
