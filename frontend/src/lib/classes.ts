export interface ClassMeta {
  id: string;
  color: string;
}

/** SWRD classes, same order as configs/data/swrd.yaml */
export const CLASSES: ClassMeta[] = [
  { id: "porosity", color: "#3dd9ff" },
  { id: "inclusion", color: "#ff5fd2" },
  { id: "crack", color: "#ff4d4d" },
  { id: "undercut", color: "#ffe14d" },
  { id: "lack_of_fusion", color: "#ff9a3d" },
  { id: "lack_of_penetration", color: "#6dff8a" },
];

const FALLBACK = ["#8fb8ff", "#c9a0ff", "#ffd1a0", "#a0ffe0", "#ff9ec1", "#d4ff7a"];

export function classColor(id: string): string {
  const k = CLASSES.findIndex((c) => c.id === id);
  if (k >= 0) return CLASSES[k].color;
  let h = 0;
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return FALLBACK[h % FALLBACK.length];
}

export const hexToRgb = (h: string): [number, number, number] => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];

export function pretty(s: string): string {
  const t = s.replace(/_/g, " ");
  return t.charAt(0).toUpperCase() + t.slice(1);
}
