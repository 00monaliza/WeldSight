import type { Film } from "./types";

export const MAX_DIM = 1024;

/** Same rule as weldsight.serve.postprocess.output_size */
export function workingSize(sw: number, sh: number, maxDim = MAX_DIM): [number, number] {
  const k = Math.min(1, maxDim / Math.max(sw, sh));
  return [Math.max(32, Math.round(sw * k)), Math.max(32, Math.round(sh * k))];
}

export function filmFromSource(src: CanvasImageSource & { width: number; height: number }, name: string, file?: File): Film {
  const [w, h] = workingSize(src.width, src.height);
  const base = document.createElement("canvas");
  base.width = w;
  base.height = h;
  const g = base.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(src, 0, 0, w, h);
  // radiographs are single channel; grey keeps colour free for the overlay
  const img = g.getImageData(0, 0, w, h);
  const d = img.data;
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const v = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    gray[i] = v;
    d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v;
    d[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return { name, w, h, base, gray, file };
}

export async function filmFromFile(file: File): Promise<Film> {
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(file);
  } catch {
    throw new Error("Cannot read this file in the browser. Use PNG or JPEG.");
  }
  const film = filmFromSource(bmp, file.name, file);
  bmp.close();
  return film;
}

export function canvasToBlob(c: HTMLCanvasElement, type = "image/png"): Promise<Blob> {
  return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), type));
}

export function download(name: string, blob: Blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
