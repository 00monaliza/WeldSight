import { afterEach, describe, expect, it, vi } from "vitest";
import { runApi } from "./apiEngine";
import type { Film } from "./types";

const film = { name: "f.png", w: 64, h: 32, file: new File([new Uint8Array([1])], "f.png") } as unknown as Film;
const ok = (body: unknown) => vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));

afterEach(() => vi.unstubAllGlobals());

describe("runApi response validation", () => {
  it("refuses a response without uncertainty maps instead of reading it as certain", async () => {
    ok({ width: 64, height: 32, classes: ["crack"], mask_png: "AA==", defects: [] });
    await expect(runApi(film, "", 5)).rejects.toThrow(/missing/);
  });
  it("refuses a region without uncertainty values", async () => {
    ok({ width: 64, height: 32, classes: ["crack"], mask_png: "AA==", maps: { pe: "AA==", mi: "AA==" }, defects: [{ cls: "crack", prob: 0.9, bbox: [0, 0, 1, 1] }] });
    await expect(runApi(film, "", 5)).rejects.toThrow(/uncertainty/);
  });
  it("surfaces the server's error detail", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "image has 80 MP, limit is 50 MP" }), { status: 413 })));
    await expect(runApi(film, "", 5)).rejects.toThrow(/413.*80 MP/);
  });
});
