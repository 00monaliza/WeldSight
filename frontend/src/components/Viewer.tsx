import { useEffect, useRef, useState, type DragEvent } from "react";
import { pretty } from "../lib/classes";
import { drawBoxes, drawOverlay } from "../lib/render";
import type { Inspector } from "../hooks/useInspector";

export function Viewer({ insp }: { insp: Inspector }) {
  const { film, result, settings, active, busy, loadFile, loadSample } = insp;
  const base = useRef<HTMLCanvasElement>(null), over = useRef<HTMLCanvasElement>(null), hi = useRef<HTMLCanvasElement>(null);
  const [drag, setDrag] = useState(false);
  const [hover, setHover] = useState<{ x: number; y: number; cls: string; u: number } | null>(null);

  useEffect(() => {
    if (!film || !base.current || !over.current || !hi.current) return;
    for (const c of [base.current, over.current, hi.current]) { c.width = film.w; c.height = film.h; }
    base.current.getContext("2d")!.drawImage(film.base, 0, 0);
  }, [film]);

  useEffect(() => {
    const c = over.current;
    if (!film || !c) return;
    const g = c.getContext("2d")!;
    if (!result) { g.clearRect(0, 0, film.w, film.h); return; }
    drawOverlay(g, film.w, film.h, result, settings.score, { seg: settings.showSeg, unc: settings.showUnc, opacity: settings.opacity });
  }, [film, result, settings.score, settings.showSeg, settings.showUnc, settings.opacity]);

  useEffect(() => {
    const c = hi.current;
    if (!film || !c) return;
    drawBoxes(c.getContext("2d")!, film.w, film.h, settings.showBox ? result : null, active);
  }, [film, result, active, settings.showBox]);

  const onMove = (e: React.MouseEvent) => {
    if (!film || !result || !base.current) return;
    const r = base.current.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) / r.width) * film.w), y = Math.floor(((e.clientY - r.top) / r.height) * film.h);
    if (x < 0 || y < 0 || x >= film.w || y >= film.h) return;
    const i = y * film.w + x, c = result.labels[i];
    setHover({ x, y, cls: c ? pretty(result.classes[c - 1] ?? "?") : "bg", u: result.maps[settings.score][i] });
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f) void loadFile(f);
  };

  return (
    <>
      <div
        className={`film${film ? " loaded" : ""}${drag ? " over" : ""}`}
        onDragEnter={(e) => { e.preventDefault(); setDrag(true); }}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={(e) => { e.preventDefault(); setDrag(false); }}
        onDrop={onDrop}
      >
        <span className="tick tl" /><span className="tick tr" /><span className="tick bl" /><span className="tick br" />
        {film ? (
          <div className="canvas-wrap" style={{ aspectRatio: `${film.w} / ${film.h}` }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
            <canvas ref={base} /><canvas ref={over} /><canvas ref={hi} />
            {busy && <div className="scan" />}
          </div>
        ) : (
          <div className="empty">
            <div className="empty-inner">
              <h1>Radiograph<br />in, decision out.</h1>
              <p>Drop a weld radiograph here. WeldSight segments defects, estimates how sure it is, and tells you which films a human should look at.</p>
              <div className="empty-actions">
                <label className="btn primary">Choose image
                  <input type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && void loadFile(e.target.files[0])} />
                </label>
                <button className="btn" type="button" onClick={loadSample}>Generate synthetic sample</button>
              </div>
              <small>{settings.engine === "demo" ? "Images stay in your browser in demo mode." : "In API mode the image is sent to the server you configured."}</small>
            </div>
          </div>
        )}
        <div className="drop-hint">Release to inspect</div>
      </div>

      <div className="readout">
        <span>x <b>{hover?.x ?? "—"}</b></span><span>y <b>{hover?.y ?? "—"}</b></span>
        <span>class <b>{hover?.cls ?? "—"}</b></span><span>uncertainty <b>{hover ? hover.u.toFixed(2) : "—"}</b></span>
        <span className="grow" />
        <span>{film ? `${film.w}×${film.h}px · ${film.name}` : "—"}</span>
      </div>
    </>
  );
}
