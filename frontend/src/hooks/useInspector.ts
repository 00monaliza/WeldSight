import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchHealth, runApi, type Health } from "../lib/apiEngine";
import { decide } from "../lib/decision";
import { runDemo } from "../lib/demoEngine";
import { filmFromFile, filmFromSource } from "../lib/imaging";
import { makeSample } from "../lib/sample";
import type { AnalysisResult, Decision, Film, Settings } from "../lib/types";

export const DEFAULTS: Settings = {
  engine: "demo", apiUrl: "", score: "pe", mc: 10, thr: 0.55, opacity: 0.6, showSeg: true, showUnc: false, showBox: true,
};

const nextFrame = () => new Promise<void>((r) => setTimeout(r, 40));

export function useInspector() {
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [film, setFilm] = useState<Film | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(-1);
  const [health, setHealth] = useState<{ ok: true; info: Health } | { ok: false; error: string } | null>(null);
  const filmRef = useRef<Film | null>(null);
  const loadSeq = useRef(0);

  const patch = useCallback((p: Partial<Settings>) => setSettings((s) => ({ ...s, ...p })), []);

  const setNewFilm = useCallback((f: Film) => {
    setFilm(f); setResult(null); setError(null); setActive(-1);
  }, []);
  const loadFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;
    const seq = ++loadSeq.current;  // a slower, earlier load must not replace a later selection
    try {
      const f = await filmFromFile(file);
      if (seq === loadSeq.current) setNewFilm(f);
    } catch (e) {
      if (seq === loadSeq.current) setError((e as Error).message);
    }
  }, [setNewFilm]);
  const loadSample = useCallback(() => {
    loadSeq.current++;
    setNewFilm(filmFromSource(makeSample(), "synthetic_sample.png"));
  }, [setNewFilm]);

  // (re)analyse when the film, the engine or the number of MC passes changes
  const { engine, mc, apiUrl } = settings;
  useEffect(() => {
    if (!film) return;
    const fresh = filmRef.current !== film;
    filmRef.current = film;
    const ac = new AbortController();
    let alive = true;
    const t = setTimeout(async () => {
      setBusy(true); setError(null);
      try {
        let res: AnalysisResult;
        if (engine === "api") res = await runApi(film, apiUrl, mc, ac.signal);
        else { await nextFrame(); res = runDemo(film.gray, film.w, film.h, mc); }
        if (alive) { setResult({ ...res, engine, mc }); setActive(-1); }
      } catch (e) {
        if (alive && (e as Error).name !== "AbortError") { setResult(null); setError((e as Error).message); }
      } finally {
        if (alive) setBusy(false);
      }
    }, fresh ? 0 : 350);
    return () => { alive = false; clearTimeout(t); ac.abort(); };
  }, [film, engine, mc, apiUrl]);

  // server status for the header chip and the banner
  useEffect(() => {
    setHealth(null);  // never show the previous endpoint's status while the new one is unknown
    if (engine !== "api") return;
    const ac = new AbortController();
    const t = setTimeout(() => {
      fetchHealth(apiUrl, ac.signal).then((info) => setHealth({ ok: true, info }), (e) => { if (e.name !== "AbortError") setHealth({ ok: false, error: String(e.message ?? e) }); });
    }, 300);
    return () => { clearTimeout(t); ac.abort(); };
  }, [engine, apiUrl]);

  const decision: Decision = useMemo(() => {
    if (busy) return { state: "idle", title: "Analysing…", sub: engine === "api" ? "Waiting for the API" : `Running ${mc} stochastic passes` };
    if (error) return { state: "idle", title: "Engine error", sub: error };
    if (!film || !result) return { state: "idle", title: "Waiting for a film", sub: "Load an image to get a recommendation." };
    return decide(result, settings.score, settings.thr);
  }, [busy, error, film, result, engine, mc, settings.score, settings.thr]);

  return { settings, patch, film, result, busy, error, active, setActive, health, decision, loadFile, loadSample };
}

export type Inspector = ReturnType<typeof useInspector>;
