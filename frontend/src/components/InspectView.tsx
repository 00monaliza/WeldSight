import type { Inspector } from "../hooks/useInspector";
import { savePng, saveJson } from "../lib/report";
import { DecisionCard, FindingsGroup, LegendGroup, SettingsGroup } from "./Panel";
import { Viewer } from "./Viewer";

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="toggle"><input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} /><span>{label}</span></label>
  );
}

function Banner({ insp }: { insp: Inspector }) {
  const { settings, health } = insp;
  if (settings.engine === "demo") {
    return (
      <div className="demo-banner"><b>Demo engine.</b> A simple blur-difference detector runs in your browser to exercise the interface.
        It is <u>not</u> the trained WeldSight model, and its detections mean nothing about real weld quality. Switch to the API engine once a model is served.</div>
    );
  }
  if (health && !health.ok) {
    return <div className="demo-banner err"><b>API unreachable.</b> {health.error}. Start the server: <code>uv run weldsight-serve --run runs/&lt;run&gt;</code></div>;
  }
  if (health?.ok && health.info.stub) {
    return <div className="demo-banner"><b>Server is running the stub predictor.</b> It is a heuristic, not a trained model; every region is labelled as the first class. Start it with <code>--run</code> to use a checkpoint.</div>;
  }
  return null;
}

export function InspectView({ insp }: { insp: Inspector }) {
  const { settings: s, patch, film, result, decision, loadFile } = insp;
  return (
    <>
      <Banner insp={insp} />
      <div className="inspect-grid">
        <div className="stage-col">
          <Viewer insp={insp} />
          <div className="toolbar">
            <Toggle label="Defects" checked={s.showSeg} onChange={(v) => patch({ showSeg: v })} />
            <Toggle label="Uncertainty" checked={s.showUnc} onChange={(v) => patch({ showUnc: v })} />
            <Toggle label="Boxes" checked={s.showBox} onChange={(v) => patch({ showBox: v })} />
            <label className="slider small"><span>Opacity</span>
              <input type="range" min={0.15} max={1} step={0.05} value={s.opacity} onChange={(e) => patch({ opacity: +e.target.value })} />
            </label>
            <span className="grow" />
            <button className="btn ghost" type="button" disabled={!film || !result} onClick={() => film && result && void savePng(film, result, s)}>Save overlay PNG</button>
            <button className="btn ghost" type="button" disabled={!film || !result} onClick={() => film && result && saveJson(film, result, s)}>Save report JSON</button>
            <label className="btn ghost">Open another
              <input type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void loadFile(f); e.target.value = ""; }} />
            </label>
          </div>
        </div>
        <aside className="panel">
          <DecisionCard decision={decision} thr={s.thr} />
          <SettingsGroup insp={insp} />
          <FindingsGroup insp={insp} />
          <LegendGroup insp={insp} />
        </aside>
      </div>
    </>
  );
}
