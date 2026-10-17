import { CLASSES, classColor, pretty } from "../lib/classes";
import type { Inspector } from "../hooks/useInspector";
import type { Decision, Settings } from "../lib/types";

export function DecisionCard({ decision, thr }: { decision: Decision; thr: number }) {
  const has = "score" in decision;
  return (
    <div className="verdict" data-state={decision.state}>
      <div className="verdict-label">Decision</div>
      <div className="verdict-main">{decision.title}</div>
      <div className="verdict-sub">{decision.sub}</div>
      {has && (
        <>
          <div className="meter">
            <div className="meter-fill" style={{ width: `${Math.min(100, decision.score * 100)}%` }} />
            <div className="meter-thr" style={{ left: `${thr * 100}%` }} />
          </div>
          <div className="meter-cap"><span>film uncertainty <b>{decision.score.toFixed(2)}</b></span><span>threshold <b>{thr.toFixed(2)}</b></span></div>
        </>
      )}
    </div>
  );
}

export function SettingsGroup({ insp }: { insp: Inspector }) {
  const { settings: s, patch } = insp;
  return (
    <details className="group" open>
      <summary>Settings</summary>
      <div className="fields">
        <label className="field"><span>Engine</span>
          <select value={s.engine} onChange={(e) => patch({ engine: e.target.value as Settings["engine"] })}>
            <option value="demo">Demo (browser, simulated)</option>
            <option value="api">WeldSight API</option>
          </select>
        </label>
        {s.engine === "api" && (
          <label className="field"><span>API URL</span>
            <input type="text" value={s.apiUrl} placeholder="same origin (dev: proxied to :8000)" spellCheck={false} onChange={(e) => patch({ apiUrl: e.target.value })} />
          </label>
        )}
        <label className="field"><span>Uncertainty score</span>
          <select value={s.score} onChange={(e) => patch({ score: e.target.value as Settings["score"] })}>
            <option value="pe">Predictive entropy</option>
            <option value="mi">Mutual information (BALD)</option>
          </select>
        </label>
        <label className="field range"><span>MC Dropout passes <b>{s.mc}</b></span>
          <input type="range" min={2} max={30} step={1} value={s.mc} onChange={(e) => patch({ mc: +e.target.value })} />
        </label>
        <label className="field range"><span>Refer to inspector above <b>{s.thr.toFixed(2)}</b></span>
          <input type="range" min={0.05} max={0.95} step={0.01} value={s.thr} onChange={(e) => patch({ thr: +e.target.value })} />
        </label>
        <p className="note">On the RIAWELC baseline, predictive entropy flagged wrong patches (AUROC 0.82) far better than mutual information (0.54). See the results tab.</p>
      </div>
    </details>
  );
}

export function FindingsGroup({ insp }: { insp: Inspector }) {
  const { result, settings: s, active, setActive } = insp;
  const defects = result?.defects ?? [];
  const order = defects.map((_, i) => i).sort((a, b) => defects[b].unc[s.score] - defects[a].unc[s.score]);
  return (
    <details className="group" open>
      <summary>Findings <em>{defects.length}</em></summary>
      <ul className="defects">
        {!defects.length && <li className="none">{result ? "No candidate regions." : "Nothing analysed yet."}</li>}
        {order.map((i) => {
          const d = defects[i], col = classColor(d.cls), u = d.unc[s.score];
          return (
            <li key={i}>
              <div
                className={`defect${d.suspect ? " suspect" : ""}${u >= s.thr ? " hot" : ""}${i === active ? " active" : ""}`}
                role="button" tabIndex={0} style={{ color: col }}
                onClick={() => setActive(active === i ? -1 : i)}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setActive(active === i ? -1 : i); } }}
              >
                <span className="sw" style={{ background: col }} />
                <span className="nm">{pretty(d.cls)}{d.suspect && <em> · suspect</em>}</span>
                <span className="pr">{(d.prob * 100).toFixed(0)}%</span>
                <span className="ubar"><i style={{ width: `${Math.min(100, u * 100)}%` }} /><s style={{ left: `${s.thr * 100}%` }} /></span>
                <span className="meta">u {u.toFixed(2)} · {d.bbox[2]}×{d.bbox[3]}px @ {d.bbox[0]},{d.bbox[1]}</span>
              </div>
            </li>
          );
        })}
      </ul>
    </details>
  );
}

export function LegendGroup({ insp }: { insp: Inspector }) {
  const defects = insp.result?.defects ?? [];
  const ids = insp.result?.classes ?? CLASSES.map((c) => c.id);
  const count = (id: string) => defects.filter((d) => !d.suspect && d.cls === id).length;
  return (
    <details className="group">
      <summary>Classes</summary>
      <ul className="legend">
        {ids.map((id) => <li key={id}><i style={{ background: classColor(id) }} /><span>{pretty(id)}</span><span>{count(id)}</span></li>)}
      </ul>
    </details>
  );
}
