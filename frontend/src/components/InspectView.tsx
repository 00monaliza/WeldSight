import type { Inspector } from "../hooks/useInspector";
import { savePng, saveJson } from "../lib/report";
import { DecisionCard, FindingsGroup, LegendGroup, PatchCard, SettingsGroup } from "./Panel";
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
      <div className="demo-banner"><b>Демо-движок.</b> В браузере работает простой детектор на разности размытий — только чтобы проверить интерфейс.
        Это <u>не</u> обученная модель WeldSight, и его находки ничего не говорят о качестве шва. Переключитесь на API, когда сервер с моделью запущен.</div>
    );
  }
  if (health && !health.ok) {
    return <div className="demo-banner err"><b>API недоступен.</b> {health.error}. Запустите сервер: <code>uv run weldsight-serve --run models/riawelc_cpu_v1 --static frontend/dist</code></div>;
  }
  if (health?.ok && health.info.mode === "patch") {
    return (
      <div className="demo-banner"><b>Классификатор патчей · {health.info.model}.</b> Модель обучена на RIAWELC: там есть метки классов, но нет масок,
        поэтому работает только голова классификации — одно предсказание на тайл 256 px, а не контур дефекта. Откройте патч во вкладке <a href="#data">Датасет</a>, чтобы сравнить с истинной меткой.</div>
    );
  }
  if (health?.ok && health.info.stub) {
    return <div className="demo-banner"><b>Сервер запущен с заглушкой.</b> Это эвристика, а не обученная модель; все области получают первый класс. Запустите с <code>--run</code>, чтобы использовать чекпоинт.</div>;
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
            <Toggle label="Дефекты" checked={s.showSeg} onChange={(v) => patch({ showSeg: v })} />
            <Toggle label="Неуверенность" checked={s.showUnc} onChange={(v) => patch({ showUnc: v })} />
            <Toggle label="Рамки" checked={s.showBox} onChange={(v) => patch({ showBox: v })} />
            <label className="slider small"><span>Прозрачность</span>
              <input type="range" min={0.15} max={1} step={0.05} value={s.opacity} onChange={(e) => patch({ opacity: +e.target.value })} />
            </label>
            <span className="grow" />
            <button className="btn ghost" type="button" disabled={!film || !result} onClick={() => film && result && void savePng(film, result, s)}>Сохранить PNG</button>
            <button className="btn ghost" type="button" disabled={!film || !result} onClick={() => film && result && saveJson(film, result, s)}>Сохранить отчёт JSON</button>
            <label className="btn ghost">Открыть другой
              <input type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void loadFile(f); e.target.value = ""; }} />
            </label>
          </div>
        </div>
        <aside className="panel">
          <DecisionCard decision={decision} thr={s.thr} />
          <PatchCard insp={insp} />
          <SettingsGroup insp={insp} />
          <FindingsGroup insp={insp} />
          <LegendGroup insp={insp} />
        </aside>
      </div>
    </>
  );
}
