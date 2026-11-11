import { useEffect, useMemo, useState } from "react";
import type { Inspector } from "../hooks/useInspector";
import { fetchSamples, sampleFile, sampleUrl, type SampleItem, type SampleList } from "../lib/apiEngine";
import { classColor, pretty } from "../lib/classes";

const SPLITS = ["train", "val", "test"] as const;
const SPLIT_NOTE: Record<string, string> = {
  train: "на них модель обучалась",
  val: "по ним выбрана лучшая эпоха",
  test: "модель их не видела",
};

const labelOf = (it: SampleItem) => (it.labels.length === 0 ? "none" : it.labels.length > 1 ? "multi" : it.labels[0]);
const labelName = (l: string) => (l === "none" ? "Без дефекта" : l === "multi" ? "Несколько" : pretty(l));
const labelColor = (l: string) => (l === "none" ? "#6b7270" : l === "multi" ? "#e9e7e0" : classColor(l));

/** Browse the prepared dataset served by the API and open any patch in the inspector. */
export function DataView({ insp, visible }: { insp: Inspector; visible: boolean }) {
  const { settings, patch, loadFile } = insp;
  const [data, setData] = useState<SampleList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [split, setSplit] = useState<string>("train");
  const [label, setLabel] = useState<string>("all");
  const [seed, setSeed] = useState(0);
  const [opening, setOpening] = useState<string | null>(null);

  useEffect(() => {
    if (!visible) return;
    const ac = new AbortController();
    setError(null);
    fetchSamples(settings.apiUrl, 16, seed, ac.signal).then(setData, (e) => {
      if (e.name !== "AbortError") setError(String(e.message ?? e));
    });
    return () => ac.abort();
  }, [visible, settings.apiUrl, seed]);

  const items = useMemo(
    () => (data?.items ?? []).filter((it) => it.split === split && (label === "all" || labelOf(it) === label)),
    [data, split, label],
  );
  const labels = useMemo(() => ["none", ...(data?.classes ?? []), "multi"], [data]);

  const open = async (it: SampleItem) => {
    setOpening(it.id);
    try {
      const file = await sampleFile(settings.apiUrl, it.id);
      patch({ engine: "api" });
      await loadFile(file, { id: it.id, split: it.split, specimen: it.specimen, labels: it.labels });
      location.hash = "#inspect";
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setOpening(null);
    }
  };

  if (error) {
    return (
      <div className="demo-banner err"><b>Датасет недоступен.</b> {error}. Подготовьте данные:{" "}
        <code>uv run weldsight-prepare --config configs/experiment/riawelc_baseline.yaml --download</code>, затем перезапустите <code>uv run weldsight-serve --run models/riawelc_cpu_v1 --static frontend/dist</code></div>
    );
  }
  if (!data) return <p className="caption">Загрузка датасета…</p>;

  const counts = data.counts[split] ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return (
    <>
      <div className="results-head">
        <p className="kicker">Датасет · {data.dataset} · версия данных <code>{data.data_version}</code> · {data.supervision === "label" ? "метки патчей, без масок" : "маски"}</p>
        <h2>На чём модель обучалась и проверялась.</h2>
        <p className="lede">Разбиение по образцам шва: ни один образец не попадает в два сплита. Нажмите на патч — модель его проанализирует, и результат можно сравнить с истинной меткой.</p>
      </div>

      <div className="split-tabs" role="tablist">
        {SPLITS.map((sp) => {
          const n = Object.values(data.counts[sp] ?? {}).reduce((a, b) => a + b, 0);
          return (
            <button key={sp} type="button" role="tab" aria-selected={split === sp} onClick={() => setSplit(sp)}>
              <b>{sp}</b><span>{n.toLocaleString("ru-RU")} патчей · {data.specimens[sp]?.length ?? 0} образцов</span><small>{SPLIT_NOTE[sp]}</small>
            </button>
          );
        })}
      </div>

      <div className="label-filter">
        <button type="button" aria-pressed={label === "all"} onClick={() => setLabel("all")}>Все <em>{total.toLocaleString("ru-RU")}</em></button>
        {labels.filter((l) => counts[l]).map((l) => (
          <button key={l} type="button" aria-pressed={label === l} onClick={() => setLabel(l)}>
            <i style={{ background: labelColor(l) }} />{labelName(l)} <em>{(counts[l] ?? 0).toLocaleString("ru-RU")}</em>
          </button>
        ))}
        <span className="grow" />
        <button type="button" className="btn ghost" onClick={() => setSeed((s) => s + 1)}>Перемешать</button>
      </div>

      <p className="caption">Образцы в {split}: {data.specimens[split]?.join(", ")}</p>

      <div className="gallery">
        {items.map((it) => {
          const l = labelOf(it);
          return (
            <button key={it.id} type="button" className="thumb" onClick={() => void open(it)} disabled={opening !== null} title={it.id}>
              <img src={sampleUrl(settings.apiUrl, it.id)} alt={`${labelName(l)}: патч образца ${it.specimen}`} loading="lazy" />
              <span className="tag" style={{ borderColor: labelColor(l), color: labelColor(l) }}>{it.labels.length ? it.labels.map(pretty).join(" + ") : "Без дефекта"}</span>
              <span className="sid">{opening === it.id ? "анализ…" : it.specimen}</span>
            </button>
          );
        })}
      </div>
    </>
  );
}
