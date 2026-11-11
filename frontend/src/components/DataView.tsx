import { useEffect, useMemo, useState } from "react";
import type { Inspector } from "../hooks/useInspector";
import { fetchSamples, sampleFile, sampleUrl, type SampleItem, type SampleList } from "../lib/apiEngine";
import { classColor, pretty } from "../lib/classes";

const SPLITS = ["train", "val", "test"] as const;
const SPLIT_NOTE: Record<string, string> = {
  train: "the model learnt from these",
  val: "used to pick the best epoch",
  test: "never seen during training",
};

const labelOf = (it: SampleItem) => (it.labels.length === 0 ? "none" : it.labels.length > 1 ? "multi" : it.labels[0]);
const labelName = (l: string) => (l === "none" ? "No defect" : l === "multi" ? "Several" : pretty(l));
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
      <div className="demo-banner err"><b>Dataset browser unavailable.</b> {error}. Start the server with a run and its prepared data:{" "}
        <code>uv run weldsight-serve --run runs/&lt;run&gt; --static frontend/dist</code></div>
    );
  }
  if (!data) return <p className="caption">Loading the dataset…</p>;

  const counts = data.counts[split] ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return (
    <>
      <div className="results-head">
        <p className="kicker">Dataset · {data.dataset} · data version <code>{data.data_version}</code> · {data.supervision === "label" ? "patch labels, no masks" : "masks"}</p>
        <h2>The patches the model was trained and tested on.</h2>
        <p className="lede">Split by weld specimen, so no specimen appears in two splits. Click a patch to run the model on it and compare with its true label.</p>
      </div>

      <div className="split-tabs" role="tablist">
        {SPLITS.map((sp) => {
          const n = Object.values(data.counts[sp] ?? {}).reduce((a, b) => a + b, 0);
          return (
            <button key={sp} type="button" role="tab" aria-selected={split === sp} onClick={() => setSplit(sp)}>
              <b>{sp}</b><span>{n.toLocaleString()} patches · {data.specimens[sp]?.length ?? 0} specimens</span><small>{SPLIT_NOTE[sp]}</small>
            </button>
          );
        })}
      </div>

      <div className="label-filter">
        <button type="button" aria-pressed={label === "all"} onClick={() => setLabel("all")}>All <em>{total.toLocaleString()}</em></button>
        {labels.filter((l) => counts[l]).map((l) => (
          <button key={l} type="button" aria-pressed={label === l} onClick={() => setLabel(l)}>
            <i style={{ background: labelColor(l) }} />{labelName(l)} <em>{(counts[l] ?? 0).toLocaleString()}</em>
          </button>
        ))}
        <span className="grow" />
        <button type="button" className="btn ghost" onClick={() => setSeed((s) => s + 1)}>Shuffle</button>
      </div>

      <p className="caption">Specimens in {split}: {data.specimens[split]?.join(", ")}</p>

      <div className="gallery">
        {items.map((it) => {
          const l = labelOf(it);
          return (
            <button key={it.id} type="button" className="thumb" onClick={() => void open(it)} disabled={opening !== null} title={it.id}>
              <img src={sampleUrl(settings.apiUrl, it.id)} alt={`${labelName(l)} patch from specimen ${it.specimen}`} loading="lazy" />
              <span className="tag" style={{ borderColor: labelColor(l), color: labelColor(l) }}>{it.labels.length ? it.labels.map(pretty).join(" + ") : "No defect"}</span>
              <span className="sid">{opening === it.id ? "analysing…" : it.specimen}</span>
            </button>
          );
        })}
      </div>
    </>
  );
}
