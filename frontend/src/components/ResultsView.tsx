const ROWS = [
  { n: "Crack", rec: 57.0, lo: 52.6, hi: 66.7, sup: "977 patches · 2 specimens", p: 67.0, f1: 0.62, auc: 0.86 },
  { n: "Porosity", rec: 93.1, lo: 86.8, hi: 97.2, sup: "936 patches · 4 specimens", p: 93.5, f1: 0.93, auc: 0.99 },
  { n: "Lack of penetration", rec: 19.0, lo: 2.3, hi: 42.8, sup: "658 patches · 3 specimens", p: 64.4, f1: 0.29, auc: 0.72 },
];
const AUROC: [string, number][] = [["Predictive entropy", 0.82], ["Expected entropy", 0.82], ["Max-prob", 0.8], ["Mutual information", 0.54]];
const FINDINGS = [
  ["No dataset has everything.", "RIAWELC has no masks, GDXray has 10 films and one class. SWRD (6 classes, polygons) is the main dataset from here."],
  ["The published split leaks.", "108 of 109 radiographs sit in both train and test. Re-split by weld specimen (29 specimens)."],
  ["A flatness shortcut.", "Patch std alone separates defect from no-defect with AUROC 0.91."],
  ["Tiny effective sample.", "Each test class occurs in only 2–4 specimens, so every recall carries a cluster-bootstrap CI."],
];

// Only the four points published in the README
const X = [0, 10, 20, 30];
const SERIES = {
  model: [60.4, 65.4, 71.2, 76.8],
  random: [60.4, 64.1, 67.9, 71.5],
  oracle: [60.4, 69.0, 75.9, 81.5],
};

function ReferralChart() {
  const L = 44, R = 462, T = 28, B = 256, y0 = 55, y1 = 85;
  const sx = (v: number) => L + (v / 30) * (R - L), sy = (v: number) => B - ((v - y0) / (y1 - y0)) * (B - T);
  const pts = (a: number[]) => X.map((x, i) => `${sx(x)},${sy(a[i])}`).join(" ");
  return (
    <svg viewBox="0 0 520 300" role="img" aria-label="Referral curve: model, random and oracle recall at 0, 10, 20 and 30 percent referral" style={{ width: "100%", height: "auto", display: "block" }}>
      {[55, 60, 65, 70, 75, 80, 85].map((v) => (
        <g key={v}><line x1={L} x2={R} y1={sy(v)} y2={sy(v)} stroke="#23292c" /><text x={L - 8} y={sy(v) + 3} textAnchor="end">{v}</text></g>
      ))}
      {X.map((x) => <text key={x} x={sx(x)} y={B + 18} textAnchor="middle">{x} %</text>)}
      <text x={(L + R) / 2} y={B + 36} textAnchor="middle">films referred to an inspector</text>
      <text x={L - 8} y={T - 14} textAnchor="end">recall %</text>
      <polyline fill="none" stroke="#6b7270" strokeWidth={1.5} strokeDasharray="5 4" points={pts(SERIES.random)} />
      <polyline fill="none" stroke="#e9e7e0" strokeWidth={1.5} strokeDasharray="2 4" points={pts(SERIES.oracle)} />
      <polyline fill="none" stroke="#ffb347" strokeWidth={2.5} points={pts(SERIES.model)} />
      {SERIES.model.map((v, i) => <rect key={i} x={sx(X[i]) - 4} y={sy(v) - 4} width={8} height={8} fill="#ffb347" transform={`rotate(45 ${sx(X[i])} ${sy(v)})`} />)}
      <text x={R + 10} y={sy(76.8) + 3} style={{ fill: "#ffb347" }}>76.8</text>
      <text x={R + 10} y={sy(71.5) + 3}>71.5</text>
      <text x={R + 10} y={sy(81.5) + 3} style={{ fill: "#e9e7e0" }}>81.5</text>
    </svg>
  );
}

export function ResultsView() {
  return (
    <>
      <div className="results-head">
        <p className="kicker">Stage 1 · RIAWELC specimen-level split · run <code>riawelc_cpu</code></p>
        <h2>A pipeline check, not a verdict on the method.</h2>
        <p className="lede">An offline CPU run: ResNet-18 <em>without</em> ImageNet weights, 8,000 patches per epoch, patch head only (RIAWELC has no masks, so there is no IoU).
          Test split: 3,073 patches, 21 radiographs, 6 weld specimens. Numbers below are copied from the repository README.</p>
      </div>

      <div className="cards4">
        {FINDINGS.map(([b, t], i) => <article key={i}><h3>{String(i + 1).padStart(2, "0")}</h3><p><b>{b}</b> {t}</p></article>)}
      </div>

      <div className="split">
        <section className="block">
          <h3>Per-class recall, test split</h3>
          <div>
            {ROWS.map((r) => (
              <div className="ci-row" key={r.n}>
                <div className="nm">{r.n}<small>{r.sup}</small></div>
                <div className="ci-track"><span className="ci-bar" style={{ left: `${r.lo}%`, width: `${r.hi - r.lo}%` }} /><span className="ci-dot" style={{ left: `${r.rec}%` }} /></div>
                <div className="ci-val">{r.rec.toFixed(1)}</div>
                <div className="ci-extra"><span>CI <b>{r.lo}–{r.hi}</b></span><span>precision <b>{r.p}</b></span><span>F1 <b>{r.f1}</b></span><span>AUROC <b>{r.auc}</b></span></div>
              </div>
            ))}
            <div className="ci-scale" style={{ marginTop: 6 }}><span /><div><span>0 %</span><span>50</span><span>100 %</span></div><span /></div>
          </div>
          <p className="caption">Dot = recall, bar = 95 % cluster bootstrap CI over radiographs. Validation macro-F1 was 0.89 against 0.61 on test: specimen-to-specimen variance, not noise a single split can resolve.</p>
        </section>

        <section className="block">
          <h3>Recall vs. share of films sent to an inspector</h3>
          <ReferralChart />
          <div className="chart-legend">
            <span><i className="l-model" />model (mutual information)</span><span><i className="l-rand" />random</span><span><i className="l-orc" />oracle</span>
          </div>
          <p className="caption">Upper bound: the inspector is assumed to find every defect on referred films. AUC 0.832 model, 0.801 random, 0.875 oracle.</p>
        </section>
      </div>

      <section className="block wide">
        <h3>Which uncertainty score flags wrong patches?</h3>
        <div className="bars">
          {AUROC.map(([n, v]) => (
            <div className={`bar${n.startsWith("Mutual") ? " mi" : ""}`} key={n}>
              <span>{n}</span><div className="track"><i style={{ width: `${v * 100}%` }} /><s /></div><span>{v.toFixed(2)}</span>
            </div>
          ))}
        </div>
        <p className="caption">AUROC of uncertainty vs. “patch has an error”. The pre-registered score (mutual information) is the weakest. Most errors are consistent uncertainty (aleatoric),
          not dropout disagreement. Changing the score based on this test observation would be test-set tuning; pick it on validation.</p>
      </section>
    </>
  );
}
