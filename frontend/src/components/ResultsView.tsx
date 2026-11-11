const ROWS = [
  { n: "Трещина", rec: 57.0, lo: 52.6, hi: 66.7, sup: "977 патчей · 2 образца", p: 67.0, f1: 0.62, auc: 0.86 },
  { n: "Пористость", rec: 93.1, lo: 86.8, hi: 97.2, sup: "936 патчей · 4 образца", p: 93.5, f1: 0.93, auc: 0.99 },
  { n: "Непровар", rec: 19.0, lo: 2.3, hi: 42.8, sup: "658 патчей · 3 образца", p: 64.4, f1: 0.29, auc: 0.72 },
];
const AUROC: [string, number][] = [["Предсказательная энтропия", 0.82], ["Ожидаемая энтропия", 0.82], ["Max-prob", 0.8], ["Взаимная информация", 0.54]];
const SELECTION: [string, number, number][] = [
  ["Max-prob (выбрана на val)", 0.961, 0.837],
  ["Предсказательная энтропия", 0.953, 0.852],
  ["Ожидаемая энтропия", 0.953, 0.852],
  ["Взаимная информация (выбрана заранее)", 0.934, 0.832],
];
const FINDINGS = [
  ["Ни в одном датасете нет всего.", "В RIAWELC нет масок, в GDXray 10 снимков и один класс. Основной датасет дальше — SWRD (6 классов, полигоны)."],
  ["Официальный сплит протекает.", "108 из 109 снимков есть и в train, и в test. Разбили заново по образцам шва (29 образцов)."],
  ["Шорткат по гладкости.", "Одно СКО патча отделяет «дефект / нет дефекта» с AUROC 0.91."],
  ["Маленькая эффективная выборка.", "Каждый класс в test встречается всего в 2–4 образцах, поэтому у каждого recall есть bootstrap-интервал по снимкам."],
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
    <svg viewBox="0 0 520 300" role="img" aria-label="Кривая передачи: recall модели, случайной передачи и оракула при 0, 10, 20 и 30 % переданных снимков" style={{ width: "100%", height: "auto", display: "block" }}>
      {[55, 60, 65, 70, 75, 80, 85].map((v) => (
        <g key={v}><line x1={L} x2={R} y1={sy(v)} y2={sy(v)} stroke="#23292c" /><text x={L - 8} y={sy(v) + 3} textAnchor="end">{v}</text></g>
      ))}
      {X.map((x) => <text key={x} x={sx(x)} y={B + 18} textAnchor="middle">{x} %</text>)}
      <text x={(L + R) / 2} y={B + 36} textAnchor="middle">доля снимков, переданных инспектору</text>
      <text x={4} y={T - 14} textAnchor="start">recall, %</text>
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
        <p className="kicker">Этап 1 · RIAWELC, сплит по образцам шва · прогон <code>riawelc_cpu</code></p>
        <h2>Проверка пайплайна, а не приговор методу.</h2>
        <p className="lede">Прогон на CPU без интернета: ResNet-18 <em>без</em> весов ImageNet, 8 000 патчей за эпоху, только голова классификации (в RIAWELC нет масок, поэтому нет IoU).
          Test: 3 073 патча, 21 снимок, 6 образцов шва. Прогон повторён с нуля 28.10 и 11.11 — метрики совпали.</p>
      </div>

      <div className="cards4">
        {FINDINGS.map(([b, t], i) => <article key={i}><h3>{String(i + 1).padStart(2, "0")}</h3><p><b>{b}</b> {t}</p></article>)}
      </div>

      <div className="split">
        <section className="block">
          <h3>Recall по классам, test</h3>
          <div>
            {ROWS.map((r) => (
              <div className="ci-row" key={r.n}>
                <div className="nm">{r.n}<small>{r.sup}</small></div>
                <div className="ci-track"><span className="ci-bar" style={{ left: `${r.lo}%`, width: `${r.hi - r.lo}%` }} /><span className="ci-dot" style={{ left: `${r.rec}%` }} /></div>
                <div className="ci-val">{r.rec.toFixed(1)}</div>
                <div className="ci-extra"><span>ДИ <b>{r.lo}–{r.hi}</b></span><span>precision <b>{r.p}</b></span><span>F1 <b>{r.f1}</b></span><span>AUROC <b>{r.auc}</b></span></div>
              </div>
            ))}
            <div className="ci-scale" style={{ marginTop: 6 }}><span /><div><span>0 %</span><span>50</span><span>100 %</span></div><span /></div>
          </div>
          <p className="caption">Точка — recall, полоса — 95 % bootstrap-интервал по снимкам. Macro-F1 на val 0.89, на test 0.61: это разброс между образцами шва, и один сплит его не разрешит — нужна k-fold кросс-валидация.</p>
        </section>

        <section className="block">
          <h3>Recall в зависимости от доли снимков, переданных инспектору</h3>
          <ReferralChart />
          <div className="chart-legend">
            <span><i className="l-model" />модель (взаимная информация)</span><span><i className="l-rand" />случайно</span><span><i className="l-orc" />оракул</span>
          </div>
          <p className="caption">Верхняя оценка: считаем, что инспектор находит все дефекты на переданных снимках. Площадь: модель 0.832, случайно 0.801, оракул 0.875.</p>
        </section>
      </div>

      <section className="block wide">
        <h3>Какая мера неуверенности находит ошибки?</h3>
        <div className="bars">
          {AUROC.map(([n, v]) => (
            <div className={`bar${n.startsWith("Mutual") ? " mi" : ""}`} key={n}>
              <span>{n}</span><div className="track"><i style={{ width: `${v * 100}%` }} /><s /></div><span>{v.toFixed(2)}</span>
            </div>
          ))}
        </div>
        <p className="caption">AUROC неуверенности против «в патче ошибка», test. Заранее выбранная мера (взаимная информация) — самая слабая. Большинство ошибок — устойчивая (алеаторная) неуверенность,
          а не расхождение между проходами dropout. Менять меру по этому наблюдению на test было бы подгонкой под test, поэтому выбор сделан на val (ниже).</p>
      </section>

      <section className="block wide">
        <h3>Выбор меры неуверенности на val (11.11)</h3>
        <table className="sel">
          <thead><tr><th>мера</th><th>val: площадь под кривой передачи</th><th>test: площадь</th></tr></thead>
          <tbody>
            {SELECTION.map(([n, v, t]) => (
              <tr key={n} className={n.startsWith("Max") ? "chosen" : undefined}><td>{n}</td><td>{v.toFixed(3)}</td><td>{t.toFixed(3)}</td></tr>
            ))}
            <tr className="ref"><td>случайно / оракул</td><td>0.932 / 0.973</td><td>0.801 / 0.875</td></tr>
          </tbody>
        </table>
        <p className="caption">Правило (наибольшая площадь на val) закоммичено до расчёта. Взаимная информация не лучше случайной передачи и на val, так что отказ от неё не подгонка под test.
          Выбор на val (max-prob) на test обошёл MI, но уступил энтропии: разница ~0.01 при 6 образцах — шум. Вывод: использовать полную неуверенность, окончательный выбор — через k-fold по образцам.</p>
      </section>
    </>
  );
}
