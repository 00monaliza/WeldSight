"""Build the WeldSight results page.

usage: uv run python docs/demo/build_results_page.py . docs/demo/results_page.html [run_dir]
Without run_dir the reference run from README (git e3dc84c) is shown.
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
RUN = Path(sys.argv[3]) if len(sys.argv) > 3 else None

CLASSES = ["crack", "porosity", "lack_of_penetration"]
RU = {"crack": "Трещина", "porosity": "Пористость", "lack_of_penetration": "Непровар"}
UNC_RU = {
    "predictive_entropy": "Предсказательная энтропия",
    "expected_entropy": "Ожидаемая энтропия",
    "max_prob": "1 − max p",
    "mutual_information": "Взаимная информация (BALD)",
}
card = json.loads((REPO / "data/processed/riawelc_v1/dataset_card.json").read_text())


def img_b64(path, width=None, quality=82):
    im = Image.open(path).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ results data
if RUN is None:
    R = {
        "source": "ref",
        "label": "Эталонный прогон <code>riawelc_cpu</code> из репозитория",
        "prov": "git e3dc84c · data 2c33eedecb1697b9 · seed 42 · torch 2.14 / smp 0.5.0 · CPU · лучшая эпоха 5 из 9",
        "n_patches": 3073,
        "n_rad": 21,
        "n_groups": 6,
        "T": 10,
        "cls": {
            "crack": dict(
                support=977,
                spec=2,
                recall=0.570,
                lo=0.526,
                hi=0.667,
                precision=0.670,
                f1=0.62,
                auroc=0.86,
            ),
            "porosity": dict(
                support=936,
                spec=4,
                recall=0.931,
                lo=0.868,
                hi=0.972,
                precision=0.935,
                f1=0.93,
                auroc=0.99,
            ),
            "lack_of_penetration": dict(
                support=658,
                spec=3,
                recall=0.190,
                lo=0.023,
                hi=0.428,
                precision=0.644,
                f1=0.29,
                auroc=0.72,
            ),
        },
        "macro": dict(recall=0.564, precision=0.750, f1=0.61, auroc=0.86),
        "ref_pts": {
            "model": [(0, 0.604), (0.1, 0.654), (0.2, 0.712), (0.3, 0.768)],
            "random": [(0, 0.604), (0.1, 0.641), (0.2, 0.679), (0.3, 0.715)],
            "oracle": [(0, 0.604), (0.1, 0.690), (0.2, 0.759), (0.3, 0.815)],
        },
        "curve": None,
        "area": {"model": 0.832, "random": 0.801, "oracle": 0.875},
        "unc_auroc": {
            "predictive_entropy": 0.82,
            "expected_entropy": 0.82,
            "max_prob": 0.80,
            "mutual_information": 0.54,
        },
        "val": {"macro_f1": 0.89, "lop_recall": 0.79},
        "history": None,
        "explorer": None,
    }
else:
    m = json.loads((RUN / "metrics_test.json").read_text())
    prov = json.loads((RUN / "provenance.json").read_text())
    hist = pd.read_csv(RUN / "history.csv")
    pred = pd.read_csv(RUN / "predictions_test.csv")
    man = pd.read_csv(REPO / "data/processed/riawelc_v1/manifest.csv")
    spec = {
        c: int(man[(man.split == "test") & (man[f"cls_{c}"] == 1)].group_id.nunique())
        for c in CLASSES
    }
    curve = pd.read_csv(RUN / "referral_test.csv")
    pl = m["patch_level"]
    rr = m["referral_radiograph"]
    best = hist.loc[hist["val/macro_f1"].idxmax()]
    git = prov.get("git", {})
    env = prov.get("env", {})
    R = {
        "source": "run",
        "label": f"Свежий прогон <code>{RUN.name}</code>",
        "prov": (
            f"git {str(git.get('commit', '?'))[:7]}{' (dirty)' if git.get('dirty') else ''} · "
            f"data {m['data_version']} · seed {prov.get('seed')} · torch {env.get('torch', '?')} / "
            f"smp {env.get('segmentation_models_pytorch', env.get('smp', '?'))} · CPU · "
            f"лучшая эпоха {m['checkpoint_epoch']} из {len(hist)}"
        ),
        "n_patches": m["n_patches"],
        "n_rad": m["n_radiographs"],
        "n_groups": m["n_groups"],
        "T": m["mc_samples"],
        "cls": {
            c: dict(
                support=pl[c]["support"],
                spec=spec[c],
                recall=pl[c]["recall"],
                lo=pl[c]["recall_ci95"][0],
                hi=pl[c]["recall_ci95"][1],
                precision=pl[c]["precision"],
                f1=pl[c]["f1"],
                auroc=pl[c]["auroc"],
            )
            for c in CLASSES
        },
        "macro": {k: pl["_macro"][k] for k in ("recall", "precision", "f1", "auroc")},
        "ref_pts": {
            k: [(b, rr[k]["all"][f"recall@{int(b * 100)}%"]) for b in (0, 0.1, 0.2, 0.3)]
            for k in ("model", "random", "oracle")
        },
        "curve": {
            k: curve[(curve.curve == k) & (curve["class"].astype(str) == "all")]
            .sort_values("fraction")[["fraction", "recall"]]
            .to_numpy()
            .tolist()
            for k in ("model", "random", "oracle")
        },
        "area": {k: rr[k]["all"]["area"] for k in ("model", "random", "oracle")},
        "unc_auroc": m["uncertainty_error_auroc"],
        "val": {
            "macro_f1": float(best["val/macro_f1"]),
            "lop_recall": float(best["val/recall_lack_of_penetration"]),
        },
        "history": hist,
        "explorer": None,
    }
    # ---------------- explorer sample: real test patches with real model outputs
    thr = m["threshold"]
    probs = pred[[f"prob_{c}" for c in CLASSES]].to_numpy()
    trues = pred[[f"true_{c}" for c in CLASSES]].to_numpy()
    err = ((probs >= thr) != trues).any(1)
    unc_key = m.get("uncertainty", "mutual_information")
    rad_score = pred.groupby("radiograph_id")[f"unc_{unc_key}"].max().sort_values(ascending=False)
    n_ref = int(np.floor(0.2 * len(rad_score)))
    referred = set(rad_score.index[:n_ref])
    rng = np.random.default_rng(0)
    pick = []
    for c_i in range(len(CLASSES)):
        pos = np.where(trues[:, c_i] == 1)[0]
        hit = pos[probs[pos, c_i] >= thr]
        miss = pos[probs[pos, c_i] < thr]
        pick += list(rng.choice(hit, min(5, len(hit)), replace=False)) + list(
            rng.choice(miss, min(5, len(miss)), replace=False)
        )
    neg = np.where(trues.sum(1) == 0)[0]
    fp = neg[err[neg]]
    tn = neg[~err[neg]]
    pick += list(rng.choice(fp, min(6, len(fp)), replace=False)) + list(
        rng.choice(tn, min(6, len(tn)), replace=False)
    )
    pe = pred["unc_predictive_entropy"].to_numpy()
    pick += list(np.argsort(-pe)[:6])
    pick = list(dict.fromkeys(int(i) for i in pick))
    pmap = man.set_index("patch_id")["image"]
    items = []
    for i in pick:
        r = pred.iloc[i]
        im = (
            Image.open(REPO / "data/processed/riawelc_v1" / pmap[r.patch_id])
            .convert("L")
            .resize((128, 128), Image.LANCZOS)
        )
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        items.append(
            {
                "id": r.patch_id,
                "rad": r.radiograph_id,
                "spec": r.group_id,
                "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                "true": [int(t) for t in trues[i]],
                "prob": [round(float(p), 3) for p in probs[i]],
                "pe": round(float(pe[i]), 3),
                "mi": round(float(r[f"unc_{unc_key}"]), 4),
                "err": bool(err[i]),
                "ref": r.radiograph_id in referred,
            }
        )
    R["explorer"] = {
        "items": items,
        "thr": thr,
        "pe_max": float(pe.max()),
        "n_ref": n_ref,
        "n_rad": len(rad_score),
        "unc_key": unc_key,
    }


# ------------------------------------------------------------------ svg helpers
def pct(x):
    return f"{100 * x:.1f}".replace(".", ",")


def recall_chart():
    W, H, L, Rm = 640, 190, 150, 24
    rows = CLASSES
    x = lambda v: L + v * (W - L - Rm)
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Recall по классам с 95% доверительными интервалами">'
    ]
    for t in range(0, 101, 20):
        out.append(
            f'<line class="grid" x1="{x(t / 100)}" x2="{x(t / 100)}" y1="14" y2="{H - 30}"/>'
            f'<text class="tick" x="{x(t / 100)}" y="{H - 12}" text-anchor="middle">{t}%</text>'
        )
    for i, c in enumerate(rows):
        d = R["cls"][c]
        y = 38 + i * 48
        out.append(
            f'<text class="rowlab" x="{L - 14}" y="{y + 5}" text-anchor="end">{RU[c]}</text>'
        )
        out.append(f'<line class="ci c{i}" x1="{x(d["lo"])}" x2="{x(d["hi"])}" y1="{y}" y2="{y}"/>')
        for e in ("lo", "hi"):
            out.append(
                f'<line class="ci c{i}" x1="{x(d[e])}" x2="{x(d[e])}" y1="{y - 7}" y2="{y + 7}"/>'
            )
        out.append(f'<circle class="dot c{i}" cx="{x(d["recall"])}" cy="{y}" r="6.5"/>')
        if d["hi"] > 0.85:
            out.append(
                f'<text class="val" x="{x(d["lo"]) - 10}" y="{y + 5}" text-anchor="end">{pct(d["recall"])}</text>'
            )
        else:
            out.append(
                f'<text class="val" x="{x(d["hi"]) + 10}" y="{y + 5}">{pct(d["recall"])}</text>'
            )
    out.append("</svg>")
    return "".join(out)


def referral_chart():
    W, H, L, B, T, Rm = 640, 300, 52, 40, 16, 20
    xmax = 0.5
    x = lambda f: L + f / xmax * (W - L - Rm)
    ys = [p[1] for k in R["ref_pts"] for p in R["ref_pts"][k]]
    if R["curve"]:
        ys += [r for k in R["curve"] for f, r in R["curve"][k] if f <= xmax]
    y0 = np.floor(min(ys) * 10) / 10
    y1 = min(1.0, np.ceil(max(ys) * 10) / 10)
    y = lambda r: T + (y1 - r) / (y1 - y0) * (H - T - B)
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Recall в зависимости от доли снимков, отправленных инспектору">'
    ]
    for f in np.arange(0, xmax + 1e-9, 0.1):
        out.append(
            f'<line class="grid" x1="{x(f)}" x2="{x(f)}" y1="{T}" y2="{H - B}"/>'
            f'<text class="tick" x="{x(f)}" y="{H - B + 18}" text-anchor="middle">{int(round(f * 100))}%</text>'
        )
    for r in np.arange(y0, y1 + 1e-9, 0.1):
        out.append(
            f'<line class="grid" x1="{L}" x2="{W - Rm}" y1="{y(r)}" y2="{y(r)}"/>'
            f'<text class="tick" x="{L - 8}" y="{y(r) + 4}" text-anchor="end">{int(round(r * 100))}%</text>'
        )
    out.append(
        f'<text class="axlab" x="{(L + W - Rm) / 2}" y="{H - 4}" text-anchor="middle">доля снимков, отправленных инспектору</text>'
    )
    for k in ("oracle", "random", "model"):
        if R["curve"]:
            pts = [(f, r) for f, r in R["curve"][k] if f <= xmax + 1e-9]
            d = f"M{x(pts[0][0])},{y(pts[0][1])}"
            for f1, r1 in pts[1:]:
                d += f" H{x(f1)} V{y(r1)}" if k != "random" else f" L{x(f1)},{y(r1)}"
        else:
            pts = R["ref_pts"][k]
            d = "M" + " L".join(f"{x(f)},{y(r)}" for f, r in pts)
        out.append(f'<path class="ln {k}" d="{d}"/>')
    for k in ("oracle", "random", "model"):
        for f, r in R["ref_pts"][k]:
            out.append(
                f'<circle class="pt {k}" cx="{x(f)}" cy="{y(r)}" r="{4.5 if k == "model" else 3.5}"/>'
            )
    f, r = R["ref_pts"]["model"][-1]
    out.append(f'<text class="endlab" x="{x(f) + 6}" y="{y(r) + 20}">{pct(r)}%</text>')
    out.append("</svg>")
    return "".join(out)


def unc_chart():
    items = sorted(R["unc_auroc"].items(), key=lambda kv: -kv[1])
    W, L, Rm = 640, 230, 56
    H = 24 + len(items) * 40
    x = lambda v: L + (v - 0.5) / 0.5 * (W - L - Rm)
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="AUROC обнаружения ошибок по мере неопределённости">'
    ]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out.append(
            f'<line class="grid" x1="{x(t)}" x2="{x(t)}" y1="4" y2="{H - 22}"/>'
            f'<text class="tick" x="{x(t)}" y="{H - 6}" text-anchor="middle">'
            + f"{t:.1f}".replace(".", ",")
            + "</text>"
        )
    for i, (k, v) in enumerate(items):
        yy = 10 + i * 40
        cls = "bar pre" if k == "mutual_information" else "bar"
        out.append(
            f'<text class="rowlab" x="{L - 12}" y="{yy + 17}" text-anchor="end">{UNC_RU.get(k, k)}</text>'
        )
        out.append(
            f'<rect class="{cls}" x="{x(0.5)}" y="{yy + 4}" width="{max(2, x(v) - x(0.5))}" height="20" rx="2"/>'
        )
        out.append(
            f'<text class="val" x="{x(v) + 8}" y="{yy + 19}">'
            + f"{v:.2f}".replace(".", ",")
            + "</text>"
        )
    out.append("</svg>")
    return "".join(out)


def history_chart():
    h = R["history"]
    if h is None:
        return ""
    W, H, L, B, T, Rm = 640, 240, 44, 36, 14, 90
    n = len(h)
    x = lambda e: L + (e - 1) / max(1, n - 1) * (W - L - Rm)
    y = lambda v: T + (1 - v) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Метрики на валидации по эпохам">']
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(
            f'<line class="grid" x1="{L}" x2="{W - Rm}" y1="{y(v)}" y2="{y(v)}"/>'
            f'<text class="tick" x="{L - 8}" y="{y(v) + 4}" text-anchor="end">'
            + f"{v:.2f}".replace(".", ",")
            + "</text>"
        )
    for e in h["epoch"]:
        out.append(
            f'<text class="tick" x="{x(e)}" y="{H - B + 18}" text-anchor="middle">{e}</text>'
        )
    out.append(
        f'<text class="axlab" x="{(L + W - Rm) / 2}" y="{H - 2}" text-anchor="middle">эпоха</text>'
    )
    series = [("val/macro_f1", "macro-F1", "model")] + [
        (f"val/recall_{c}", RU[c], f"c{i}") for i, c in enumerate(CLASSES)
    ]
    ends = []
    for col, lab, cls in series:
        pts = list(zip(h["epoch"], h[col], strict=True))
        out.append(
            f'<path class="ln {cls}" d="M' + " L".join(f"{x(e)},{y(v)}" for e, v in pts) + '"/>'
        )
        ends.append([y(pts[-1][1]) + 4, lab, cls, x(pts[-1][0]) + 8])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for ty, lab, cls, tx in ends:
        out.append(f'<text class="endlab {cls}" x="{tx}" y="{ty}">{lab}</text>')
    be = int(h.loc[h["val/macro_f1"].idxmax(), "epoch"])
    out.append(f'<line class="best" x1="{x(be)}" x2="{x(be)}" y1="{T}" y2="{H - B}"/>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ html
rows = []
for i, c in enumerate(CLASSES):
    d = R["cls"][c]
    rows.append(
        f'<tr><th scope="row"><span class="sw c{i}"></span>{RU[c]}</th><td>{d["support"]} / {d["spec"]}</td>'
        f'<td><b>{pct(d["recall"])}</b> <span class="ci-t">{pct(d["lo"])}–{pct(d["hi"])}</span></td>'
        f"<td>{pct(d['precision'])}</td><td>{d['f1']:.2f}</td><td>{d['auroc']:.2f}</td></tr>".replace(
            ".", ","
        )
    )
mc = R["macro"]
rows.append(
    f'<tr class="macro"><th scope="row">Среднее (macro)</th><td></td><td><b>{pct(mc["recall"])}</b></td>'
    f"<td>{pct(mc['precision'])}</td><td>{mc['f1']:.2f}</td><td>{mc['auroc']:.2f}</td></tr>".replace(
        ".", ","
    )
)
ref_rows = "".join(
    f'<tr><th scope="row">{int(b * 100)}%</th><td><b>{pct(R["ref_pts"]["model"][i][1])}</b></td>'
    f"<td>{pct(R['ref_pts']['random'][i][1])}</td><td>{pct(R['ref_pts']['oracle'][i][1])}</td></tr>"
    for i, b in enumerate((0, 0.1, 0.2, 0.3))
)
ref_rows += (
    f'<tr class="macro"><th scope="row">Площадь</th><td><b>{R["area"]["model"]:.3f}</b></td>'
    f"<td>{R['area']['random']:.3f}</td><td>{R['area']['oracle']:.3f}</td></tr>"
).replace(".", ",")

lop = R["cls"]["lack_of_penetration"]
mi = R["unc_auroc"].get("mutual_information", float("nan"))
pe = R["unc_auroc"].get("predictive_entropy", float("nan"))
examples = img_b64(REPO / "notebooks/figures/examples.png", width=1100, quality=78)
gps = card["groups_per_split"]

if R["source"] == "ref":
    status = (
        '<div class="note" id="run-status"><b>Идёт новый прогон.</b> Ниже результаты эталонного прогона из репозитория. '
        "Сейчас то же обучение повторяется в этой сессии на тех же данных (data_version совпал). Когда оно закончится, "
        "страница обновится по этой же ссылке: появятся свежие метрики, кривая обучения и разбор тестовых патчей.</div>"
    )
else:
    status = ""

explorer_html = ""
if R["explorer"]:
    ex = R["explorer"]
    explorer_html = f"""
<section id="explorer">
  <div class="sec-head"><span class="eyebrow">Демо</span><h2>Что модель видит на тестовых патчах</h2></div>
  <p class="lede">{len(ex["items"])} настоящих патчей из тестовой выборки и настоящие выходы модели: попадания и пропуски по каждому классу, ложные срабатывания, чистые патчи и самые неуверенные случаи. Порог {str(ex["thr"]).replace(".", ",")}. Снимок помечен «к инспектору», если попадает в {ex["n_ref"]} из {ex["n_rad"]} самых неопределённых снимков (бюджет 20%).</p>
  <div class="chips" role="group" aria-label="Фильтр патчей">
    <button class="chip" aria-pressed="true" data-f="all">Все</button>
    <button class="chip" aria-pressed="false" data-f="err">Ошибки модели</button>
    <button class="chip" aria-pressed="false" data-f="ref">Ушли к инспектору</button>
    <button class="chip" aria-pressed="false" data-f="c0">Трещина</button>
    <button class="chip" aria-pressed="false" data-f="c1">Пористость</button>
    <button class="chip" aria-pressed="false" data-f="c2">Непровар</button>
    <button class="chip" aria-pressed="false" data-f="none">Без дефекта</button>
  </div>
  <div class="ex-wrap">
    <div class="ex-grid" id="grid"></div>
    <aside class="ex-detail" id="detail" aria-live="polite"></aside>
  </div>
</section>
<script>
const EX = {json.dumps(ex, ensure_ascii=False)};
const CL = ["Трещина", "Пористость", "Непровар"];
let cur = 0, filt = "all";
const pass = (it) => filt === "all" || (filt === "err" && it.err) || (filt === "ref" && it.ref)
  || (filt === "none" && it.true.every(t => !t)) || (filt[0] === "c" && it.true[+filt[1]]);
function grid() {{
  const g = document.getElementById("grid"); g.innerHTML = "";
  EX.items.forEach((it, i) => {{
    if (!pass(it)) return;
    const b = document.createElement("button");
    b.className = "tile" + (it.err ? " is-err" : "") + (i === cur ? " sel" : "");
    b.setAttribute("aria-label", it.id);
    b.innerHTML = `<img src="${{it.img}}" alt="">` + (it.ref ? '<span class="flag">инсп.</span>' : "");
    b.onclick = () => {{ cur = i; grid(); detail(); }};
    g.appendChild(b);
  }});
}}
function detail() {{
  const it = EX.items[cur], d = document.getElementById("detail");
  const truth = it.true.map((t, k) => t ? CL[k] : null).filter(Boolean).join(", ") || "без дефекта";
  const bars = it.prob.map((p, k) => {{
    const on = p >= EX.thr, ok = on === !!it.true[k];
    return `<div class="pb"><span class="pb-l">${{CL[k]}}${{it.true[k] ? ' <i>разметка</i>' : ''}}</span>
      <span class="pb-t"><span class="pb-f c${{k}}" style="width:${{(p*100).toFixed(1)}}%"></span><span class="pb-thr" style="left:${{EX.thr*100}}%"></span></span>
      <span class="pb-v ${{ok ? '' : 'bad'}}">${{(p*100).toFixed(0)}}%</span></div>`;
  }}).join("");
  const u = Math.min(1, it.pe / EX.pe_max);
  d.innerHTML = `<img class="big" src="${{it.img}}" alt="Патч ${{it.id}}">
    <div class="pill-row"><span class="pill ${{it.err ? 'bad' : 'good'}}">${{it.err ? 'Ошибка модели' : 'Модель права'}}</span>
    <span class="pill ${{it.ref ? 'warn' : ''}}">${{it.ref ? 'Снимок уходит инспектору' : 'Решает модель'}}</span></div>
    <dl><dt>Разметка</dt><dd>${{truth}}</dd><dt>Снимок</dt><dd><code>${{it.rad}}</code></dd><dt>Образец шва</dt><dd><code>${{it.spec}}</code></dd></dl>
    <div class="pbs">${{bars}}</div>
    <div class="unc"><span>Неопределённость (энтропия)</span><span class="pb-t"><span class="pb-f unc-f" style="width:${{(u*100).toFixed(0)}}%"></span></span><span class="pb-v">${{it.pe.toFixed(2)}}</span></div>`;
}}
document.querySelectorAll(".chip").forEach(c => c.onclick = () => {{
  filt = c.dataset.f;
  document.querySelectorAll(".chip").forEach(o => o.setAttribute("aria-pressed", o === c));
  const first = EX.items.findIndex(pass); if (first >= 0 && !pass(EX.items[cur])) cur = first;
  grid(); detail();
}});
grid(); detail();
</script>"""

history_html = ""
if R["history"] is not None:
    history_html = f"""
<section>
  <div class="sec-head"><span class="eyebrow">Обучение</span><h2>Валидация по эпохам</h2></div>
  <p class="lede">Модель выбирается по macro-F1 на валидации (вертикальная линия — лучшая эпоха). Ранняя остановка, если 4 эпохи подряд нет улучшения.</p>
  <div class="chart">{history_chart()}</div>
</section>"""

html = f"""<title>WeldSight</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column like an inspection report; charts full column width; film-viewer neutrals. */
:root {{
  --bg: #eef1f3; --surface: #ffffff; --ink: #172028; --muted: #5a6670; --line: #d3d9de;
  --accent: #1f5f8b; --accent-soft: #dbe8f1;
  --c0: #c2410c; --c1: #6d28d9; --c2: #0f766e;
  --good: #1d7a43; --bad: #b42318; --warn: #a15c00;
  --f-display: "Barlow Condensed", "Arial Narrow", sans-serif;
  --f-body: "IBM Plex Sans", system-ui, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f151a; --surface: #172028; --ink: #e4eaee; --muted: #93a1ab; --line: #2a3640;
  --accent: #7fb6dd; --accent-soft: #1c3346;
  --c0: #fb923c; --c1: #a78bfa; --c2: #2dd4bf;
  --good: #4ade80; --bad: #f87171; --warn: #fbbf24; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0f151a; --surface: #172028; --ink: #e4eaee; --muted: #93a1ab; --line: #2a3640;
  --accent: #7fb6dd; --accent-soft: #1c3346;
  --c0: #fb923c; --c1: #a78bfa; --c2: #2dd4bf;
  --good: #4ade80; --bad: #f87171; --warn: #fbbf24; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font: 16px/1.6 var(--f-body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
main > * {{ min-width: 0; }}
main {{ max-width: 860px; margin: 0 auto; padding-block: 40px 72px; display: grid; gap: 56px; }}
code {{ font: 0.88em var(--f-mono); }}
h1, h2 {{ font-family: var(--f-display); font-weight: 600; letter-spacing: 0.01em; text-wrap: balance; margin: 0; line-height: 1.1; }}
h1 {{ font-size: clamp(44px, 9vw, 76px); font-weight: 700; }}
h2 {{ font-size: clamp(28px, 5vw, 36px); }}
.eyebrow {{ font: 500 12px var(--f-mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); }}
.sec-head {{ display: grid; gap: 6px; margin-bottom: 14px; }}
.lede {{ max-width: 66ch; color: var(--muted); margin: 0 0 18px; }}
p {{ max-width: 66ch; }}
header {{ display: grid; gap: 16px; }}
.sub {{ font-size: 19px; max-width: 60ch; margin: 0; }}
.meta {{ font: 13px/1.5 var(--f-mono); color: var(--muted); }}
.note {{ background: var(--accent-soft); border-left: 3px solid var(--accent); padding: 12px 16px; border-radius: 4px; max-width: 72ch; }}
.facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }}
.fact {{ background: var(--surface); padding: 16px 18px; display: grid; gap: 4px; align-content: start; }}
.fact b {{ font: 600 34px/1 var(--f-display); font-variant-numeric: tabular-nums; }}
.fact span {{ color: var(--muted); font-size: 14px; line-height: 1.45; }}
figure {{ margin: 0; }}
.film {{ background: #0b0e10; border-radius: 6px; padding: 10px; overflow-x: auto; }}
.film img {{ display: block; width: 100%; min-width: 560px; }}
figcaption {{ color: var(--muted); font-size: 14px; margin-top: 8px; }}
.chart {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px; overflow-x: auto; }}
.chart svg {{ display: block; width: 100%; min-width: 480px; height: auto; font-family: var(--f-body); }}
.grid {{ stroke: var(--line); stroke-width: 1; }}
.tick {{ fill: var(--muted); font: 12px var(--f-mono); }}
.axlab {{ fill: var(--muted); font-size: 13px; }}
.rowlab {{ fill: var(--ink); font-size: 14px; }}
.val, .endlab {{ fill: var(--ink); font: 500 13px var(--f-mono); }}
.ci {{ stroke-width: 2.5; }}
.c0 {{ stroke: var(--c0); fill: var(--c0); }} .c1 {{ stroke: var(--c1); fill: var(--c1); }} .c2 {{ stroke: var(--c2); fill: var(--c2); }}
.ln {{ fill: none; stroke-width: 2.5; }}
.ln.model {{ stroke: var(--accent); stroke-width: 3; }} .ln.random {{ stroke: var(--muted); stroke-dasharray: 5 5; }} .ln.oracle {{ stroke: var(--muted); opacity: .55; }}
.ln.c0 {{ stroke: var(--c0); }} .ln.c1 {{ stroke: var(--c1); }} .ln.c2 {{ stroke: var(--c2); }}
text.endlab.c0 {{ fill: var(--c0); stroke: none; }} text.endlab.c1 {{ fill: var(--c1); stroke: none; }} text.endlab.c2 {{ fill: var(--c2); stroke: none; }} text.endlab.model {{ fill: var(--accent); stroke: none; }}
.pt.model {{ fill: var(--accent); }} .pt.random {{ fill: var(--muted); }} .pt.oracle {{ fill: var(--muted); opacity: .55; }}
.best {{ stroke: var(--ink); stroke-dasharray: 2 4; opacity: .6; }}
.bar {{ fill: var(--muted); opacity: .55; }} .bar.pre {{ fill: var(--warn); opacity: 1; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 14px; color: var(--muted); margin-top: 10px; }}
.legend i {{ display: inline-block; width: 22px; height: 0; border-top: 3px solid; vertical-align: middle; margin-right: 6px; }}
.tbl {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 15px; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
th[scope="row"], thead th:first-child {{ text-align: left; font-weight: 500; }}
thead th {{ font: 500 12px var(--f-mono); text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
tr.macro th, tr.macro td {{ border-bottom: 0; color: var(--muted); }}
.ci-t {{ color: var(--muted); font-size: 13px; margin-left: 4px; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
.sw.c0 {{ background: var(--c0); }} .sw.c1 {{ background: var(--c1); }} .sw.c2 {{ background: var(--c2); }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; align-items: start; }}
.two > * {{ min-width: 0; }}
.callout {{ border-top: 2px solid var(--ink); padding-top: 12px; }}
.callout h3 {{ margin: 0 0 6px; font: 600 20px var(--f-display); letter-spacing: .02em; }}
.callout p {{ margin: 0; color: var(--muted); font-size: 15px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
.chip {{ font: 500 14px var(--f-body); padding: 6px 12px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); cursor: pointer; }}
.chip[aria-pressed="true"] {{ background: var(--accent); border-color: var(--accent); color: var(--surface); }}
.chip:focus-visible, .tile:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.ex-wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 20px; align-items: start; }}
@media (max-width: 720px) {{ .ex-wrap {{ grid-template-columns: 1fr; }} }}
.ex-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(72px, 1fr)); gap: 6px; }}
.tile {{ position: relative; padding: 0; border: 2px solid transparent; border-radius: 4px; background: #0b0e10; cursor: pointer; aspect-ratio: 1; max-width: 100%; overflow: hidden; }}
.tile img {{ display: block; width: 100%; height: 100%; object-fit: cover; }}
.tile.is-err {{ border-color: var(--bad); }}
.tile.sel {{ outline: 3px solid var(--accent); outline-offset: 1px; }}
.flag {{ position: absolute; right: 2px; bottom: 2px; font: 500 10px var(--f-mono); background: var(--warn); color: #000; padding: 0 4px; border-radius: 2px; }}
.ex-detail {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px; display: grid; gap: 12px; position: sticky; top: calc(env(safe-area-inset-top, 0px) + 12px); }}
.ex-detail .big {{ width: 100%; border-radius: 4px; image-rendering: auto; }}
.pill-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.pill {{ font: 500 12px var(--f-mono); padding: 3px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }}
.pill.good {{ color: var(--good); border-color: var(--good); }} .pill.bad {{ color: var(--bad); border-color: var(--bad); }} .pill.warn {{ color: var(--warn); border-color: var(--warn); }}
dl {{ display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; margin: 0; font-size: 14px; }}
dt {{ color: var(--muted); }} dd {{ margin: 0; min-width: 0; overflow-wrap: anywhere; }}
.pbs {{ display: grid; gap: 8px; }}
.pb, .unc {{ display: grid; grid-template-columns: 1fr 44px; gap: 2px 8px; font-size: 13px; align-items: center; }}
.pb-l, .unc > span:first-child {{ grid-column: 1 / -1; }}
.pb-l i {{ font-style: normal; font: 11px var(--f-mono); color: var(--muted); margin-left: 4px; }}
.pb-t {{ position: relative; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }}
.pb-f {{ position: absolute; inset: 0 auto 0 0; border-radius: 4px; }}
.pb-f.c0 {{ background: var(--c0); }} .pb-f.c1 {{ background: var(--c1); }} .pb-f.c2 {{ background: var(--c2); }} .unc-f {{ background: var(--warn); }}
.pb-thr {{ position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--ink); }}
.pb-v {{ font: 500 13px var(--f-mono); text-align: right; font-variant-numeric: tabular-nums; }}
.pb-v.bad {{ color: var(--bad); }}
.limits {{ display: grid; gap: 10px; padding-left: 20px; margin: 0; max-width: 72ch; }}
footer {{ color: var(--muted); font-size: 13px; display: grid; gap: 6px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
<main>
<header>
  <span class="eyebrow">Этап 1 · радиографический контроль сварных швов</span>
  <h1>WeldSight</h1>
  <p class="sub">Модель находит дефекты сварных швов на рентгеновских снимках и сама отмечает снимки, в которых не уверена, чтобы их проверил инспектор.</p>
  <div class="meta">{R["label"]} · {R["prov"]}</div>
  {status}
</header>

<section>
  <div class="sec-head"><span class="eyebrow">Данные</span><h2>RIAWELC после очистки</h2></div>
  <p class="lede">Патчи 227×227 из 109 радиографий, классы «трещина», «пористость», «непровар» и «без дефекта». Оригинальное разбиение датасета пересекается между train и test, поэтому разбиение сделано заново по образцам сварных швов.</p>
  <div class="facts">
    <div class="fact"><b>{card["n_records_raw"]:,}</b><span>патчей в исходном датасете</span></div>
    <div class="fact"><b>{card["duplicates_dropped"]:,}</b><span>побайтовых дубликатов между train и test удалено</span></div>
    <div class="fact"><b>108 / 109</b><span>радиографий встречались и в train, и в test в оригинальном разбиении</span></div>
    <div class="fact"><b>{len(gps["train"])} · {len(gps["val"])} · {len(gps["test"])}</b><span>образцов швов в train · val · test, без пересечений</span></div>
  </div>
</section>

<section>
  <figure>
    <div class="film"><img src="{examples}" alt="Примеры патчей: по 8 на класс без дефекта, трещина, пористость, непровар"></div>
    <figcaption>Примеры патчей по классам (сверху вниз): без дефекта, трещина, пористость, непровар. Подпись — образец шва.</figcaption>
  </figure>
</section>

<section>
  <div class="two">
    <div class="callout"><h3>Ловушка «плоского» патча</h3><p>Патчи без дефекта заметно ровнее остальных: одно только стандартное отклонение яркости отделяет их от дефектных с AUROC 0,91. Поэтому точность «дефект / не дефект» здесь не показатель, и смотрим recall по каждому классу.</p></div>
    <div class="callout"><h3>Мало независимых образцов</h3><p>В тесте каждый класс дефекта есть только в 2–4 образцах швов. Поэтому recall приводится с 95% доверительным интервалом (кластерный бутстрэп по снимкам), и интервалы широкие.</p></div>
  </div>
</section>

<section>
  <div class="sec-head"><span class="eyebrow">Результаты · тест</span><h2>Сколько дефектов модель находит</h2></div>
  <p class="lede">Только реальные снимки: {R["n_patches"]:,} патчей, {R["n_rad"]} радиографий, {R["n_groups"]} образцов швов, которых модель не видела. MC Dropout, T = {R["T"]}, порог 0,5. Точка — recall, отрезок — 95% интервал.</p>
  <div class="chart">{recall_chart()}</div>
  <div class="tbl" style="margin-top:16px">
    <table>
      <thead><tr><th>Класс</th><th>Патчи / образцы</th><th>Recall, %</th><th>Precision, %</th><th>F1</th><th>AUROC</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</section>

<section>
  <div class="two">
    <div class="callout"><h3>Пористость находится уверенно, непровар — нет</h3><p>Пористость находится в {pct(R["cls"]["porosity"]["recall"])}% случаев. Непровар — только в {pct(lop["recall"])}%, и интервал {pct(lop["lo"])}–{pct(lop["hi"])}% говорит, что результат сильно зависит от конкретного шва.</p></div>
    <div class="callout"><h3>Валидация обещала больше</h3><p>На валидации macro-F1 был {R["val"]["macro_f1"]:.2f}, recall непровара {pct(R["val"]["lop_recall"])}%. На тесте — {R["macro"]["f1"]:.2f} и {pct(lop["recall"])}%. Обе выборки из незнакомых образцов, так что это разброс между швами. Нужна k-fold проверка по образцам.</p></div>
  </div>
</section>
{history_html}
<section>
  <div class="sec-head"><span class="eyebrow">Неопределённость</span><h2>Если сомнительные снимки отдать инспектору</h2></div>
  <p class="lede">Снимки сортируются по максимальной неопределённости модели (взаимная информация MC Dropout), верхние f% уходят к человеку. Считаем, что инспектор находит все дефекты на своих снимках, поэтому кривая — верхняя оценка. Для сравнения: случайный выбор снимков и идеальный «оракул».</p>
  <div class="chart">{referral_chart()}
    <div class="legend"><span><i style="border-color:var(--accent)"></i>Модель</span><span><i style="border-color:var(--muted);border-top-style:dashed"></i>Случайно</span><span><i style="border-color:var(--muted);opacity:.55"></i>Оракул</span></div>
  </div>
  <div class="tbl" style="margin-top:16px">
    <table>
      <thead><tr><th>К инспектору</th><th>Модель, %</th><th>Случайно, %</th><th>Оракул, %</th></tr></thead>
      <tbody>{ref_rows}</tbody>
    </table>
  </div>
</section>

<section>
  <div class="sec-head"><span class="eyebrow">Какая мера сомнения работает</span><h2>Отличает ли неопределённость ошибки</h2></div>
  <p class="lede">AUROC: насколько хорошо каждая мера неопределённости отделяет патчи, где модель ошиблась, от правильных. 0,5 — не лучше монетки. Оранжевым — мера, выбранная заранее для ранжирования.</p>
  <div class="chart">{unc_chart()}</div>
  <p>Заранее выбранная взаимная информация почти не несёт сигнала об ошибках ({mi:.2f}), а обычная предсказательная энтропия даёт {pe:.2f}. То есть модель чаще ошибается там, где она стабильно не уверена (шумные, неоднозначные патчи), а не там, где расходятся dropout-маски. Менять меру по результату на тесте нельзя — это подгонка под тест. Выбирать её нужно на валидации.</p>
</section>
{explorer_html}
<section>
  <div class="sec-head"><span class="eyebrow">Границы результата</span><h2>Что этот прогон не показывает</h2></div>
  <ol class="limits">
    <li>Это классификация патчей, а не сегментация: в RIAWELC нет масок, поэтому IoU нет. Сегментация — на датасете SWRD (6 классов, полигоны), прогон в Colab.</li>
    <li>Обучение на 4 ядрах CPU без весов ImageNet: ResNet-18, 8 000 случайных патчей за эпоху. Это проверка конвейера, а не потолок качества. Эталонная конфигурация — ResNet-34/ImageNet на GPU.</li>
    <li>Тест из 6 образцов швов. Разница в несколько процентных пунктов между прогонами лежит внутри доверительных интервалов.</li>
    <li>Главный вопрос проекта — помогают ли синтетические дефекты из диффузионной модели находить редкие классы — относится к этапам 2–3 и здесь не проверяется.</li>
  </ol>
</section>

<footer>
  <div>Данные: Totino B., Spagnolo F., Perri S. RIAWELC, ICMECE 2022; Perri S. et al., Manufacturing Letters, 2023. data_version {card["data_version"]}.</div>
  <div>Код: github.com/00monaliza/WeldSight</div>
</footer>
</main>
"""
OUT.write_text(html)
print(OUT, f"{len(html) / 1024:.0f} KB")
