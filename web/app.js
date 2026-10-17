"use strict";

/* ===================================================================
   WeldSight web UI. No build step, no dependencies.
   Engines: "demo" (heuristic, in-browser) and "api" (POST /predict).
   =================================================================== */

const CLASSES = [
  { id: "porosity",            color: "#3dd9ff" },
  { id: "inclusion",           color: "#ff5fd2" },
  { id: "crack",               color: "#ff4d4d" },
  { id: "undercut",            color: "#ffe14d" },
  { id: "lack_of_fusion",      color: "#ff9a3d" },
  { id: "lack_of_penetration", color: "#6dff8a" },
];
const MAX_DIM = 1024;
// demo detector tuning (score z is in robust sigmas of the high-pass residual)
const DEMO = { gate: 2.5, offset: 5.2, scale: 0.8, jitter: 0.9, lo: 0.2, minArea: 18, minPeak: 0.35 };
const LN2 = Math.LN2;
const $ = (id) => document.getElementById(id);
const pretty = (s) => { s = s.replace(/_/g, " "); return s[0].toUpperCase() + s.slice(1); };
const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const COLORS = CLASSES.map((c) => hex2rgb(c.color));

const S = {
  w: 0, h: 0, name: "",
  labels: null,            // Uint8Array, 0 = background, k = CLASSES[k-1]
  maps: { pe: null, mi: null },   // Float32Array 0..1
  defects: [],             // {cls, prob, bbox:[x,y,w,h], unc:{pe,mi}, suspect}
  active: -1,
  busy: false,
};

/* ------------------------------ tabs ------------------------------ */
function showTab() {
  const t = (location.hash || "#inspect").slice(1);
  const name = ["inspect", "results", "api"].includes(t) ? t : "inspect";
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== "view-" + name));
  document.querySelectorAll(".tabs a").forEach((a) => a.setAttribute("aria-selected", a.dataset.tab === name));
}
window.addEventListener("hashchange", showTab);

/* ---------------------------- legend ------------------------------ */
function renderLegend() {
  const counts = {};
  S.defects.forEach((d) => { if (!d.suspect) counts[d.cls] = (counts[d.cls] || 0) + 1; });
  $("legend").innerHTML = CLASSES.map((c) =>
    `<li><i style="background:${c.color}"></i><span>${pretty(c.id)}</span><span>${counts[c.id] || 0}</span></li>`).join("");
}

/* ---------------------- synthetic sample film --------------------- */
function mulberry(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function makeSample(seed = Date.now() % 100000) {
  const w = 960, h = 360, r = mulberry(seed);
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d");
  const base = g.createLinearGradient(0, 0, 0, h);
  base.addColorStop(0, "#3c3f3f"); base.addColorStop(.35, "#555958"); base.addColorStop(.5, "#6c706e");
  base.addColorStop(.65, "#555958"); base.addColorStop(1, "#36393a");
  g.fillStyle = base; g.fillRect(0, 0, w, h);
  // weld bead: thicker metal = brighter on a radiograph
  const cy = h / 2, bead = g.createLinearGradient(0, cy - 62, 0, cy + 62);
  bead.addColorStop(0, "#ffffff00"); bead.addColorStop(.25, "#9a9f9c66"); bead.addColorStop(.5, "#b4b9b588");
  bead.addColorStop(.75, "#9a9f9c66"); bead.addColorStop(1, "#ffffff00");
  g.fillStyle = bead; g.fillRect(0, cy - 62, w, 124);
  g.globalAlpha = .07; g.fillStyle = "#fff";
  for (let x = 0; x < w; x += 11) { g.beginPath(); g.ellipse(x, cy + Math.sin(x / 30) * 3, 5, 52, 0, 0, 7); g.fill(); }
  g.globalAlpha = 1;
  const dark = (a) => `rgba(8,10,10,${a})`;
  // porosity
  for (let i = 0; i < 6; i++) {
    const x = 90 + r() * (w - 180), y = cy + (r() - .5) * 70, rad = 3 + r() * 5;
    g.fillStyle = dark(.62); g.shadowColor = dark(.6); g.shadowBlur = 4;
    g.beginPath(); g.arc(x, y, rad, 0, 7); g.fill();
  }
  // inclusion
  for (let i = 0; i < 2; i++) {
    const x = 160 + r() * (w - 320), y = cy + (r() - .5) * 40;
    g.fillStyle = dark(.5); g.shadowBlur = 3; g.beginPath();
    for (let k = 0; k < 9; k++) { const a = k / 9 * 6.283, rr = 9 + r() * 9; g.lineTo(x + Math.cos(a) * rr * 1.5, y + Math.sin(a) * rr * .8); }
    g.fill();
  }
  // crack: jagged, roughly across the bead
  { let x = 200 + r() * 360, y = cy - 48, a = 1.2 + r() * .4;
    g.strokeStyle = dark(.7); g.lineWidth = 2.2; g.shadowBlur = 2; g.beginPath(); g.moveTo(x, y);
    for (let k = 0; k < 16; k++) { x += Math.cos(a + (r() - .5) * 1.1) * 6; y += Math.sin(a + (r() - .5) * 1.1) * 6.2; g.lineTo(x, y); }
    g.stroke(); }
  // lack of penetration: faint straight streak along the root, deliberately low contrast
  { const x0 = 420 + r() * 220, y = cy + 22;
    g.strokeStyle = dark(.2); g.lineWidth = 3; g.shadowBlur = 3; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + 190, y + (r() - .5) * 2); g.stroke(); }
  g.shadowBlur = 0;
  // film grain
  const img = g.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (r() + r() + r() - 1.5) * 16, v = Math.max(0, Math.min(255, d[i] + n));
    d[i] = d[i + 1] = d[i + 2] = v;
  }
  g.putImageData(img, 0, 0);
  return c;
}

/* ------------------------- demo engine ---------------------------- */
function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h), d = 2 * r + 1;
  const cl = (v, m) => (v < 0 ? 0 : v > m ? m : v);
  for (let y = 0; y < h; y++) {
    const o = y * w; let s = 0;
    for (let x = -r; x <= r; x++) s += src[o + cl(x, w - 1)];
    for (let x = 0; x < w; x++) { tmp[o + x] = s / d; s += src[o + cl(x + r + 1, w - 1)] - src[o + cl(x - r, w - 1)]; }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[cl(y, h - 1) * w + x];
    for (let y = 0; y < h; y++) { out[y * w + x] = s / d; s += tmp[cl(y + r + 1, h - 1) * w + x] - tmp[cl(y - r, h - 1) * w + x]; }
  }
  return out;
}
const H = (p) => { p = Math.min(1 - 1e-6, Math.max(1e-6, p)); return -(p * Math.log(p) + (1 - p) * Math.log(1 - p)); };
const sigmoid = (v) => 1 / (1 + Math.exp(-v));
const gauss = () => { let s = 0; for (let i = 0; i < 4; i++) s += Math.random(); return (s - 2) * 1.73; };

function demoInfer(gray, w, h, T) {
  const n = w * h;
  const bg = boxBlur(gray, w, h, 14), sm = boxBlur(gray, w, h, 1);
  const z = new Float32Array(n), sample = [];
  for (let i = 0; i < n; i++) z[i] = bg[i] - sm[i];
  for (let i = 0; i < n; i += 7) sample.push(z[i]);
  sample.sort((a, b) => a - b);
  const med = sample[sample.length >> 1];
  const dev = sample.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
  const sd = Math.max(0.5, 1.4826 * dev[dev.length >> 1]);
  for (let i = 0; i < n; i++) z[i] = (z[i] - med) / sd;

  // T stochastic passes (noise on the score stands in for dropout masks)
  const meanP = new Float32Array(n), pe = new Float32Array(n), mi = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (z[i] < DEMO.gate) continue;
    let sp = 0, se = 0;
    for (let t = 0; t < T; t++) { const p = sigmoid((z[i] + gauss() * DEMO.jitter - DEMO.offset) / DEMO.scale); sp += p; se += H(p); }
    const m = sp / T; meanP[i] = m; pe[i] = H(m) / LN2; mi[i] = Math.min(1, Math.max(0, (H(m) - se / T) / 0.3));
  }
  return { meanP, pe, mi };
}

function components(meanP, w, h, lo) {
  const seen = new Uint8Array(w * h), comps = [], stack = [];
  for (let s = 0; s < w * h; s++) {
    if (seen[s] || meanP[s] < lo) continue;
    const px = []; let x0 = w, y0 = h, x1 = 0, y1 = 0;
    seen[s] = 1; stack.push(s);
    while (stack.length) {
      const i = stack.pop(); px.push(i);
      const x = i % w, y = (i / w) | 0;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (x > 0 && !seen[i - 1] && meanP[i - 1] >= lo) { seen[i - 1] = 1; stack.push(i - 1); }
      if (x < w - 1 && !seen[i + 1] && meanP[i + 1] >= lo) { seen[i + 1] = 1; stack.push(i + 1); }
      if (y > 0 && !seen[i - w] && meanP[i - w] >= lo) { seen[i - w] = 1; stack.push(i - w); }
      if (y < h - 1 && !seen[i + w] && meanP[i + w] >= lo) { seen[i + w] = 1; stack.push(i + w); }
    }
    if (px.length >= DEMO.minArea) comps.push({ px, bbox: [x0, y0, x1 - x0 + 1, y1 - y0 + 1] });
  }
  return comps;
}

function classify(c, w, h) {
  const [, y, bw, bh] = c.bbox, area = c.px.length;
  let mx = 0, my = 0;
  for (const i of c.px) { mx += i % w; my += (i / w) | 0; }
  mx /= area; my /= area;
  let sxx = 0, syy = 0, sxy = 0;
  for (const i of c.px) { const dx = (i % w) - mx, dy = ((i / w) | 0) - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  sxx /= area; syy /= area; sxy /= area;
  const tr = sxx + syy, det = sxx * syy - sxy * sxy, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(1e-6, tr / 2 - disc);
  const elong = Math.sqrt(l1 / l2), angle = Math.abs(0.5 * Math.atan2(2 * sxy, sxx - syy)) * 180 / Math.PI;  // 0 = horizontal
  const horizontal = angle < 20, fill = area / (bw * bh);
  if (elong > 5) return horizontal ? "lack_of_penetration" : "crack";
  if (elong > 3 && horizontal) return (y < h * .3 || y > h * .7) ? "undercut" : "lack_of_fusion";
  if (area < 160 && fill > .45) return "porosity";
  return "inclusion";
}

function meanOver(arr, idx) {
  let s = 0; for (const i of idx) s += arr[i]; return s / idx.length;
}

function runDemo(gray, w, h, T) {
  const { meanP, pe, mi } = demoInfer(gray, w, h, T);
  const labels = new Uint8Array(w * h), defects = [];
  for (const c of components(meanP, w, h, 0.2)) {
    let sp = 0, peak = 0; for (const i of c.px) { sp += meanP[i]; if (meanP[i] > peak) peak = meanP[i]; }
    if (peak < DEMO.minPeak) continue;
    const prob = sp / c.px.length, cls = classify(c, w, h), suspect = prob < 0.5;
    if (!suspect) { const k = CLASSES.findIndex((x) => x.id === cls) + 1; for (const i of c.px) labels[i] = k; }
    defects.push({ cls, prob, bbox: c.bbox, suspect, unc: { pe: meanOver(pe, c.px), mi: meanOver(mi, c.px) } });
  }
  return { labels, maps: { pe: halo(pe, w, h), mi: halo(mi, w, h) }, defects };
}

function halo(map, w, h) {   // display only: spread the per-pixel score so thin regions are visible
  const b = boxBlur(map, w, h, 4);
  for (let i = 0; i < b.length; i++) b[i] = Math.min(1, Math.max(map[i], b[i] * 3));
  return b;
}

/* ---------------------------- API engine -------------------------- */
async function pngToChannel(b64, w, h) {
  const bmp = await createImageBitmap(await (await fetch("data:image/png;base64," + b64)).blob());
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d"); g.imageSmoothingEnabled = false; g.drawImage(bmp, 0, 0, w, h);
  const d = g.getImageData(0, 0, w, h).data, out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4];
  return out;
}

async function runApi(canvas, T) {
  const base = $("apiUrl").value.replace(/\/+$/, "");
  const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
  const fd = new FormData(); fd.append("image", blob, S.name || "film.png"); fd.append("mc_samples", String(T));
  const resp = await fetch(base + "/predict", { method: "POST", body: fd });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${(await resp.text()).slice(0, 160)}`);
  const j = await resp.json(), w = canvas.width, h = canvas.height;
  const names = j.classes || CLASSES.map((c) => c.id);
  const remap = names.map((n) => CLASSES.findIndex((c) => c.id === n) + 1);
  const raw = await pngToChannel(j.mask_png, w, h), labels = new Uint8Array(w * h);
  for (let i = 0; i < labels.length; i++) labels[i] = raw[i] ? remap[raw[i] - 1] || 0 : 0;
  const mk = async (b64) => { if (!b64) return new Float32Array(w * h); const u = await pngToChannel(b64, w, h), f = new Float32Array(w * h); for (let i = 0; i < f.length; i++) f[i] = u[i] / 255; return f; };
  const sx = w / (j.width || w), sy = h / (j.height || h);
  const defects = (j.defects || []).map((d) => ({
    cls: d.cls, prob: d.prob, suspect: d.prob < 0.5,
    bbox: [d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy].map(Math.round),
    unc: { pe: d.uncertainty?.pe ?? 0, mi: d.uncertainty?.mi ?? 0 },
  }));
  return { labels, maps: { pe: await mk(j.maps?.pe), mi: await mk(j.maps?.mi) }, defects };
}

/* ------------------------------ load ------------------------------ */
const cBase = $("cBase"), cOver = $("cOver"), cHi = $("cHi");
const gBase = cBase.getContext("2d", { willReadFrequently: true }), gOver = cOver.getContext("2d"), gHi = cHi.getContext("2d");

function toWorkingCanvas(src) {
  const sw = src.naturalWidth || src.width, sh = src.naturalHeight || src.height, k = Math.min(1, MAX_DIM / Math.max(sw, sh));
  const w = Math.max(32, Math.round(sw * k)), h = Math.max(32, Math.round(sh * k));
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").drawImage(src, 0, 0, w, h);
  return c;
}

async function loadSource(src, name) {
  if (S.busy) return;
  S.busy = true; S.name = name;
  const work = toWorkingCanvas(src), { width: w, height: h } = work;
  S.w = w; S.h = h;
  for (const c of [cBase, cOver, cHi]) { c.width = w; c.height = h; }
  // grayscale base (radiographs are single channel; keeps colours free for the overlay)
  const img = work.getContext("2d").getImageData(0, 0, w, h), d = img.data, gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) { const v = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]; gray[i] = v; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255; }
  gBase.putImageData(img, 0, 0);
  $("canvasWrap").hidden = false; $("empty").hidden = true; $("film").classList.add("loaded");
  $("canvasWrap").style.aspectRatio = `${w} / ${h}`;
  $("rsize").textContent = `${w}×${h}px · ${name}`;
  $("scan").hidden = false;
  setVerdict("idle", "Analysing…", `${$("engine").value === "api" ? "Waiting for the API" : "Running " + $("mc").value + " stochastic passes"}`);
  gOver.clearRect(0, 0, w, h); gHi.clearRect(0, 0, w, h);
  await new Promise((r) => setTimeout(r, 40));   // let the sweep paint before the CPU-bound pass
  try {
    const T = +$("mc").value;
    const res = $("engine").value === "api" ? await runApi(work, T) : runDemo(gray, w, h, T);
    S.labels = res.labels; S.maps = res.maps; S.defects = res.defects; S.active = -1;
    $("btnPng").disabled = $("btnJson").disabled = false;
    refresh();
  } catch (e) {
    console.error(e);
    S.defects = []; S.labels = null; renderDefects(); renderLegend();
    setVerdict("idle", "Engine error", String(e.message || e));
  } finally {
    $("scan").hidden = true; S.busy = false;
  }
}

function loadFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  const url = URL.createObjectURL(f), im = new Image();
  im.onload = () => { loadSource(im, f.name); URL.revokeObjectURL(url); };
  im.onerror = () => setVerdict("idle", "Cannot read file", "Use PNG, JPEG or another browser-readable image.");
  im.src = url;
}

/* --------------------------- rendering ---------------------------- */
const LUT = (() => {   // black → violet → orange → pale yellow
  const stops = [[0, [0, 0, 0]], [.35, [90, 20, 120]], [.7, [240, 100, 30]], [1, [255, 240, 150]]], out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 1; while (k < stops.length - 1 && t > stops[k][0]) k++;
    const [t0, c0] = stops[k - 1], [t1, c1] = stops[k], f = (t - t0) / (t1 - t0);
    for (let j = 0; j < 3; j++) out[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * f;
  }
  return out;
})();

function score() { return $("score").value; }
function thr() { return +$("thr").value; }
const defUnc = (d) => d.unc[score()];

function drawOverlay() {
  const { w, h, labels } = S; if (!labels) return;
  const img = gOver.createImageData(w, h), d = img.data, a = +$("opacity").value, map = S.maps[score()];
  if ($("lUnc").checked && map) {
    for (let i = 0; i < w * h; i++) {
      const u = map[i]; if (u < 0.03) continue;
      const k = Math.min(255, (u * 255) | 0) * 3;
      d[i * 4] = LUT[k]; d[i * 4 + 1] = LUT[k + 1]; d[i * 4 + 2] = LUT[k + 2]; d[i * 4 + 3] = Math.min(255, 255 * a * Math.min(1, u * 1.6));
    }
  }
  if ($("lSeg").checked) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, c = labels[i]; if (!c) continue;
      const edge = x === 0 || y === 0 || x === w - 1 || y === h - 1 || labels[i - 1] !== c || labels[i + 1] !== c || labels[i - w] !== c || labels[i + w] !== c;
      const col = COLORS[c - 1];
      d[i * 4] = col[0]; d[i * 4 + 1] = col[1]; d[i * 4 + 2] = col[2]; d[i * 4 + 3] = edge ? 255 : 255 * a * 0.7;
    }
  }
  gOver.putImageData(img, 0, 0);
}

function drawHighlight() {
  gHi.clearRect(0, 0, S.w, S.h);
  if (!$("lBox").checked) return;
  const lw = Math.max(1.5, S.w / 600);
  S.defects.forEach((d, i) => {
    const [x, y, w, h] = d.bbox, col = CLASSES.find((c) => c.id === d.cls).color, on = i === S.active, pad = 4 + lw;
    gHi.lineWidth = on ? lw * 2 : lw; gHi.strokeStyle = d.suspect ? "#ffffffcc" : col;
    gHi.setLineDash(d.suspect ? [6, 4] : []); gHi.globalAlpha = on ? 1 : (S.active >= 0 ? .35 : .85);
    gHi.strokeRect(x - pad, y - pad, w + 2 * pad, h + 2 * pad);
  });
  gHi.globalAlpha = 1; gHi.setLineDash([]);
}

function filmScore() { return S.defects.reduce((m, d) => Math.max(m, defUnc(d)), 0); }

function setVerdict(state, main, sub) {
  const v = $("verdict"); v.dataset.state = state; $("vMain").textContent = main; $("vSub").textContent = sub;
}

function updateVerdict() {
  const sc = filmScore(), t = thr(), real = S.defects.filter((d) => !d.suspect);
  $("meter").hidden = $("meterCap").hidden = false;
  $("meterFill").style.width = Math.min(100, sc * 100) + "%"; $("meterThr").style.left = t * 100 + "%";
  $("mVal").textContent = sc.toFixed(2); $("mThr").textContent = t.toFixed(2);
  if (sc >= t) {
    const worst = S.defects.reduce((a, b) => (defUnc(b) > defUnc(a) ? b : a));
    setVerdict("refer", "Send to inspector", `Uncertainty ${sc.toFixed(2)} ≥ ${t.toFixed(2)} on ${worst.suspect ? "a sub-threshold " : "a "}${pretty(worst.cls).toLowerCase()} region. Do not rely on the model alone.`);
  } else if (real.length) {
    setVerdict("flag", `${real.length} defect${real.length > 1 ? "s" : ""} flagged`, "Model is confident enough to handle this film automatically. Spot-check per your QA policy.");
  } else {
    setVerdict("clear", "No defects found", "Nothing above the detection threshold, and nothing uncertain enough to refer.");
  }
}

function renderDefects() {
  const ul = $("defects"), t = thr();
  $("findCount").textContent = S.defects.length;
  if (!S.defects.length) { ul.innerHTML = `<li class="none">${S.labels ? "No candidate regions." : "Nothing analysed yet."}</li>`; return; }
  const order = S.defects.map((d, i) => i).sort((a, b) => defUnc(S.defects[b]) - defUnc(S.defects[a]));
  ul.innerHTML = order.map((i) => {
    const d = S.defects[i], col = CLASSES.find((c) => c.id === d.cls).color, u = defUnc(d);
    return `<li><div class="defect ${d.suspect ? "suspect" : ""} ${u >= t ? "hot" : ""} ${i === S.active ? "active" : ""}" data-i="${i}" tabindex="0" style="color:${col}">
      <span class="sw" style="background:${col}"></span>
      <span class="nm">${pretty(d.cls)}${d.suspect ? " <em>· suspect</em>" : ""}</span>
      <span class="pr">${(d.prob * 100).toFixed(0)}%</span>
      <span class="ubar"><i style="width:${Math.min(100, u * 100)}%"></i><s style="left:${t * 100}%"></s></span>
      <span class="meta">u ${u.toFixed(2)} · ${d.bbox[2]}×${d.bbox[3]}px @ ${d.bbox[0]},${d.bbox[1]}</span></div></li>`;
  }).join("");
}

function refresh() { drawOverlay(); drawHighlight(); renderDefects(); renderLegend(); updateVerdict(); }

/* ------------------------------ events ---------------------------- */
$("file").onchange = (e) => loadFile(e.target.files[0]);
$("file2").onchange = (e) => loadFile(e.target.files[0]);
$("btnSample").onclick = () => loadSource(makeSample(), "synthetic_sample.png");

const film = $("film");
["dragenter", "dragover"].forEach((ev) => film.addEventListener(ev, (e) => { e.preventDefault(); film.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => film.addEventListener(ev, (e) => { e.preventDefault(); film.classList.remove("over"); }));
film.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));

["lSeg", "lUnc", "opacity"].forEach((id) => $(id).addEventListener("input", drawOverlay));
$("lBox").addEventListener("input", drawHighlight);
$("score").addEventListener("change", () => S.labels && refresh());
$("thr").addEventListener("input", () => { $("thrVal").textContent = (+$("thr").value).toFixed(2); if (S.labels) { renderDefects(); updateVerdict(); } });
$("mc").addEventListener("input", () => { $("tVal").textContent = $("mc").value; });
$("mc").addEventListener("change", () => {  // re-run on the same film
  if (S.labels && !S.busy) loadSource(cBase, S.name);
});
$("engine").addEventListener("change", () => {
  const api = $("engine").value === "api";
  $("apiRow").hidden = !api; $("demoBanner").hidden = api;
  $("engineChip").dataset.mode = api ? "api" : "demo"; $("engineChip").querySelector("span").textContent = api ? "API engine" : "Demo engine";
  if (S.labels && !S.busy) loadSource(cBase, S.name);
});

$("defects").addEventListener("click", (e) => {
  const el = e.target.closest(".defect"); if (!el) return;
  S.active = S.active === +el.dataset.i ? -1 : +el.dataset.i; drawHighlight(); renderDefects();
});
$("defects").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.target.click(); } });

$("canvasWrap").addEventListener("mousemove", (e) => {
  if (!S.labels) return;
  const r = cBase.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) / r.width * S.w), y = Math.floor((e.clientY - r.top) / r.height * S.h);
  if (x < 0 || y < 0 || x >= S.w || y >= S.h) return;
  const i = y * S.w + x, c = S.labels[i];
  $("rx").textContent = x; $("ry").textContent = y;
  $("rc").textContent = c ? pretty(CLASSES[c - 1].id) : "bg";
  $("ru").textContent = (S.maps[score()]?.[i] ?? 0).toFixed(2);
});

function download(name, blob) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
$("btnPng").onclick = () => {
  const c = document.createElement("canvas"); c.width = S.w; c.height = S.h;
  const g = c.getContext("2d"); g.drawImage(cBase, 0, 0); g.drawImage(cOver, 0, 0); g.drawImage(cHi, 0, 0);
  c.toBlob((b) => download(S.name.replace(/\.\w+$/, "") + "_weldsight.png", b));
};
$("btnJson").onclick = () => {
  const rep = {
    film: S.name, width: S.w, height: S.h, engine: $("engine").value, mc_samples: +$("mc").value,
    uncertainty_score: score(), refer_threshold: thr(), film_uncertainty: +filmScore().toFixed(4),
    decision: filmScore() >= thr() ? "refer_to_inspector" : (S.defects.some((d) => !d.suspect) ? "defects_flagged" : "clear"),
    note: $("engine").value === "demo" ? "Produced by the heuristic demo engine, not the trained model." : undefined,
    defects: S.defects,
  };
  download(S.name.replace(/\.\w+$/, "") + "_weldsight.json", new Blob([JSON.stringify(rep, null, 2)], { type: "application/json" }));
};

/* ----------------------- stage 1 results tab ---------------------- */
function renderResults() {
  const rows = [
    { n: "Crack", rec: 57.0, lo: 52.6, hi: 66.7, sup: "977 patches · 2 specimens", p: 67.0, f1: 0.62, auc: 0.86 },
    { n: "Porosity", rec: 93.1, lo: 86.8, hi: 97.2, sup: "936 patches · 4 specimens", p: 93.5, f1: 0.93, auc: 0.99 },
    { n: "Lack of penetration", rec: 19.0, lo: 2.3, hi: 42.8, sup: "658 patches · 3 specimens", p: 64.4, f1: 0.29, auc: 0.72 },
  ];
  $("ciTable").innerHTML = rows.map((r) => `
    <div class="ci-row"><div class="nm">${r.n}<small>${r.sup}</small></div>
      <div class="ci-track"><span class="ci-bar" style="left:${r.lo}%;width:${r.hi - r.lo}%"></span><span class="ci-dot" style="left:${r.rec}%"></span></div>
      <div class="ci-val">${r.rec.toFixed(1)}</div>
      <div class="ci-extra"><span>CI <b>${r.lo}–${r.hi}</b></span><span>precision <b>${r.p}</b></span><span>F1 <b>${r.f1}</b></span><span>AUROC <b>${r.auc}</b></span></div></div>`).join("") +
    `<div class="ci-scale" style="margin-top:6px"><span></span><div><span>0 %</span><span>50</span><span>100 %</span></div><span></span></div>`;

  // referral chart: only the four points published in the README
  const X = [0, 10, 20, 30], model = [60.4, 65.4, 71.2, 76.8], rnd = [60.4, 64.1, 67.9, 71.5], orc = [60.4, 69.0, 75.9, 81.5];
  const L = 44, R = 462, T = 28, B = 256, y0 = 55, y1 = 85;
  const sx = (v) => L + (v / 30) * (R - L), sy = (v) => B - ((v - y0) / (y1 - y0)) * (B - T);
  const line = (a, dash, col, wd) => `<polyline fill="none" stroke="${col}" stroke-width="${wd}" ${dash ? `stroke-dasharray="${dash}"` : ""} points="${X.map((x, i) => `${sx(x)},${sy(a[i])}`).join(" ")}"/>`;
  let s = "";
  for (let v = y0; v <= y1; v += 5) s += `<line x1="${L}" x2="${R}" y1="${sy(v)}" y2="${sy(v)}" stroke="#23292c"/><text x="${L - 8}" y="${sy(v) + 3}" text-anchor="end">${v}</text>`;
  X.forEach((x) => (s += `<text x="${sx(x)}" y="${B + 18}" text-anchor="middle">${x} %</text>`));
  s += `<text x="${(L + R) / 2}" y="${B + 36}" text-anchor="middle">films referred to an inspector</text>`;
  s += `<text x="${L - 8}" y="${T - 14}" text-anchor="end">recall %</text>`;
  s += line(rnd, "5 4", "#6b7270", 1.5) + line(orc, "2 4", "#e9e7e0", 1.5) + line(model, "", "#ffb347", 2.5);
  model.forEach((v, i) => (s += `<rect x="${sx(X[i]) - 4}" y="${sy(v) - 4}" width="8" height="8" fill="#ffb347" transform="rotate(45 ${sx(X[i])} ${sy(v)})"/>`));
  s += `<text x="${R + 10}" y="${sy(76.8) + 3}" style="fill:#ffb347">76.8</text><text x="${R + 10}" y="${sy(71.5) + 3}">71.5</text><text x="${R + 10}" y="${sy(81.5) + 3}" style="fill:#e9e7e0">81.5</text>`;
  $("refChart").innerHTML = s;

  const au = [["Predictive entropy", 0.82], ["Expected entropy", 0.82], ["Max-prob", 0.80], ["Mutual information", 0.54]];
  $("aurocBars").innerHTML = au.map(([n, v]) => `<div class="bar ${n.startsWith("Mutual") ? "mi" : ""}"><span>${n}</span><div class="track"><i style="width:${v * 100}%"></i><s></s></div><span>${v.toFixed(2)}</span></div>`).join("");
}

/* ------------------------------- init ----------------------------- */
renderLegend();
renderResults();
showTab();
window.__weldsight = { S, DEMO, loadSource, makeSample };   // for tests / console poking
