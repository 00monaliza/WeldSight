function mulberry(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Procedural weld radiograph with planted defects, for trying the UI without a film. */
export function makeSample(seed = Date.now() % 100000): HTMLCanvasElement {
  const w = 960, h = 360, r = mulberry(seed);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d")!;
  const base = g.createLinearGradient(0, 0, 0, h);
  base.addColorStop(0, "#3c3f3f"); base.addColorStop(0.35, "#555958"); base.addColorStop(0.5, "#6c706e");
  base.addColorStop(0.65, "#555958"); base.addColorStop(1, "#36393a");
  g.fillStyle = base;
  g.fillRect(0, 0, w, h);
  const cy = h / 2, bead = g.createLinearGradient(0, cy - 62, 0, cy + 62);
  bead.addColorStop(0, "#ffffff00"); bead.addColorStop(0.25, "#9a9f9c66"); bead.addColorStop(0.5, "#b4b9b588");
  bead.addColorStop(0.75, "#9a9f9c66"); bead.addColorStop(1, "#ffffff00");
  g.fillStyle = bead;
  g.fillRect(0, cy - 62, w, 124);
  g.globalAlpha = 0.07;
  g.fillStyle = "#fff";
  for (let x = 0; x < w; x += 11) { g.beginPath(); g.ellipse(x, cy + Math.sin(x / 30) * 3, 5, 52, 0, 0, 7); g.fill(); }
  g.globalAlpha = 1;
  const dark = (a: number) => `rgba(8,10,10,${a})`;
  for (let i = 0; i < 6; i++) { // porosity
    const x = 90 + r() * (w - 180), y = cy + (r() - 0.5) * 70, rad = 3 + r() * 5;
    g.fillStyle = dark(0.62); g.shadowColor = dark(0.6); g.shadowBlur = 4;
    g.beginPath(); g.arc(x, y, rad, 0, 7); g.fill();
  }
  for (let i = 0; i < 2; i++) { // inclusion
    const x = 160 + r() * (w - 320), y = cy + (r() - 0.5) * 40;
    g.fillStyle = dark(0.5); g.shadowBlur = 3; g.beginPath();
    for (let k = 0; k < 9; k++) { const a = (k / 9) * 6.283, rr = 9 + r() * 9; g.lineTo(x + Math.cos(a) * rr * 1.5, y + Math.sin(a) * rr * 0.8); }
    g.fill();
  }
  { // crack
    let x = 200 + r() * 360, y = cy - 48;
    const a = 1.2 + r() * 0.4;
    g.strokeStyle = dark(0.7); g.lineWidth = 2.2; g.shadowBlur = 2; g.beginPath(); g.moveTo(x, y);
    for (let k = 0; k < 16; k++) { x += Math.cos(a + (r() - 0.5) * 1.1) * 6; y += Math.sin(a + (r() - 0.5) * 1.1) * 6.2; g.lineTo(x, y); }
    g.stroke();
  }
  { // lack of penetration: faint streak along the root, deliberately low contrast
    const x0 = 420 + r() * 220, y = cy + 22;
    g.strokeStyle = dark(0.2); g.lineWidth = 3; g.shadowBlur = 3; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + 190, y + (r() - 0.5) * 2); g.stroke();
  }
  g.shadowBlur = 0;
  const img = g.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (r() + r() + r() - 1.5) * 16, v = Math.max(0, Math.min(255, d[i] + n));
    d[i] = d[i + 1] = d[i + 2] = v;
  }
  g.putImageData(img, 0, 0);
  return c;
}
