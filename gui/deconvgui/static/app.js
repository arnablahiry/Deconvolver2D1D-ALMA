/* Deconvolver 2D1D front end. Plain JS, no build step: edit and reload. */
"use strict";

// ------------------------------------------------------------------ utils --
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) el.append(c.nodeType ? c : String(c));
  return el;
}

// Every request has a timeout. A port forward (VS Code, ssh -L) can leave a request
// hanging while the server restarts; the browser allows only ~6 connections per host,
// so a few hung polls would block every later request, page reloads included.
async function api(path, opts = {}) {
  const { timeout = 120000, ...o } = opts;
  if (o.body && typeof o.body !== "string") {
    o.body = JSON.stringify(o.body);
    o.headers = { "content-type": "application/json" };
  }
  o.signal = AbortSignal.timeout(timeout);
  const r = await fetch(path, o);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch { msg = await r.text().catch(() => msg); }
    throw new Error(msg);
  }
  return r.json();
}

async function getArray(url) {
  const r = await fetch(url, { signal: AbortSignal.timeout(300000) });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const [ny, nx] = r.headers.get("X-Shape").split(",").map(Number);
  const it = r.headers.get("X-Iter"), fr = r.headers.get("X-Fraction"), msg = r.headers.get("X-Message");
  return { data: new Float32Array(await r.arrayBuffer()), ny, nx, iter: it === null ? null : +it,
    // a live job frame also carries the progress of its own iteration
    fraction: fr === null || fr === "" ? null : +fr, message: msg === null ? null : decodeURIComponent(msg) };
}

// Channel stepping. A slider fires dozens of input events a second and every step needs one
// array per panel (~300 kB each at 288²); requesting them all floods the connection (a port
// forward is the slow part), so the newest frame would arrive last. Instead: one load in flight,
// always for the slider's latest position (positions passed over meanwhile are skipped)...
function latestOnly(run) {
  let busy = false, again = false;
  return async () => {
    if (busy) { again = true; return; }
    busy = true;
    try { do { again = false; await run(); } while (again); } finally { busy = false; }
  };
}
// ...and arrays cached for the life of a view (LRU, ~150 MB), so revisiting a channel is instant.
// Caching the promise also lets a prefetch and a real load of the same channel share one request.
function arrayCache(imsize) {
  const max = Math.max(24, Math.floor(150e6 / (4 * (imsize || 256) ** 2))), m = new Map();
  return (url) => {
    let p = m.get(url);
    if (p) { m.delete(url); m.set(url, p); return p; }
    p = getArray(url); p.catch(() => m.delete(url));
    m.set(url, p);
    if (m.size > max) m.delete(m.keys().next().value);
    return p;
  };
}

const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "–"
  : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(d - 1) : (+v).toPrecision(d);
const tick = (v) => {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace("e+", "e");
  return String(+v.toPrecision(4));
};
const ago = (t) => { const s = Date.now() / 1000 - t; return s < 90 ? `${s | 0}s ago` : s < 5400 ? `${(s / 60) | 0} min ago` : s < 129600 ? `${(s / 3600) | 0} h ago` : `${(s / 86400) | 0} d ago`; };

function toast(msg, isErr = false) {
  const t = h("div", { class: "toast" + (isErr ? " err" : "") }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), isErr ? 9000 : 3500);
}

// ------------------------------------------------------------- colormaps --
const LUTS = {};
async function lut(name) {
  if (!LUTS[name]) LUTS[name] = await api(`/api/colormap/${name}`);
  return LUTS[name];
}
const CMAPS = ["inferno", "magma", "viridis", "cividis", "gray", "RdBu_r", "cubehelix"];

const EDGE_FRAC = 0.02;   // fraction of the field hidden on each side in image displays (edge artifacts)
const edgePx = (ny, nx) => Math.max(2, Math.round(EDGE_FRAC * Math.min(ny, nx)));

function stretchFn(mode) {
  if (mode === "asinh") { const a = 0.1, n = Math.asinh(1 / a); return (t) => Math.asinh(t / a) / n; }
  if (mode === "sqrt") return Math.sqrt;
  return (t) => t;
}

function percentiles(data, lo, hi) {
  const n = data.length, step = Math.max(1, Math.floor(n / 60000));
  const s = [];
  for (let i = 0; i < n; i += step) if (Number.isFinite(data[i])) s.push(data[i]);
  s.sort((a, b) => a - b);
  if (!s.length) return [0, 1];
  const q = (p) => s[Math.min(s.length - 1, Math.max(0, Math.round(p / 100 * (s.length - 1))))];
  return [q(lo), q(hi)];
}

// --------------------------------------------------------- image panels --
class ImagePanel {
  // onRegion(box, panel): a box was drawn or moved; onDrag(box, panel): while drawing or moving
  // (default: preview on this panel); square: boxes are squares; canDraw(): whether drawing is
  // allowed right now. With canDraw, pressing inside the current box moves it instead of starting a
  // new one, and drawing also works on a cropped view (boxes stay inside what is shown).
  // edge: hide a small border (EDGE_FRAC of the field per side) where deconvolutions pile up edge artifacts
  // (default on, off for box-drawing panels, which must reach the edges); cbar: "bottom" | "right" | null
  // (with a colour bar the unit is its label, not in the title row); setBeam([bmaj″, bmin″, pa°]) draws the beam.
  constructor({ title, unit = "", onRegion = null, onDrag = null, square = false, canDraw = null, cell = null,
                edge = null, cbar = null }) {
    this.title = title; this.unit = unit; this.onRegion = onRegion; this.cell = cell;
    this.onDrag = onDrag; this.square = square; this.canDraw = canDraw;
    this.edge = edge === null ? !canDraw : edge;
    this.arr = null; this.crop = null; this.box = null; this.limits = null; this.beam = null;
    this.canvas = h("canvas");
    this.overlay = h("canvas", { style: "pointer-events:none" });
    this.readout = h("div", { class: "readout" }, " ");
    this.unitEl = h("span", { class: "punit" }, unit);
    this.titleEl = h("b", {}, title);
    this.tag = h("div", { class: "cv-tag" });   // e.g. the channel shown, in the bottom-right corner
    this.cv = h("div", { class: "cv" }, this.canvas, this.overlay, this.tag);
    this.cbarPos = cbar;
    let body = this.cv;
    if (cbar) {
      this.cbarCanvas = h("canvas", { class: "cbar-strip" });
      this.cbarTicks = h("div", { class: "cbar-ticks" });
      this.cbarUnit = h("div", { class: "cbar-unit" }, unit);
      this.cbarEl = h("div", { class: `cbar cbar-${cbar}` }, this.cbarCanvas, this.cbarTicks, this.cbarUnit);
      if (cbar === "right") body = h("div", { class: "cv-row" }, this.cv, this.cbarEl);
    }
    this.el = h("div", { class: "panel" + (cbar ? " has-cbar" : "") }, h("div", { class: "ptitle" }, this.titleEl, this.unitEl), body,
      cbar === "bottom" ? this.cbarEl : null, this.readout);
    this._mouse();
    new ResizeObserver(() => this.drawOverlay()).observe(this.cv);
  }
  setUnit(u) { this.unit = u; this.unitEl.textContent = u; if (this.cbarUnit) this.cbarUnit.textContent = u; }
  setTitle(t) { this.title = t; this.titleEl.textContent = t; }
  setTag(t) { this.tag.textContent = t; }
  set(arr) { this.arr = arr; this._auto = null; }
  view() {   // [x0, x1, y0, y1] inclusive, in data pixels
    if (!this.arr) return [0, 0, 0, 0];
    if (this.crop) return this.crop;
    const m = this.edge ? edgePx(this.arr.ny, this.arr.nx) : 0;
    return [m, this.arr.nx - 1 - m, m, this.arr.ny - 1 - m];
  }
  autoLimits(clip) {   // percentiles of what is shown (the cropped edges do not set the scale)
    if (!this.arr) return [0, 1];
    if (!this._auto || this._auto.clip !== clip) {
      let src = this.arr.data;
      if (this.crop || this.edge) src = this._cropData();
      const [lo, hi] = percentiles(src, 100 - clip, clip);
      this._auto = { clip, lo, hi };
    }
    return [this._auto.lo, this._auto.hi];
  }
  _cropData() {
    const [x0, x1, y0, y1] = this.view(), nx = this.arr.nx, out = [];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) out.push(this.arr.data[y * nx + x]);
    return out;
  }
  render(lutArr, mode, lo, hi) {
    if (!this.arr) { const c = this.canvas.getContext("2d"); c.clearRect(0, 0, this.canvas.width, this.canvas.height); return; }
    const [x0, x1, y0, y1] = this.view(), w = x1 - x0 + 1, hgt = y1 - y0 + 1, nx = this.arr.nx;
    this.canvas.width = w; this.canvas.height = hgt;
    this.cv.style.aspectRatio = `${w} / ${hgt}`;
    const ctx = this.canvas.getContext("2d"), img = ctx.createImageData(w, hgt), f = stretchFn(mode);
    const span = (hi - lo) || 1, d = this.arr.data;
    for (let r = 0; r < hgt; r++) {
      const y = y1 - r;                          // row 0 of the data is south: draw it at the bottom
      for (let c = 0; c < w; c++) {
        let t = (d[y * nx + x0 + c] - lo) / span;
        t = t < 0 ? 0 : t > 1 ? 1 : f(t);
        const k = (t * 255) | 0, rgb = lutArr[k], o = (r * w + c) * 4;
        img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    this.limits = [lo, hi];
    this.drawOverlay();
    if (this.cbarEl) this._drawCbar(lutArr, mode, lo, hi);
  }
  _drawCbar(lutArr, mode, lo, hi) {   // gradient linear in value (so the stretch shows), nice ticks, unit
    const n = 256, vert = this.cbarPos === "right", f = stretchFn(mode), c = this.cbarCanvas;
    c.width = vert ? 1 : n; c.height = vert ? n : 1;
    const ctx = c.getContext("2d"), img = ctx.createImageData(c.width, c.height);
    for (let i = 0; i < n; i++) {
      const rgb = lutArr[(f(i / (n - 1)) * 255) | 0], o = (vert ? n - 1 - i : i) * 4;
      img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    this.cbarTicks.innerHTML = "";
    const span = (hi - lo) || 1;
    for (const v of niceTicks(lo, hi, vert ? 5 : 4)) {
      const frac = (v - lo) / span;
      if (frac < -1e-6 || frac > 1 + 1e-6) continue;
      this.cbarTicks.append(h("span", { style: vert ? `bottom:${frac * 100}%` : `left:${frac * 100}%` }, tick(v)));
    }
    // a vertical bar's labels are positioned absolutely: size their column to the longest, so the unit clears them
    if (vert) this.cbarTicks.style.width = `calc(${Math.max(1, ...[...this.cbarTicks.children].map((e) => e.textContent.length))}ch + 14px)`;
    this.cbarUnit.textContent = this.unit;
  }
  setBox(box) { this.box = box; this.drawOverlay(); }
  setBeam(beam) { this.beam = beam && beam[0] > 0 && beam[1] > 0 ? beam : null; this.drawOverlay(); }
  // a map ({data, nx, ny}, on this panel's grid) outlined along its pixel edges at each of `levels` (value >= level
  // inside), the outer levels fainter; null removes it
  setContour(map, levels = [0.5]) {
    this.contour = map && this.arr && map.nx === this.arr.nx && map.ny === this.arr.ny ? map : null;
    this.contourLevels = levels; this.drawOverlay();
  }
  _drawContour(ctx, dpr) {
    const m = this.contour;
    if (!m || !this.arr) return;
    const [x0, x1, y0, y1] = this.view(), nx = m.nx, d = m.data, L = this.contourLevels || [0.5];
    ctx.save(); ctx.setLineDash([]); ctx.lineCap = "round"; ctx.lineWidth = 1.2 * dpr;
    L.forEach((lv, i) => {
      const path = new Path2D();
      const seg = (ax, ay, bx, by) => { const [p, q] = this._toScreen(ax, ay), [u, v] = this._toScreen(bx, by); path.moveTo(p, q); path.lineTo(u, v); };
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const v = d[y * nx + x] >= lv;
        if (x < x1 && v !== d[y * nx + x + 1] >= lv) seg(x + 1, y, x + 1, y + 1);
        if (y < y1 && v !== d[(y + 1) * nx + x] >= lv) seg(x, y + 1, x + 1, y + 1);
      }
      ctx.strokeStyle = `rgba(255,255,255,${(0.35 + 0.45 * (i + 1) / L.length).toFixed(2)})`; ctx.stroke(path);
    });
    ctx.restore();
  }
  // an element (e.g. inputs for the beam) shown in the bottom-left corner beside the beam ellipse, instead of its label
  setBeamControl(el) { this.beamCtl = el; el.classList.add("beam-ctl"); this.cv.append(el); this.drawOverlay(); }
  _drawBeam(ctx, dpr) {   // clean beam, to scale, in the bottom-left corner; "bmaj″ × bmin″, pa°" next to it
    if (!this.beam || !this.cell || !this.arr) return;
    const [x0, x1] = this.view(), W = this.overlay.width, H = this.overlay.height;
    const s = W / ((x1 - x0 + 1) * this.cell);                  // overlay px per arcsec
    const [bmaj, bmin, pa] = this.beam, a = Math.max(bmaj * s / 2, 1.5 * dpr), b = Math.max(bmin * s / 2, 1.5 * dpr);
    const rot = -(Math.PI / 2 + pa * Math.PI / 180);             // pa from north (up) through east (left)
    // bounding half-extents of the rotated ellipse, so it sits clear of the corner
    const hx = Math.hypot(a * Math.cos(rot), b * Math.sin(rot)), hy = Math.hypot(a * Math.sin(rot), b * Math.cos(rot));
    const pad = 9 * dpr, cx = pad + hx, cy = H - pad - hy;
    if (2 * hx > W * 0.6 || 2 * hy > H * 0.6) return;            // beam bigger than the view (zoomed far in)
    ctx.save(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.ellipse(cx, cy, a, b, rot, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(255,255,255,.28)"; ctx.fill();
    ctx.lineWidth = 1.4 * dpr; ctx.strokeStyle = "#fff"; ctx.stroke();
    if (this.beamCtl) {   // editable beam: the control sits where the label would be
      this.beamCtl.style.left = `${(cx + hx + 7 * dpr) / dpr}px`; this.beamCtl.style.bottom = `${Math.max(pad, H - cy - 14 * dpr) / dpr}px`;
      ctx.restore(); return;
    }
    const t = `${fmt(bmaj, 3)}″ × ${fmt(bmin, 3)}″, ${fmt(pa, 3)}°`, fs = 12 * dpr;
    ctx.font = `700 ${fs}px ${getComputedStyle(document.body).fontFamily}`;
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    const tx = cx + hx + 7 * dpr, ty = Math.min(cy, H - pad - fs / 2);
    ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.strokeText(t, tx, ty);
    ctx.fillStyle = "#fff"; ctx.fillText(t, tx, ty);
    ctx.restore();
  }
  _toScreen(x, y) {   // data pixel corner -> overlay px
    const [x0, x1, y0, y1] = this.view(), W = this.overlay.width, H = this.overlay.height;
    return [(x - x0) / (x1 - x0 + 1) * W, (y1 + 1 - y) / (y1 - y0 + 1) * H];
  }
  drawOverlay(tmp) {
    const r = this.cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    this.overlay.width = Math.max(1, Math.round(r.width * dpr)); this.overlay.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = this.overlay.getContext("2d");
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    this._drawContour(ctx, dpr);
    this._drawBeam(ctx, dpr);
    const b = tmp || this.box;
    if (!b || !this.arr || (this.crop && !this.canDraw)) return;
    const [ax, ay] = this._toScreen(b.x0, b.y1 + 1), [bx, by] = this._toScreen(b.x1 + 1, b.y0);
    ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = b.color || "#7fd3ff"; ctx.setLineDash(b.color ? [] : [5 * dpr, 3 * dpr]);
    ctx.strokeRect(ax, ay, bx - ax, by - ay);
    if (!b.label) return;
    // label lines under the box (above it if there is no room), with a dark halo to read on any image
    const lines = [].concat(b.label), fs = 11.5 * dpr, lh = fs * 1.3, W = this.overlay.width, H = this.overlay.height;
    ctx.font = `700 ${fs}px ${getComputedStyle(document.body).fontFamily}`;
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.setLineDash([]);
    const below = by + 4 * dpr + lines.length * lh <= H, y0 = below ? by + 4 * dpr : ay - 4 * dpr - lines.length * lh;
    lines.forEach((t, i) => {
      const half = ctx.measureText(t).width / 2 + 3 * dpr, x = Math.min(W - half, Math.max(half, (ax + bx) / 2));
      ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.strokeText(t, x, y0 + i * lh);
      ctx.fillStyle = b.color; ctx.fillText(t, x, y0 + i * lh);
    });
  }
  _square(s, x, y) {   // square with one corner at s, growing towards (x, y), kept inside the view
    const [vx0, vx1, vy0, vy1] = this.view(), dx = x >= s[0] ? 1 : -1, dy = y >= s[1] ? 1 : -1;
    const n = Math.min(Math.max(Math.abs(x - s[0]), Math.abs(y - s[1])) + 1,
      dx > 0 ? vx1 + 1 - s[0] : s[0] - vx0 + 1, dy > 0 ? vy1 + 1 - s[1] : s[1] - vy0 + 1);
    const x0 = dx > 0 ? s[0] : s[0] - n + 1, y0 = dy > 0 ? s[1] : s[1] - n + 1;
    return { x0, x1: x0 + n - 1, y0, y1: y0 + n - 1 };
  }
  _box(s, x, y) {
    return this.square ? this._square(s, x, y)
      : { x0: Math.min(s[0], x), x1: Math.max(s[0], x), y0: Math.min(s[1], y), y1: Math.max(s[1], y) };
  }
  _pix(e) {
    const r = this.canvas.getBoundingClientRect(), [x0, x1, y0, y1] = this.view();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const x = Math.min(x1, Math.max(x0, x0 + Math.floor(fx * (x1 - x0 + 1))));
    const y = Math.min(y1, Math.max(y0, y1 - Math.floor(fy * (y1 - y0 + 1))));
    return [x, y];
  }
  _inBox(x, y) { const b = this.box; return b && x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1; }
  _moved(m, x, y) {   // the box being moved, shifted by the drag and kept inside the view
    const [vx0, vx1, vy0, vy1] = this.view(), b = m.box, w = b.x1 - b.x0, hh = b.y1 - b.y0;
    const x0 = Math.min(vx1 - w, Math.max(vx0, b.x0 + x - m.at[0]));
    const y0 = Math.min(vy1 - hh, Math.max(vy0, b.y0 + y - m.at[1]));
    return { x0, x1: x0 + w, y0, y1: y0 + hh };
  }
  _mouse() {
    let start = null, move = null;
    this.canvas.addEventListener("mousemove", (e) => {
      if (!this.arr) return;
      const [x, y] = this._pix(e), v = this.arr.data[y * this.arr.nx + x];
      const drawable = this.canDraw && this.canDraw();
      this.canvas.style.cursor = move ? "grabbing" : drawable && this._inBox(x, y) ? "grab" : "";
      let off = "";
      if (this.cell) {
        const dx = -(x - this.arr.nx / 2) * this.cell, dy = (y - this.arr.ny / 2) * this.cell;
        off = `  Δα ${dx.toFixed(2)}″ Δδ ${dy.toFixed(2)}″`;
      }
      this.readout.textContent = `x ${x}  y ${y}  ${fmt(v, 4)}${off}`;
      if (move) { const b = this._moved(move, x, y); if (this.onDrag) this.onDrag(b, this); else this.drawOverlay(b); }
      else if (start) {
        const b = this._box(start, x, y);
        if (this.onDrag) this.onDrag(b, this); else this.drawOverlay(b);
      }
    });
    this.canvas.addEventListener("mouseleave", () => { this.readout.textContent = " "; });
    if (!this.onRegion) return;
    this.canvas.addEventListener("mousedown", (e) => {
      if (!this.arr || (this.canDraw ? !this.canDraw() : this.crop)) return;
      const p = this._pix(e);
      if (this.canDraw && this._inBox(...p)) move = { at: p, box: { x0: this.box.x0, x1: this.box.x1, y0: this.box.y0, y1: this.box.y1 } };
      else start = p;
      e.preventDefault();
    });
    window.addEventListener("mouseup", (e) => {
      if (move) { const [x, y] = this._pix(e), b = this._moved(move, x, y); move = null; this.canvas.style.cursor = "grab"; this.onRegion(b, this); return; }
      if (!start) return;
      const [x, y] = this._pix(e), s = start; start = null;
      const b = this._box(s, x, y);
      if (b.x1 - b.x0 < 2 || b.y1 - b.y0 < 2) { if (this.onDrag) this.onDrag(null, this); else this.drawOverlay(); return; }
      this.onRegion(b, this);
    });
  }
}

// ------------------------------------------------------------ line plots --
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : +v.toPrecision(12));
  return out;
}

// band: [xa, xb] shades that x range (edges get class "edge", data-edge 0/1, for dragging);
// the returned svg carries X (data -> viewBox x) and fromX (viewBox x -> data).
// marks: [{x, label?}] dashed vertical lines (e.g. reweighting), drawn once the x range reaches them
function linePlot(series, { xlabel = "", ylabel = "", width = 520, height = 220, logy = false, vline = null, band = null, left = 58, marks = null } = {}) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.setAttribute("class", "plot");
  const m = { l: left, r: 12, t: 10, b: 36 }, W = width - m.l - m.r, H = height - m.t - m.b;
  const xs = series.flatMap((s) => s.x), tr = (v) => logy ? Math.log10(Math.max(v, 1e-30)) : v;
  const ys = series.flatMap((s) => s.y.map(tr)).filter(Number.isFinite);
  if (!xs.length || !ys.length) return svg;
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) { y1 = y0 + Math.abs(y0 || 1) * 0.1; y0 -= Math.abs(y0 || 1) * 0.1; }
  const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
  const X = (v) => m.l + (v - x0) / (x1 - x0) * W, Y = (v) => m.t + H - (v - y0) / (y1 - y0) * H;
  const el = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.append(e); return e; };
  for (const t of niceTicks(y0, y1, 4)) {
    el("line", { x1: m.l, x2: m.l + W, y1: Y(t), y2: Y(t), class: "grid" });
    el("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end" }).textContent = tick(logy ? Math.pow(10, t) : t);
  }
  for (const t of niceTicks(x0, x1, 6)) {
    el("line", { x1: X(t), x2: X(t), y1: m.t + H, y2: m.t + H + 4, class: "axis" });
    el("text", { x: X(t), y: m.t + H + 16, "text-anchor": "middle" }).textContent = tick(t);
  }
  el("rect", { x: m.l, y: m.t, width: W, height: H, class: "frame" });   // full frame
  if (y0 < 0 && y1 > 0 && !logy) el("line", { x1: m.l, x2: m.l + W, y1: Y(0), y2: Y(0), class: "axis" });
  el("text", { x: m.l + W / 2, y: height - 4, "text-anchor": "middle" }).textContent = xlabel;
  const yl = el("text", { x: 12, y: m.t + H / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + H / 2})` });
  yl.textContent = ylabel;
  svg.X = X; svg.fromX = (px) => x0 + (px - m.l) / W * (x1 - x0);
  for (const mk of marks || []) {
    if (mk.x < Math.min(x0, x1) || mk.x > Math.max(x0, x1)) continue;
    const mx = X(mk.x);
    el("line", { x1: mx, x2: mx, y1: m.t, y2: m.t + H, class: "mark" });
    if (mk.label)   // reads upwards beside the line, from the top of the frame
      el("text", { x: mx - 4, y: m.t + 5, "text-anchor": "end", class: "mark-label", transform: `rotate(-90 ${mx - 4} ${m.t + 5})` }).textContent = mk.label;
  }
  if (band) {
    const [a, b] = band.map(X);
    el("rect", { x: Math.min(a, b), y: m.t, width: Math.abs(b - a), height: H, class: "band" });
    for (const cls of ["edge", "edge-hit"])   // the hit line is a wide transparent strip to grab
      band.forEach((v, i) => el("line", { x1: X(v), x2: X(v), y1: m.t, y2: m.t + H, class: cls, "data-edge": i }));
  }
  if (vline !== null && vline >= Math.min(x0, x1) && vline <= Math.max(x0, x1))
    el("line", { x1: X(vline), x2: X(vline), y1: m.t, y2: m.t + H, class: "vline" });
  for (const s of series) {
    const pts = s.x.map((x, i) => [x, tr(s.y[i])]).filter((p) => Number.isFinite(p[1]));
    if (!pts.length) continue;
    el("path", {
      d: pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(""),
      fill: "none", stroke: s.color, "stroke-width": s.width || 1.6, "stroke-dasharray": s.dash || "",
      "stroke-linejoin": "round",
    });
  }
  return svg;
}
// uv coverage: sets = [{u, v (metres), ghz, color}]; each baseline is drawn with its conjugate
// (-u, -v), in kλ at that set's frequency. Square, symmetric axes; colours are CSS values (theme vars).
function uvPlot(sets, size = 400) {
  const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`); svg.setAttribute("class", "plot uv");
  const m = { l: 50, r: 12, t: 10, b: 38 }, S = Math.min(size - m.l - m.r, size - m.t - m.b);
  const el = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.append(e); return e; };
  const kl = sets.map((s) => ({ ...s, k: s.ghz / 299.792458 }));   // metres -> kλ
  let R = 0;
  for (const s of kl) for (let i = 0; i < s.u.length; i++) R = Math.max(R, Math.abs(s.u[i] * s.k), Math.abs(s.v[i] * s.k));
  R = R * 1.05 || 1;
  const X = (u) => m.l + (u + R) / (2 * R) * S, Y = (v) => m.t + S - (v + R) / (2 * R) * S;
  for (const t of niceTicks(-R, R, 4)) {
    el("line", { x1: X(t), x2: X(t), y1: m.t, y2: m.t + S, class: "grid" });
    el("line", { x1: m.l, x2: m.l + S, y1: Y(t), y2: Y(t), class: "grid" });
    el("text", { x: X(t), y: m.t + S + 16, "text-anchor": "middle" }).textContent = tick(t);
    el("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end" }).textContent = tick(t);
  }
  el("rect", { x: m.l, y: m.t, width: S, height: S, class: "axis", fill: "none" });
  el("text", { x: m.l + S / 2, y: size - 4, "text-anchor": "middle" }).textContent = "u (kλ)";
  el("text", { x: 12, y: m.t + S / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + S / 2})` }).textContent = "v (kλ)";
  for (const s of kl) {   // one path of zero-length round-capped segments per set: cheap to draw, styled by CSS
    let d = "";
    for (let i = 0; i < s.u.length; i++) {
      const u = s.u[i] * s.k, v = s.v[i] * s.k;
      d += `M${X(u).toFixed(1)} ${Y(v).toFixed(1)}h0M${X(-u).toFixed(1)} ${Y(-v).toFixed(1)}h0`;
    }
    el("path", { d, class: "uvpts" }).style.stroke = s.color;
  }
  return svg;
}
// Iterations where a run reweights, in the numbering of its convergence plots, from its settings and
// the solvers' own rules: the torch solvers (gpu_pd, gpu_fista) at burn-in + 1, then every
// reweight_every; numpy primal-dual at burn-in, then every reweight_every; numpy FISTA reweights on
// every iteration from burn-in on, so that is one "starts" mark.
function reweightMarks(p) {
  if (!p || !p.reweight) return [];
  const n = +p.n_iter || 5000, start = +(p.burn_in_iters || Math.floor(n / 3)), every = Math.max(1, +(p.reweight_every || 1000));
  if (p.backend === "fista") return start <= n ? [{ x: start, label: "Reweighting starts" }] : [];
  const out = [];
  for (let k = start + (String(p.backend).startsWith("gpu_") ? 1 : 0); k <= n && out.length < 50; k += every) out.push({ x: k, label: `Reweighting #${out.length + 1}` });
  return out;
}
const unlabelled = (marks) => marks.map((mk) => ({ x: mk.x }));
const markLegend = () => h("div", { class: "legend mark-legend" }, h("span", {}, h("i", { class: "dash" }), "Re-weighting"));
const legend = (series) => h("div", { class: "legend" }, series.map((s) => h("span", {}, h("i", { style: `background:${s.color}` }), s.name)));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ------------------------------------------------------------------ state --
const S = {
  info: null, datasets: [], dsId: null, tab: "overview", resId: null,
  disp: { cmap: "inferno", stretch: "linear", clip: 100 },
  timers: new Set(),
};
function clearTimers() { for (const t of S.timers) clearInterval(t); S.timers.clear(); }
function every(ms, fn) { const t = setInterval(fn, ms); S.timers.add(t); return t; }

// -------------------------------------------------------------- top bar --
async function loadInfo() {
  S.info = await api("/api/info", { timeout: 15000 });
  const r = S.info.repo, el = $("#repo");
  el.innerHTML = "";
  if (r.git) {
    el.append(h("span", {}, `${r.branch} @ ${r.commit}`), h("span", {}, r.date));
    if (r.conflicts && r.conflicts.length) el.append(h("span", { class: "err", title: r.conflicts.join("\n") }, `${r.conflicts.length} unresolved conflict(s)`));
    else if (r.modified) el.append(h("span", { class: "warn" }, `${r.modified} modified`));
    if (r.behind) el.append(h("span", { class: "warn", title: "as of your last git fetch" }, `${r.behind} behind upstream`));
  } else el.append(h("span", {}, "not a git checkout"));
  $("#side-foot").innerHTML = "";
  $("#side-foot").append(
    h("div", {}, `host ${S.info.host}`), h("div", {}, `workdir ${S.info.workdir}`),
    h("div", {}, `CASA ${S.info.casa ? "available" : "not found"} · torch ${S.info.torch ? "yes" : "no"}`),
    ...(S.devices ? [h("div", {}, `devices: ${S.devices.devices.map((d) => d.id).join(", ")}`)] : []));
}

let polling = false;
async function pollJobs() {
  if (polling) return;
  polling = true;
  try {
    const jobs = await api("/api/jobs", { timeout: 10000 }), running = jobs.filter((j) => j.status === "running" || j.status === "queued"), el = $("#jobs-indicator");
    el.hidden = !running.length; el.innerHTML = "";
    if (running.length) {
      const j = running[0];
      el.append(h("span", { class: "dot" }), h("span", {}, running.length > 1 ? `${jobs.filter((j) => j.status === "running").length} running, ${jobs.filter((j) => j.status === "queued").length} queued` : `${j.label}: ${j.fraction != null ? Math.round(j.fraction * 100) + "%" : "…"}`));
      el.style.cursor = j.ds_id ? "pointer" : "";
      el.onclick = () => j.ds_id && (j.kind === "deconvolve" ? selectDataset(j.ds_id, "deconvolve", j.res_id) : selectDataset(j.ds_id, "overview"));
    }
  } catch { /* server restarting */ } finally { polling = false; }
}

// -------------------------------------------------------------- sidebar --
async function loadDatasets() {
  S.datasets = await api("/api/datasets");
  const ul = $("#ds-list");
  ul.innerHTML = "";
  if (!S.datasets.length) ul.append(h("li", { class: "note", style: "cursor:default" }, "No datasets yet."));
  for (const d of S.datasets) {
    const st = d.status, nres = (d.results || []).length;
    const chip = st === "ready" ? null : h("span", { class: "chip " + (st === "imaging" ? "run" : "err") }, st);
    const running = (d.results || []).some((r) => r.status === "running" || r.status === "queued");
    ul.append(h("li", { class: d.id === S.dsId ? "active" : "", onclick: () => selectDataset(d.id) },
      h("div", { class: "name" }, d.name),
      h("div", { class: "sub" },
        d.example ? h("span", { class: "chip ex" }, "example") : null,
        h("span", {}, d.source === "ms" ? "from MS" : d.source === "toy" ? "synthetic" : "imported"),
        h("span", {}, `${d.nchan ?? "?"}×${d.imsize ?? "?"}²`),
        nres ? h("span", {}, `${nres} result${nres > 1 ? "s" : ""}`) : null,
        running ? h("span", { class: "chip run" }, "running") : null, chip)));
  }
}

function setMain(...nodes) {
  clearTimers(); const m = $("#main"); m.innerHTML = "";
  if (S.linkTo && S.dsId === null)   // Load MS / Import, opened to add a high-resolution reference
    m.append(h("div", { class: "card link-banner" }, h("div", { class: "row" },
      h("b", {}, `Adding the high-resolution reference for ${S.linkTo.name}`),
      h("span", { class: "note" }, S.linkTo.grid ? "imaging is pre-filled with its grid, so pixels and channels line up" : ""),
      h("div", { style: "flex:1" }), h("button", { onclick: () => { S.linkTo = null; m.firstChild.remove(); } }, "Cancel"))));
  m.append(...nodes); m.scrollTop = 0;
}
// The dataset just made (or loaded, or imported) becomes the high-resolution reference of S.linkTo.ds.
async function linkReference(newId) {
  if (!S.linkTo || !newId) return;
  const to = S.linkTo; S.linkTo = null;
  try { await api(`/api/datasets/${to.ds}/meta`, { method: "POST", body: { hires: "yes", hires_ref: newId } }); toast(`Linked as the high-resolution reference of ${to.name}`); }
  catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------ browser --
function fileBrowser({ start, onSelect, filter = () => true, badge = () => null, pickDirs = true }) {
  const root = $("#tpl-browser").content.firstElementChild.cloneNode(true);
  const input = $(".path", root), list = $(".entries", root);
  let cur = null, selected = null;
  async function open(path) {
    try {
      const r = await api(`/api/browse?path=${encodeURIComponent(path || "")}`);
      cur = r; input.value = r.path; list.innerHTML = "";
      $(".up", root).disabled = !r.parent;
      for (const e of r.entries.filter(filter)) {
        const p = r.path.replace(/\/$/, "") + "/" + e.name;
        const li = h("li", {}, h("span", { class: "ic" }, e.is_dir ? "▸" : "·"), h("span", {}, e.name), badge(e));
        li.addEventListener("click", () => {
          $$(".sel", list).forEach((x) => x.classList.remove("sel")); li.classList.add("sel");
          selected = { ...e, path: p };
          onSelect && onSelect(selected);
        });
        li.addEventListener("dblclick", () => { if (e.is_dir) open(p); });
        list.append(li);
      }
      if (!list.children.length) list.append(h("li", { class: "note" }, "(empty)"));
    } catch (err) { toast(err.message, true); }
  }
  $(".up", root).onclick = () => cur && cur.parent && open(cur.parent);
  $(".go", root).onclick = () => open(input.value);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") open(input.value); });
  open(start);
  return root;
}

// A small window over the page; returns close(). Escape or a click outside closes it.
function modal(title, ...content) {
  const box = h("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": title }, h("h2", {}, title), ...content);
  const bg = h("div", { class: "modal-bg" }, box);
  const esc = (e) => { if (e.key === "Escape") close(); };
  const close = () => { bg.remove(); document.removeEventListener("keydown", esc); };
  bg.addEventListener("mousedown", (e) => { if (e.target === bg) close(); });
  document.addEventListener("keydown", esc);
  document.body.append(bg);
  return close;
}

// Save on disk: pick a folder (on the server) and a file name, then write the pickle.
function saveOnDiskDialog(defaultName, body) {
  const name = h("input", { value: defaultName, spellcheck: "false", class: "mono" });
  const go = h("button", { class: "primary" }, "Save"), status = h("div", { class: "note" });
  let dir = S.lastDir || "";
  const br = fileBrowser({ start: dir, filter: (e) => e.is_dir || e.ext === ".pkl",
    onSelect: (e) => { if (e.is_dir) dir = e.path; else { name.value = e.name; dir = e.path.replace(/\/[^/]+$/, ""); } } });
  const close = modal("Save on disk", h("div", { class: "note" }, "Folder on the server (double-click to open), then a file name."), br,
    h("label", { class: "field" }, "file name", name), status, h("div", { class: "row end" }, h("button", { onclick: () => close() }, "Cancel"), go));
  go.onclick = async () => {
    const folder = $(".path", br).value.replace(/\/$/, "") || dir, file = name.value.trim();
    if (!file.endsWith(".pkl")) return toast("the file name must end in .pkl", true);
    const send = (overwrite) => api(`/api/datasets/${body.id}/export`, { method: "POST", timeout: 600000, body: { ...body.req, path: `${folder}/${file}`, overwrite } });
    go.disabled = true; status.textContent = "writing…";
    try {
      let r;
      try { r = await send(false); }
      catch (e) { if (!/already exists/.test(e.message) || !confirm(`${folder}/${file} exists. Overwrite it?`)) throw e; r = await send(true); }
      S.lastDir = folder; close(); toast(`Saved ${r.path} (${(r.size / 1e6).toFixed(1)} MB)`);
    } catch (e) { status.textContent = ""; go.disabled = false; toast(e.message, true); }
  };
}

// Load saved data: pick a .pkl written by Save on disk; it becomes a new dataset.
function loadSavedDialog() {
  let file = null;
  const go = h("button", { class: "primary", disabled: true }, "Load"), status = h("div", { class: "note" });
  const br = fileBrowser({ start: S.lastDir || "", filter: (e) => e.is_dir || e.ext === ".pkl",
    badge: (e) => e.ext === ".pkl" && e.size ? h("span", { class: "chip badge" }, `${(e.size / 1e6).toFixed(1)} MB`) : null,
    onSelect: (e) => { file = e.is_dir ? null : e.path; go.disabled = !file; } });
  const close = modal("Load saved data", h("div", { class: "note" }, "A .pkl written by Save on disk (double-click folders to open)."), br, status,
    h("div", { class: "row end" }, h("button", { onclick: () => close() }, "Cancel"), go));
  go.onclick = async () => {
    go.disabled = true; status.textContent = "loading…";
    try {
      const m = await api("/api/datasets/from_pkl", { method: "POST", timeout: 600000, body: { path: file } });
      await linkReference(m.id);
      S.lastDir = file.replace(/\/[^/]+$/, ""); close(); toast(`Loaded "${m.name}"`); await loadDatasets(); selectDataset(m.id);
    } catch (e) { status.textContent = ""; go.disabled = false; toast(e.message, true); }
  };
}

// ------------------------------------------------------- job monitor --
function jobMonitor(job, { onDone, live = false, cell = null, dv = 1, marks = [] } = {}) {
  const bar = h("div", { class: "bar" }, h("span"));
  const msg = h("div", { class: "progress-msg" }, job.message || "");
  const log = h("pre", { class: "log" });
  const cancel = h("button", { class: "danger", onclick: async () => { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); cancel.disabled = true; } }, "Cancel");
  const err = h("div", { class: "err-box" });
  const devEl = h("span", { class: "chip run" }, job.device ? `on ${job.device}` : job.status === "queued" ? "queued" : "");
  const card = h("div", { class: "card" }, h("div", { class: "row" }, h("h2", {}, job.label), devEl, h("div", { class: "spacer", style: "flex:1" }), cancel), bar, msg);
  let panel = null, plotBox = null, stats = [], seq = -1, busy = false;
  let liveStop = false, liveProgress = false;   // liveProgress: the frames carry the bar and message
  if (live) {
    const sbUnit = cell ? `Jy ${dv !== 1 ? "km/s" : "ch"} arcsec\u207b\u00b2` : "Jy/pixel km/s";
    panel = new ImagePanel({ title: "current iterate, moment 0", unit: sbUnit, cell, cbar: "bottom" });
    plotBox = h("div", { class: "live-plots" });
    card.append(h("div", { class: "live-grid" }, panel.el, plotBox));
    // newest iterate, fetched as fast as the connection allows (the worker sends one every iteration;
    // each fetch returns the latest, so every redraw shows the current iteration)
    let lastIt = -1, shown = false;
    const frameLoop = async () => {
      // the card is put on the page by the caller after this starts: wait for it, stop once it is gone
      if (liveStop) return;
      if (!document.body.contains(card)) { if (!shown) setTimeout(frameLoop, 100); return; }
      shown = true;
      try {
        const a = await getArray(`/api/jobs/${job.id}/frame`);
        if (a.iter !== lastIt) {
          lastIt = a.iter;
          panel.set(a);
          panel.setTitle(`iteration ${a.iter}`);
          // the bar and message of this very iteration, so they never lag the image
          if (a.message !== null) {
            liveProgress = true; msg.textContent = a.message; bar.classList.remove("indet");
            if (a.fraction !== null) $("span", bar).style.width = `${Math.round(a.fraction * 100)}%`;
          }
          const [, hi] = panel.autoLimits(100);                 // clip 100%: 0 .. the maximum shown
          panel.render(await lut(S.disp.cmap), S.disp.stretch, 0, hi || 1);
        }
      } catch { /* no frame yet, or a dropped request: try again */ }
      setTimeout(frameLoop, 40);
    };
    frameLoop();
  }
  card.append(h("details", { open: !live }, h("summary", { class: "note" }, "log"), log), err);
  const tick = async () => {
    if (busy) return; busy = true;
    try {
      const j = await api(`/api/jobs/${job.id}?since=${stats.length}`, { timeout: 15000 });
      card.classList.remove("offline");
      stats.push(...j.stats);
      bar.classList.toggle("indet", j.fraction === null && j.status === "running");
      if (j.device) devEl.textContent = `on ${j.device}`;
      else if (j.status === "queued") devEl.textContent = `queued for ${(j.devices || []).join(" / ")}`;
      if (!(liveProgress && j.status === "running")) {   // while frames arrive, they set these (fresher)
        $("span", bar).style.width = j.fraction === null ? "" : `${Math.round(j.fraction * 100)}%`;
        msg.textContent = j.message || "";
      }
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
      log.textContent = j.log.join("\n");
      if (atBottom) log.scrollTop = log.scrollHeight;
      if (live) drawPlots();
      if (!["running", "queued"].includes(j.status)) {
        clearInterval(t); S.timers.delete(t); liveStop = true;
        cancel.disabled = true; bar.classList.remove("indet");
        if (j.status === "error") { err.textContent = j.error || "failed"; msg.textContent = "failed"; }
        if (j.status === "cancelled") msg.textContent = "cancelled";
        onDone && onDone(j);
      }
    } catch (e) {
      // No answer at all (port forward stopped, network down, timeout): the job keeps running
      // on the server, so keep trying and pick up where we left off. An answer that is an
      // error (e.g. unknown job after a server restart) is final.
      if (e instanceof TypeError || e.name === "TimeoutError") {
        card.classList.add("offline");
        msg.textContent = "no connection to the server, retrying… (the job keeps running there)";
      } else {
        msg.textContent = `lost contact with the job (${e.message})`;
        clearInterval(t); S.timers.delete(t);
      }
    } finally { busy = false; }
  };
  // live: the two convergence plots, sized in pixels so that together they are as tall as the image
  // (with its colour bar) beside them; redrawn with each status update and when the layout changes
  let plotSize = "";
  function drawPlots(force = true) {
    if (!live || !stats.length) return;
    const leg = marks.length ? 30 : 0;   // the legend row above the plots
    const w = Math.round(plotBox.clientWidth), hh = Math.max(140, Math.floor((panel.el.getBoundingClientRect().height - 12 - leg) / 2));
    if (!w || (!force && `${w}x${hh}` === plotSize)) return;
    plotSize = `${w}x${hh}`;
    const x = stats.map((s) => s.iter);
    const a = [{ name: "residual rms", color: cssVar("--s-dirty"), x, y: stats.map((s) => s.residual_rms) }];
    const b = [{ name: "model flux (Jy)", color: cssVar("--s-model"), x, y: stats.map((s) => s.flux) }];
    plotBox.replaceChildren(marks.length ? markLegend() : "",
      linePlot(a, { xlabel: "iteration", ylabel: "residual rms", width: w, height: hh, logy: true, marks: unlabelled(marks) }),
      linePlot(b, { xlabel: "iteration", ylabel: "model flux (Jy)", width: w, height: hh, marks: unlabelled(marks) }));
  }
  if (live) new ResizeObserver(() => requestAnimationFrame(() => drawPlots(false))).observe(panel.el);
  const t = every(800, tick);
  tick();
  return card;
}

// ============================================================ Load MS view
const LINES = [
  ["CO(1-0)", 115.271202], ["CO(2-1)", 230.538], ["CO(3-2)", 345.79599], ["13CO(2-1)", 220.398684],
  ["C18O(2-1)", 219.560358], ["HCN(1-0)", 88.631602], ["HCO+(1-0)", 89.188525], ["[CI](1-0)", 492.160651], ["[CII]", 1900.5369],
];
const C_KMS = 299792.458;

function viewMS() {
  S.dsId = null; loadDatasets(); saveView({ view: "ms" });
  const noCasa = !S.info.casa ? h("div", { class: "card", style: "border-color:var(--warn)" },
    h("b", {}, "CASA is not available in this Python environment."),
    h("div", { class: "note" }, "You can browse, but inspecting an MS and dirty imaging need casatools + casatasks (pip install casatools casatasks). Start the app from your CASA environment, or use Import cubes.")) : null;
  const sel = h("div", { class: "note" }, "Select a measurement set (marked MS). Double-click a folder to open it. To image several MSs together (e.g. the execution blocks of one configuration), select one, click Add, then select the next.");
  const inspectBtn = h("button", { class: "primary", disabled: true }, "Inspect");
  const addBtn = h("button", { disabled: true, title: "Keep this MS and select another one to image with it" }, "+ Add another MS");
  const picked = h("div", { class: "row", style: "flex-wrap:wrap;gap:6px" });
  const out = h("div", { style: "display:grid;gap:14px" });
  let ms = null;
  const kept = [];
  const targets = () => (ms && !kept.includes(ms) ? [...kept, ms] : [...kept]);
  function refresh() {
    picked.innerHTML = "";
    if (kept.length) picked.append(h("span", { class: "note" }, "Imaged together:"),
      ...targets().map((p) => h("span", { class: "chip ok", title: p }, p.split("/").filter(Boolean).pop(), " ",
        h("a", { href: "#", title: "remove", onclick: (e) => { e.preventDefault(); if (kept.includes(p)) kept.splice(kept.indexOf(p), 1); else ms = null; refresh(); } }, "×"))));
    const n = targets().length;
    addBtn.disabled = !ms || kept.includes(ms);
    inspectBtn.disabled = !n || !S.info.casa;
    inspectBtn.textContent = n > 1 ? `Inspect ${n} MSs` : "Inspect";
  }
  const br = fileBrowser({
    start: S.lastMSDir || "",
    badge: (e) => e.is_ms ? h("span", { class: "chip ok badge" }, "MS") : null,
    onSelect: (e) => {
      ms = e.is_ms ? e.path : null;
      sel.textContent = ms ? ms : "Not a measurement set. Double-click to open the folder.";
      refresh();
    },
  });
  addBtn.onclick = () => { if (ms && !kept.includes(ms)) kept.push(ms); refresh(); };
  inspectBtn.onclick = async () => {
    const paths = targets();
    S.lastMSDir = paths[paths.length - 1].replace(/\/[^/]+\/?$/, "");
    out.innerHTML = "";
    try {
      const job = await api("/api/ms/inspect", { method: "POST", body: { paths } });
      out.append(jobMonitor(job, { onDone: (j) => { if (j.status === "done") { out.innerHTML = ""; out.append(imagingForm(paths, j.result)); } } }));
    } catch (e) { toast(e.message, true); }
  };
  setMain(h("div", { class: "view-head" }, h("h1", {}, "Load a measurement set")), noCasa,
    h("div", { class: "card" }, br, h("div", { class: "row" }, sel, h("div", { style: "flex:1" }), addBtn, inspectBtn), picked), out);
}

function imagingForm(msList, info) {
  const spws = info.spws || [], fields = info.target_fields || [];
  const perMS = info.per_ms || [info];
  // The same spw can carry a different id, and is Doppler-shifted, in each
  // execution block: pick the one overlapping the chosen spw most in frequency.
  const matchSpw = (s, m) => {
    let best = null, bestOv = 0;
    for (const t of m.spws || []) {
      const ov = Math.min(s.f_hi, t.f_hi) - Math.max(s.f_lo, t.f_lo);
      if (ov > bestOv) { best = t; bestOv = ov; }
    }
    return best && bestOv > 0.5 * (s.f_hi - s.f_lo) ? best.spw : null;
  };
  const f0 = spws.length ? (spws[0].f_lo + spws[0].f_hi) / 2 : 230;
  const guessLine = LINES.reduce((best, l) => Math.abs(l[1] / f0 - 1) < Math.abs(best[1] / f0 - 1) ? l : best, LINES[1]);
  const inp = (name, val, attrs = {}) => h("input", { name, value: val ?? "", ...attrs });
  const form = h("form", { class: "form", onsubmit: (e) => e.preventDefault() });
  const lineSel = h("select", { name: "line" }, LINES.map(([n, f]) => h("option", { value: f, selected: n === guessLine[0] }, `${n}  ${f} GHz`)), h("option", { value: "" }, "custom"));
  const fieldSel = h("select", { name: "field" }, fields.map((f) => h("option", { value: f.name }, `${f.name}`)));
  const spwSel = h("select", { name: "spw" }, spws.map((s) => h("option", { value: s.spw }, `spw ${s.spw}: ${s.f_lo.toFixed(3)}–${s.f_hi.toFixed(3)} GHz, ${s.nchan} ch`)));
  form.append(
    h("label", {}, "field", fieldSel),
    h("label", { class: "wide" }, "phase centre", inp("phasecenter", fields[0] && fields[0].phasecenter, { class: "mono" })),
    h("label", {}, "spectral window", spwSel),
    h("label", {}, "line", lineSel),
    h("label", {}, "rest frequency (GHz)", inp("restfreq_ghz", guessLine[1], { type: "number", step: "any" })),
    h("label", {}, "start velocity (km/s, radio LSRK)", inp("start_kms", "", { type: "number", step: "any" })),
    h("label", {}, "channel width (km/s)", inp("width_kms", "", { type: "number", step: "any" })),
    h("label", {}, "number of channels", inp("nchan", "", { type: "number", step: 1 })),
    h("label", {}, "image size (px)", inp("imsize", 256, { type: "number", step: 2 })),
    h("label", {}, "cell (arcsec)", inp("cell_arcsec", "", { type: "number", step: "any" })),
    h("label", {}, "weighting", h("select", { name: "weighting" }, ["natural", "briggs", "uniform"].map((w) => h("option", {}, w)))),
    h("label", {}, "dataset name", inp("name", `${fields[0] ? fields[0].name : "dataset"}`)),
  );
  const F = (n) => form.elements[n];
  function suggest() {
    const s = spws.find((x) => String(x.spw) === F("spw").value) || spws[0];
    const rest = +F("restfreq_ghz").value;
    if (!s || !rest) return;
    const fc = (s.f_lo + s.f_hi) / 2, vc = C_KMS * (1 - fc / rest);
    const dvNative = C_KMS * (s.chanwidth_khz * 1e-6) / rest;
    const width = Math.max(5, Math.ceil(dvNative / 5) * 5);
    const bw = C_KMS * (s.f_hi - s.f_lo) / rest;
    const nchan = Math.min(Math.floor(bw / width) - 2, 120);
    F("width_kms").value = width;
    F("nchan").value = nchan;
    F("start_kms").value = Math.round(vc - nchan / 2 * width);
    if (info.bl_max) {
      const res = (C_KMS * 1e3 / (fc * 1e9)) / info.bl_max * 206265;     // lambda / Bmax, arcsec
      const cell = +(res / 5).toPrecision(2);
      const pb = 1.13 * (C_KMS * 1e3 / (fc * 1e9)) / 12 * 206265;         // 12m primary beam FWHM
      let n = Math.ceil(1.2 * pb / cell / 32) * 32;
      F("cell_arcsec").value = cell;
      F("imsize").value = Math.min(Math.max(n, 64), 512);
    }
  }
  lineSel.onchange = () => { if (lineSel.value) F("restfreq_ghz").value = lineSel.value; suggest(); };
  const spwMap = h("td", { class: "mono" });
  function spwList() {
    const s = spws.find((x) => String(x.spw) === F("spw").value) || spws[0];
    return perMS.map((m, i) => (i === 0 ? (s ? s.spw : null) : s ? matchSpw(s, m) : null));
  }
  function showSpwMap() {
    spwMap.textContent = spwList().map((x, i) => `${msList[i].split("/").filter(Boolean).pop()}: ${x ?? "no matching spw"}`).join("\n");
    spwMap.style.whiteSpace = "pre";
  }
  // ---- uv coverage of the inspected MSs, at the chosen spw's frequency in each
  const uvBox = h("div", { class: "uv-box" }), uvLegend = h("div", { class: "legend" });
  const uvColors = ["var(--accent)", "color-mix(in srgb, var(--accent) 45%, var(--fg))", "color-mix(in srgb, var(--accent) 55%, var(--bg))", "var(--muted)"];
  function drawUV() {
    const ids = spwList(), sets = [];
    perMS.forEach((pm, i) => {
      if (!pm.uv || !pm.uv.u.length) return;
      const sp = (pm.spws || []).find((x) => x.spw === ids[i]) || (pm.spws || [])[0];
      sets.push({ u: pm.uv.u, v: pm.uv.v, ghz: sp ? (sp.f_lo + sp.f_hi) / 2 : 230, color: uvColors[i % uvColors.length],
        name: msList[i].split("/").filter(Boolean).pop().replace(/\.ms.*$/, ""), n: pm.uv.u.length, rows: pm.uv.rows });
    });
    if (!sets.length) { uvBox.replaceChildren(h("div", { class: "note" }, perMS.some((pm) => pm.uv_error) ? `no uv data: ${perMS.find((pm) => pm.uv_error).uv_error}` : "Inspect the MS again to see its uv coverage.")); uvLegend.replaceChildren(); return; }
    uvBox.replaceChildren(uvPlot(sets));
    uvLegend.replaceChildren(...sets.map((s) => h("span", { title: `${s.n} of ${s.rows} baselines (target, one spw), and their conjugates` },
      h("i", { style: `background:${s.color}` }), s.name)));
  }
  spwSel.onchange = () => { suggest(); showSpwMap(); drawUV(); };
  showSpwMap(); drawUV();
  fieldSel.onchange = () => { const f = fields.find((x) => x.name === fieldSel.value); if (f) F("phasecenter").value = f.phasecenter; F("name").value = fieldSel.value; };
  suggest();
  if (S.linkTo && S.linkTo.grid) {   // a high-resolution reference: on the grid of the dataset it is compared with
    const g = S.linkTo.grid;
    if (fields.some((f) => f.name === g.field)) fieldSel.value = g.field;
    for (const k of ["phasecenter", "restfreq_ghz", "start_kms", "width_kms", "nchan", "imsize", "cell_arcsec"]) if (g[k] != null && F(k)) F(k).value = g[k];
    F("name").value = `${S.linkTo.name} (high-res reference)`;
  }

  const summary = h("table", { class: "kv" },
    h("tr", {}, h("td", {}, msList.length > 1 ? `${msList.length} MSs` : "MS"), h("td", { class: "mono" }, msList.map((m) => h("div", {}, m)))),
    msList.length > 1 ? h("tr", {}, h("td", {}, "spw per MS"), spwMap) : null,
    h("tr", {}, h("td", {}, "array"), h("td", {}, `${info.array ?? "?"}, ${info.nant} antennas`)),
    info.bl_max ? h("tr", {}, h("td", {}, "baselines"), h("td", {}, `${info.bl_min.toFixed(1)} – ${info.bl_max.toFixed(1)} m`)) : null,
    h("tr", {}, h("td", {}, "targets"), h("td", {}, fields.map((f) => f.name).join(", ") || "–")),
    info.mosaic ? h("tr", {}, h("td", {}, "note"), h("td", { style: "color:var(--warn)" }, "Several pointings of one target (mosaic): tclean_dirty_cube uses the standard gridder and the fast operator assumes one pointing.")) : null);
  const spwTable = h("table", { class: "list" }, h("tr", {}, ["spw", "channels", "GHz", "kHz / chan"].map((t) => h("th", {}, t))),
    spws.map((s) => h("tr", {}, h("td", {}, s.spw), h("td", {}, s.nchan), h("td", {}, `${s.f_lo.toFixed(4)} – ${s.f_hi.toFixed(4)}`), h("td", {}, s.chanwidth_khz.toFixed(1)))));
  const run = h("button", { class: "primary" }, "Make dirty cube + PSF");
  const mon = h("div");
  run.onclick = async () => {
    const p = Object.fromEntries(new FormData(form).entries());
    for (const k of ["imsize", "nchan", "cell_arcsec", "start_kms", "width_kms", "restfreq_ghz"])
      if (p[k] === "" || Number.isNaN(+p[k])) return toast(`set ${k}`, true);
    const name = p.name; delete p.name; delete p.line;
    p.ms = msList;
    p.spw_list = spwList();
    if (p.spw_list.some((x) => x === null)) return toast("no matching spectral window in every MS", true);
    run.disabled = true;
    try {
      const r = await api("/api/datasets/from_ms", { method: "POST", body: { name, params: p } });
      await linkReference(r.dataset);
      loadDatasets();
      mon.append(jobMonitor(r.job, { onDone: (j) => { loadDatasets(); if (j.status === "done") selectDataset(r.dataset); else run.disabled = false; } }));
    } catch (e) { toast(e.message, true); run.disabled = false; }
  };
  return h("div", { style: "display:grid;gap:14px" },
    h("div", { class: "grid2" }, h("div", { class: "card" }, h("h3", {}, "Measurement set"), summary), h("div", { class: "card scroll-x" }, h("h3", {}, "Science spectral windows"), spwTable)),
    h("div", { class: "imaging-row" },
      h("div", { class: "card" }, h("h3", {}, "Dirty imaging (tclean, niter = 0)"),
        h("div", { class: "note" }, "Suggested values come from the spectral setup and the longest baseline: cell ≈ λ/Bmax / 5, image ≈ 1.2 × the primary beam. The PSF is made at twice the image size."),
        form, h("div", { class: "row" }, run)),
      h("div", { class: "card" }, h("h3", {}, "uv coverage"), uvBox, uvLegend)), mon);
}

// ============================================================ Import view
function viewImport() {
  S.dsId = null; loadDatasets(); saveView({ view: "import" });
  const form = h("form", { class: "form", onsubmit: (e) => e.preventDefault() },
    h("label", { class: "wide" }, "folder (dirty.npy + psf.npy, or dirty_cube.fits + dirty_beam_cube.fits)", h("input", { name: "path", class: "mono" })),
    h("label", { class: "wide" }, "…or explicit dirty cube file", h("input", { name: "dirty", class: "mono", placeholder: "optional" })),
    h("label", { class: "wide" }, "…and PSF file (ideally 2× the image size)", h("input", { name: "psf", class: "mono", placeholder: "optional" })),
    h("label", {}, "name", h("input", { name: "name" })),
    h("label", {}, "cell (arcsec)", h("input", { name: "cell_arcsec", type: "number", step: "any" })),
    h("label", {}, "beam major (″)", h("input", { name: "bmaj", type: "number", step: "any" })),
    h("label", {}, "beam minor (″)", h("input", { name: "bmin", type: "number", step: "any" })),
    h("label", {}, "beam PA (deg)", h("input", { name: "bpa", type: "number", step: "any" })),
    h("label", {}, "start velocity (km/s)", h("input", { name: "start_kms", type: "number", step: "any" })),
    h("label", {}, "channel width (km/s)", h("input", { name: "width_kms", type: "number", step: "any" })));
  const F = (n) => form.elements[n];
  const br = fileBrowser({
    start: "",
    badge: (e) => e.is_cubes ? h("span", { class: "chip ok badge" }, "cubes") : e.is_ms ? h("span", { class: "chip badge" }, "MS") : null,
    onSelect: (e) => {
      if (e.is_dir) { F("path").value = e.path; if (!F("name").value) F("name").value = e.name; }
      else if (/\.(npy|fits?)$/i.test(e.name)) {
        const target = /psf|beam/i.test(e.name) ? "psf" : "dirty";
        F(target).value = e.path; F("path").value = e.path.replace(/\/[^/]+$/, "");
      }
    },
  });
  const btn = h("button", { class: "primary" }, "Import");
  btn.onclick = async () => {
    const p = Object.fromEntries(new FormData(form).entries());
    const body = { path: p.path || (p.dirty || "").replace(/\/[^/]+$/, ""), dirty: p.dirty || null, psf: p.psf || null, name: p.name,
      cell_arcsec: p.cell_arcsec, start_kms: p.start_kms, width_kms: p.width_kms };
    if (p.bmaj && p.bmin) body.beam = [+p.bmaj, +p.bmin, +(p.bpa || 0)];
    btn.disabled = true;
    try { const m = await api("/api/datasets/import", { method: "POST", body }); await linkReference(m.id); await loadDatasets(); selectDataset(m.id); }
    catch (e) { toast(e.message, true); btn.disabled = false; }
  };
  setMain(h("div", { class: "view-head" }, h("h1", {}, "Import existing cubes")),
    h("div", { class: "card" }, h("div", { class: "note" }, "Click a folder marked cubes, or click the dirty and PSF files. Without cell and beam the app still deconvolves, but cannot make a restored cube or show offsets in arcsec. A beam_arcsec_deg.npy next to the cubes is picked up automatically."), br),
    h("div", { class: "card" }, form, h("div", { class: "row" }, btn)));
}

// ============================================================ Dataset view
// The view in the address (#ds=…&tab=…&res=…, or #view=ms / import), so a reload comes back to it.
// replaceState: no history entry per click.
function saveView(v) {
  const q = new URLSearchParams(v);
  try { history.replaceState(null, "", `${location.pathname}${location.search}${q.toString() ? "#" + q : ""}`); } catch { }
}
function restoreView() {   // at start: the view in the address, if it still exists
  const q = new URLSearchParams(location.hash.slice(1)), ds = q.get("ds");
  if (q.get("view") === "ms") return viewMS(), true;
  if (q.get("view") === "import") return viewImport(), true;
  if (ds && S.datasets.some((d) => d.id === ds)) { S.cleanId = q.get("clean"); return selectDataset(ds, q.get("tab") || "overview", q.get("res") || undefined), true; }
  return false;
}

async function selectDataset(id, tab, resId) {
  S.linkTo = null;   // leaving the reference-adding flow
  S.dsId = id; if (tab) S.tab = tab; if (resId !== undefined) S.resId = resId;
  await loadDatasets();
  let m;
  try { m = await api(`/api/datasets/${id}`); } catch (e) { toast(e.message, true); return; }
  if (m.status !== "ready" && S.tab !== "overview") S.tab = "overview";
  saveView({ ds: id, tab: S.tab, ...(S.tab !== "overview" && S.resId ? { res: S.resId } : {}), ...(S.tab === "analysis" && S.cleanId ? { clean: S.cleanId } : {}) });
  const tabs = h("div", { class: "tabs" }, [["overview", "Overview"], ["deconvolve", "Deconvolve"], ["results", `Results (${m.results.length})`], ["analysis", "Analysis & Comparison(s)"]]
    .map(([k, label]) => h("button", { class: S.tab === k ? "on" : "", disabled: k !== "overview" && m.status !== "ready", onclick: () => selectDataset(id, k) }, label)));
  const body = h("div");
  const head = h("div", { class: "view-head" }, h("h1", {}, m.name),
    h("span", { class: "meta" }, `${m.nchan ?? "?"} channels · ${m.imsize ?? "?"} px · ${m.source === "ms" ? [].concat(m.ms).map((x) => x.split("/").filter(Boolean).pop()).join(" + ") : m.source}`),
    h("div", { style: "flex:1" }),
    h("label", { class: "note", style: "display:flex;gap:6px;align-items:center" },
      h("input", { type: "checkbox", checked: !!m.example, onchange: async (e) => { await api(`/api/datasets/${id}/meta`, { method: "POST", body: { example: e.target.checked } }); loadDatasets(); } }), "example"),
    h("button", { class: "danger", onclick: async () => {
      const b = head.querySelector(".confirm");
      if (!b) { head.append(h("span", { class: "confirm note" }, " delete dataset and all its results? ", h("button", { class: "danger", onclick: async () => { await api(`/api/datasets/${id}`, { method: "DELETE" }); S.dsId = null; await loadDatasets(); setMain($("#welcome-tpl") || welcome()); } }, "Delete"))); }
    } }, "Delete…"));
  setMain(head, tabs, body);
  if (S.tab === "overview") overviewTab(body, m);
  else if (S.tab === "deconvolve") deconvolveTab(body, m);
  else if (S.tab === "analysis") analysisTab(body, m);
  else resultsTab(body, m);
}

function welcome() {
  return h("section", { class: "empty" }, h("h1", {}, "Deconvolver 2D1D"), h("p", {}, "Pick a dataset on the left, or start one with the buttons above it."));
}

// shared display controls -------------------------------------------------
function displayControls(onChange) {
  const cm = h("select", {}, CMAPS.map((c) => h("option", { selected: c === S.disp.cmap }, c)));
  const st = h("select", {}, ["linear", "sqrt", "asinh"].map((c) => h("option", { selected: c === S.disp.stretch }, c)));
  const cl = h("select", {}, [99, 99.5, 99.9, 100].map((c) => h("option", { value: c, selected: c === S.disp.clip }, `${c}%`)));
  cm.onchange = () => { S.disp.cmap = cm.value; onChange(); };
  st.onchange = () => { S.disp.stretch = st.value; onChange(); };
  cl.onchange = () => { S.disp.clip = +cl.value; onChange(); };
  return [h("label", {}, "colour", cm), h("label", {}, "stretch", st), h("label", {}, "clip", cl)];
}

function chanLabel(m, k) {
  if (m.start_kms != null && m.width_kms) return `ch ${k} · ${(m.start_kms + k * m.width_kms).toFixed(1)} km/s`;
  return `ch ${k}`;
}

// ---- overview ----
function overviewTab(body, m) {
  if (m.status === "imaging") {
    body.append(h("p", { class: "note" }, "Dirty imaging is running."));
    if (m.job) body.append(jobMonitor({ id: m.job, label: "dirty imaging" }, { onDone: () => selectDataset(m.id) }));
    return;
  }
  if (m.status !== "ready") { body.append(h("div", { class: "card" }, h("b", {}, `status: ${m.status}`), m.error ? h("div", { class: "err-box" }, m.error) : null)); return; }
  const cell = m.cell_arcsec, unitD = m.width_kms ? "Jy/beam km/s" : "Jy/beam · ch", nchan = m.nchan || 1;
  const hasV = m.start_kms != null && !!m.width_kms, velOf = (k) => m.start_kms + k * m.width_kms;
  const kpcPerArcsec = cell && m.distance_mpc ? m.distance_mpc * 1e3 / 206264.806 : null;
  // Crop state. `base` is the rendered cut ({box, chans}, either may be null; null = the full cube):
  // the panels, slider and spectrum show it, and every crop option works inside it. Selections (a
  // square, a channel range) are previewed on the base; Render makes them the new base, Reset goes
  // back to the full cube. Boxes are inclusive dirty-image pixels {x0, x1, y0, y1} (the PSF panel's
  // view is offset differently, see toDirty); channel ranges are inclusive [c0, c1].
  let base = null, drawn = null, manualChans = null;
  const draw2 = { square: true, canDraw: () => manual.checked,
    onDrag: (b, p) => showBox(b ? toDirty(b, p) : drawn), onRegion: (b, p) => { drawn = toDirty(b, p); showBox(drawn); sync(); } };
  const pD = new ImagePanel({ title: "dirty, moment 0", unit: unitD, cell, cbar: "bottom", ...draw2 });
  pD.setBeam(m.beam);
  const pP = new ImagePanel({ title: "PSF, central channel", unit: "peak 1", cell, cbar: "bottom", ...draw2 });
  const hasTruth = m.source === "toy";
  const pT = hasTruth ? new ImagePanel({ title: "true sky, moment 0", unit: "Jy/pixel km/s", cell, cbar: "bottom" }) : null;
  const panels = [pD, pP, pT].filter(Boolean);
  const plane = h("select", {}, h("option", { value: "mom0" }, "moment 0"), h("option", { value: "chan" }, "channel"));
  const slider = h("input", { type: "range", min: 0, max: nchan - 1, value: Math.floor(nchan / 2), disabled: true });
  const get = arrayCache(m.imsize);
  const chans = () => base && base.chans || [0, nchan - 1];
  const baseBox = () => base && base.box || (pD.arr && { x0: 0, x1: pD.arr.nx - 1, y0: 0, y1: pD.arr.ny - 1 });
  const urls = (isCh, k) => {
    const [c0, c1] = chans(), whole = c0 === 0 && c1 === nchan - 1;
    const q = isCh ? `plane=chan&chan=${k}` : `plane=mom0${whole ? "" : `&c0=${c0}&c1=${c1}`}`;
    return [`/api/datasets/${m.id}/array?kind=dirty&${q}`,
      `/api/datasets/${m.id}/array?kind=psf&plane=chan&chan=${isCh ? k : Math.floor((c0 + c1) / 2)}`,
      hasTruth ? `/api/datasets/${m.id}/array?kind=truth&${q}` : null];
  };
  const label = () => {
    const isCh = plane.value === "chan"; slider.disabled = !isCh; pD.setTag(isCh ? chanLabel(m, +slider.value) : "");
    drawSpectrum();
  };
  const load = latestOnly(async () => {
    const isCh = plane.value === "chan", k = +slider.value;
    const [d, p, t] = await Promise.all(urls(isCh, k).map((u) => u && get(u)));
    pD.set(d); pP.set(p); if (pT) pT.set(t);
    const [c0, c1] = chans(), part = c0 === 0 && c1 === nchan - 1 ? "" : `, ch ${c0}–${c1}`;
    pD.setTitle(isCh ? "dirty, channel" : `dirty, moment 0${part}`); pD.setUnit(isCh ? "Jy/beam" : unitD);
    if (pT) { pT.setTitle(isCh ? "true sky, channel" : `true sky, moment 0${part}`); pT.setUnit(isCh ? "Jy/pixel" : "Jy/pixel km/s"); }
    await draw();
    // the neighbours are the likely next stop (arrow keys, small drags)
    if (isCh) for (const j of [k + 1, k - 1]) if (j >= c0 && j <= c1) urls(true, j).forEach((u) => u && get(u).catch(() => {}));
  });
  async function draw() {
    const L = await lut(S.disp.cmap);
    for (const p of [pD, pT]) if (p && p.arr) { const [lo, hi] = p.autoLimits(S.disp.clip); p.render(L, S.disp.stretch, lo, hi); }
    if (pP.arr) pP.render(L, "linear", -0.2, 1);
  }
  plane.onchange = slider.oninput = () => { label(); load(); };

  // ---- spectrum: over the field, or over the rendered crop; dashed line at the current channel;
  // the selected channel range shaded, with draggable edges in Manual range
  const specBox = h("div", { class: "spec" }), specUnit = h("span", { class: "punit" }), specTitle = h("b", {}, "spectrum, whole field");
  let spec = null;
  async function loadSpectrum() {
    const b = baseBox() || { x0: 0, x1: m.imsize - 1, y0: 0, y1: m.imsize - 1 };
    try { spec = await api(`/api/datasets/${m.id}/region`, { method: "POST", body: b }); }
    catch (e) { spec = null; specBox.textContent = e.message; return; }
    specTitle.textContent = base && base.box ? "spectrum, crop" : "spectrum, whole field";
    specUnit.textContent = spec.unit;
    drawSpectrum();
  }
  function drawSpectrum() {
    if (!spec) return;
    const [c0, c1] = chans(), cut = (a) => a.slice(c0, c1 + 1), x = cut(spec.axis);
    const col = { dirty: "--s-dirty", truth: "--s-truth" };
    const series = ["dirty", "truth"].filter((k) => spec[k]).map((k) => ({ name: k === "truth" ? "true sky" : k,
      color: cssVar(col[k]), x, y: cut(spec[k]), dash: k === "truth" ? "1 3" : "", width: k === "truth" ? 2 : 1.6 }));
    const vline = plane.value === "chan" ? spec.axis[+slider.value] : null;
    const sel = chanRange(), band = sel ? sel.map((k) => spec.axis[k]) : null;
    const svg = linePlot(series, { xlabel: spec.axis_label, ylabel: `flux density (${spec.unit})`, width: 320, height: 300, vline, band, left: 50 });
    if (band && manRange.checked) dragEdges(svg);
    if (plane.value === "chan") scrub(svg);
    specBox.replaceChildren(svg, series.length > 1 ? legend(series) : "");
  }
  function chanAt(svg, e) {   // the channel nearest the pointer, among those shown
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse()), v = svg.fromX(pt.x);
    const [c0, c1] = chans();
    let best = c0;
    for (let k = c0; k <= c1; k++) if (Math.abs(spec.axis[k] - v) < Math.abs(spec.axis[best] - v)) best = k;
    return best;
  }
  function scrub(svg) {   // channel mode: press or drag anywhere on the spectrum (or the dashed line) to pick the channel
    svg.classList.add("scrub");
    svg.addEventListener("pointerdown", (e) => {
      if (e.target.classList.contains("edge-hit")) return;   // that is a range edge being dragged
      svg.setPointerCapture(e.pointerId); e.preventDefault();
      const go = (ev) => {
        const k = chanAt(svg, ev);
        if (k === +slider.value) return;
        slider.value = k; paintSlider(slider); pD.setTag(chanLabel(m, k)); load();
        // move the line in place: redrawing the plot would end the drag
        const v = $(".vline", svg), x = svg.X(spec.axis[k]);
        if (v) { v.setAttribute("x1", x); v.setAttribute("x2", x); }
      };
      go(e);
      svg.onpointermove = go;
      svg.onpointerup = svg.onpointercancel = () => { svg.onpointermove = svg.onpointerup = svg.onpointercancel = null; };
    });
  }
  function dragEdges(svg) {   // move the Manual range edges; snaps to channels
    $$(".edge-hit", svg).forEach((line) => line.addEventListener("pointerdown", (e) => {
      const i = +line.dataset.edge; line.setPointerCapture(e.pointerId); e.preventDefault();
      const moveTo = (ev) => {
        const r = [...manualChans]; r[i] = chanAt(svg, ev);
        if (r[1 - i] === r[i]) return;
        manualChans = r[0] <= r[1] ? r : [r[1], r[0]];
        const xs = manualChans.map((k) => svg.X(spec.axis[k])), rect = $(".band", svg);
        rect.setAttribute("x", Math.min(...xs)); rect.setAttribute("width", Math.abs(xs[1] - xs[0]));
        for (const e2 of $$(".edge, .edge-hit", svg)) { const xx = xs[+e2.dataset.edge]; e2.setAttribute("x1", xx); e2.setAttribute("x2", xx); }
        sync();
      };
      line.onpointermove = moveTo;
      line.onpointerup = () => { line.onpointermove = line.onpointerup = null; drawSpectrum(); };
    }));
  }

  // ---- crop controls
  const ACCENT = "#e3b341";   // the dark-theme accent, in both themes: it has to read on the colour maps
  const fov = (n) => {   // side of a square of n px, in every unit we know
    const parts = [cell ? `${fmt(n * cell, 3)}″ × ${fmt(n * cell, 3)}″` : null, kpcPerArcsec ? `${fmt(n * cell * kpcPerArcsec, 3)} × ${fmt(n * cell * kpcPerArcsec, 3)} kpc` : null];
    return [parts.filter(Boolean).join("  ·  "), `${n} × ${n} px`].filter(Boolean);
  };
  // the dirty image and true sky share pixels; the PSF panel shows its own centred cut, so a box
  // moves between them by the offset of the two views
  const origin = (p) => p.crop ? [p.crop[0], p.crop[2]] : [0, 0];
  const shift = (b, dx, dy) => ({ x0: b.x0 + dx, x1: b.x1 + dx, y0: b.y0 + dy, y1: b.y1 + dy });
  function toDirty(b, p) { const [a, c] = origin(p), [d, e] = origin(pD); return shift(b, d - a, e - c); }
  const showBox = (b) => {
    for (const p of panels) {
      const [a, c] = origin(p), [d, e] = origin(pD);
      p.setBox(b ? { ...shift(b, a - d, c - e), color: ACCENT, label: fov(b.x1 - b.x0 + 1) } : null);
    }
  };
  const opt = (v, t, sel) => h("option", { value: v, selected: sel }, t);
  // spatial
  const byCentre = h("input", { type: "checkbox" }), manual = h("input", { type: "checkbox" });
  const size = h("input", { type: "number", min: 0, step: "any", class: "num" });
  const sizeUnit = h("select", {}, opt("px", "px", !cell), cell ? opt("arcsec", "″", true) : null, kpcPerArcsec ? opt("kpc", "kpc") : null);
  // spectral
  const bySys = h("input", { type: "checkbox" }), manRange = h("input", { type: "checkbox" });
  const width = h("input", { type: "number", min: 0, step: "any", class: "num" });
  const widthUnit = h("select", {}, opt("ch", "ch", !hasV), hasV ? opt("kms", "km/s", true) : null);
  const vsys = hasV ? h("input", { type: "number", step: "any", class: "num", value: m.vsys_kms ?? +velOf((nchan - 1) / 2).toFixed(1) }) : null;
  const renderBtn = h("button", { class: "primary" }, "Render"), resetBtn = h("button", {}, "Reset");
  const saveMem = h("button", {}, "Save in memory"), saveDisk = h("button", {}, "Save on disk…");
  const hint = h("div", { class: "hint" });

  function spatialBox() {
    if (manual.checked) return drawn;
    const v = baseBox();
    if (!byCentre.checked || !v || !(+size.value > 0)) return null;
    const perPx = { px: 1, arcsec: cell, kpc: cell * kpcPerArcsec }[sizeUnit.value], w = v.x1 - v.x0 + 1, hh = v.y1 - v.y0 + 1;
    const n = Math.min(Math.round(+size.value / perPx), w, hh);
    if (n < 8) return null;
    const x0 = v.x0 + Math.floor(w / 2) - Math.floor(n / 2), y0 = v.y0 + Math.floor(hh / 2) - Math.floor(n / 2);   // centre of the view
    return { x0, x1: x0 + n - 1, y0, y1: y0 + n - 1 };
  }
  function chanRange() {
    if (manRange.checked) return manualChans;
    if (!bySys.checked || !(+width.value > 0)) return null;
    const [b0, b1] = chans();
    const n = Math.min(b1 - b0 + 1, widthUnit.value === "ch" ? Math.round(+width.value) : Math.round(+width.value / Math.abs(m.width_kms)));
    if (n < 2) return null;
    const kc = vsys && vsys.value !== "" ? (+vsys.value - m.start_kms) / m.width_kms : (b0 + b1) / 2;
    const c0 = Math.min(b1 - n + 1, Math.max(b0, Math.round(kc - (n - 1) / 2)));
    return [c0, c0 + n - 1];
  }
  const chanText = ([c0, c1]) => `channels ${c0}–${c1} (${c1 - c0 + 1})` + (hasV ? `, ${fmt(velOf(c0), 5)} to ${fmt(velOf(c1), 5)} km/s` : "");

  function sync() {   // enable/disable controls, previews and the hint from the current state
    size.disabled = sizeUnit.disabled = !byCentre.checked;
    width.disabled = widthUnit.disabled = !bySys.checked; if (vsys) vsys.disabled = !bySys.checked;
    const box = spatialBox(), cr = chanRange(), any = !!(box || cr);
    showBox(box);
    renderBtn.disabled = !any;
    resetBtn.disabled = !any && !base;
    saveMem.disabled = !any && !base && !geoChanged();
    const cut = (b, c) => [b ? fov(b.x1 - b.x0 + 1).join(", ") : null, c ? chanText(c) : null].filter(Boolean).join(" · ");
    saveDisk.title = any || base ? `Saves the cut (${cut(box || base && base.box, cr || base && base.chans)}) as a .pkl` : "Saves this dataset as a .pkl";
    hint.textContent = [base ? `Rendered: ${cut(base.box, base.chans)}.` : null,
      box || cr ? cut(box, cr) : manual.checked ? "Drag a square on the dirty image or the PSF; drag inside it to move it." : null]
      .filter(Boolean).join("\n");
  }
  function setView(r) {   // r: {box, chans} to show the cut, or null for the full cube
    base = r;
    const b = r && r.box, n = b ? b.x1 - b.x0 + 1 : 0;
    for (const p of [pD, pT]) if (p) p.crop = b ? [b.x0, b.x1, b.y0, b.y1] : null;
    if (pP.arr) {   // the PSF is shift-invariant: the solvers use it centred on its peak, whatever the box
      const c = Math.floor(pP.arr.nx / 2), rr = Math.floor(pP.arr.ny / 2), o = Math.floor(n / 2);
      pP.crop = b ? [c - o, c - o + n - 1, rr - o, rr - o + n - 1] : null;
    }
    pP.setTitle(b ? `PSF, central ${n} px` : "PSF, central channel");
    for (const p of panels) p._auto = null;
    const [c0, c1] = chans();
    slider.min = c0; slider.max = c1; slider.value = Math.min(c1, Math.max(c0, +slider.value));
    label(); load().then(loadSpectrum); sync();
  }
  const edited = () => { sync(); drawSpectrum(); };
  const clearSelections = () => { for (const c of [byCentre, manual, bySys, manRange]) c.checked = false; drawn = manualChans = null; };
  const exclusive = (a, b, clear) => () => { if (a.checked) b.checked = false; clear(); edited(); };
  byCentre.onchange = exclusive(byCentre, manual, () => { drawn = null; });
  manual.onchange = exclusive(manual, byCentre, () => { if (!manual.checked) drawn = null; });
  bySys.onchange = exclusive(bySys, manRange, () => {});
  manRange.onchange = exclusive(manRange, bySys, () => {
    const [b0, b1] = chans();
    manualChans = manRange.checked ? (chanRange() || [b0 + Math.round((b1 - b0) * 0.25), b0 + Math.round((b1 - b0) * 0.75)]) : null;
  });
  for (const el of [size, sizeUnit, width, widthUnit, vsys].filter(Boolean)) el.oninput = edited;
  // what Render would show: the selections, falling back to the base for whatever is not selected
  const merged = () => ({ box: spatialBox() || base && base.box || null, chans: chanRange() || base && base.chans || null });
  renderBtn.onclick = () => { const r = merged(); clearSelections(); setView(r); };
  resetBtn.onclick = () => { clearSelections(); setView(null); };
  const cutRequest = () => {
    const r = merged();
    return { box: r.box, c0: r.chans ? r.chans[0] : null, c1: r.chans ? r.chans[1] : null, geometry: geometry() };
  };
  saveMem.onclick = async () => {
    const req = cutRequest();
    saveMem.disabled = true;
    try {
      if (!req.box && req.c0 === null) {   // nothing cut: keep the edited geometry on this dataset
        await api(`/api/datasets/${m.id}/meta`, { method: "POST", body: req.geometry });
        toast("Saved"); return selectDataset(m.id);
      }
      const nm = await api(`/api/datasets/${m.id}/crop`, { method: "POST", timeout: 600000, body: req });
      toast(`Saved as "${nm.name}"`); await loadDatasets(); selectDataset(nm.id);
    } catch (e) { toast(e.message, true); sync(); }
  };
  saveDisk.onclick = () => {
    const req = cutRequest(), box = req.box, n = box ? box.x1 - box.x0 + 1 : null;
    const stem = [m.name, n ? `${n}px` : null, req.c0 !== null ? `ch${req.c0}-${req.c1}` : null].filter(Boolean).join("_");
    saveOnDiskDialog(`${stem.replace(/[^A-Za-z0-9_.-]+/g, "_")}.pkl`, { id: m.id, req });
  };
  const line = (...k) => h("div", { class: "opt" }, ...k);
  const cropEl = h("div", { class: "crop" }, h("h3", {}, "Crop"),
    h("h4", {}, "Spatial"),
    line(h("label", { class: "opt" }, byCentre, "Central crop"), size, sizeUnit),
    line(h("label", { class: "opt" }, manual, "Manual square crop")),
    h("h4", {}, "Spectral"),
    line(h("label", { class: "opt" }, bySys, "Systemic crop"), width, widthUnit),
    vsys ? line(h("span", { class: "sub" }, "around v", h("sub", {}, "sys")), vsys, h("span", { class: "sub" }, "km/s")) : null,
    line(h("label", { class: "opt" }, manRange, "Manual range")),
    hint,
    h("div", { class: "row buttons" }, renderBtn, resetBtn, saveMem));

  // ---- geometry: from CASA imaging it is fixed (read only); for imported cubes it can be set here,
  // and is kept by Save in memory / Save on disk
  const casa = m.source === "ms";
  const geoField = (name, labelText, v) => h("label", {}, labelText, h("input", { name, type: "number", step: "any", value: v ?? "", disabled: casa }));
  const metaForm = h("form", { class: "form", onsubmit: (e) => e.preventDefault(), oninput: () => sync() },
    geoField("cell_arcsec", "cell (arcsec)", m.cell_arcsec),
    // the clean beam is listed under Dataset; it is only set here for imported cubes
    casa ? null : geoField("bmaj", "beam major (″)", m.beam && m.beam[0]),
    casa ? null : geoField("bmin", "beam minor (″)", m.beam && m.beam[1]),
    casa ? null : geoField("bpa", "beam PA (deg)", m.beam && m.beam[2]),
    geoField("width_kms", "channel width (km/s)", m.width_kms));
  function geometry() {
    if (casa) return null;
    const p = Object.fromEntries(new FormData(metaForm).entries()), num = (v) => v === "" ? null : +v;
    return { cell_arcsec: num(p.cell_arcsec), width_kms: num(p.width_kms), beam: p.bmaj && p.bmin ? [+p.bmaj, +p.bmin, +(p.bpa || 0)] : null };
  }
  function geoChanged() {
    const g = geometry();
    return !!g && JSON.stringify([g.cell_arcsec, g.width_kms, g.beam]) !== JSON.stringify([m.cell_arcsec ?? null, m.width_kms ?? null, m.beam ?? null]);
  }
  // ---- uv coverage of the measurement sets this cube was imaged from (square, as tall as the Dataset card)
  const msList = [].concat(m.ms || []);
  const uvBox = h("div", { class: "uv-box" }, h("div", { class: "note" }, "reading the visibilities…"));
  const uvLegend = h("div", { class: "legend" });
  // the slot takes the Dataset card's height (the card inside is absolutely placed, so it adds none) and
  // is given the same width below, so the plot is square
  const uvCard = msList.length ? h("div", { class: "uv-slot" }, h("div", { class: "card uv-card" }, h("h3", {}, "uv coverage"), uvBox, uvLegend)) : null;
  async function loadUV() {
    try {
      const r = await api(`/api/datasets/${m.id}/uv`, { timeout: 600000 });
      // kλ at the observed frequency of the cube's central channel (radio velocity)
      const rest = +(m.restfreq_ghz || (m.imaging || {}).restfreq_ghz || 0);
      const vc = m.start_kms != null && m.width_kms ? m.start_kms + (nchan - 1) / 2 * m.width_kms : 0;
      const ghz = rest ? rest * (1 - vc / 299792.458) : 230;
      const colors = ["var(--accent)", "color-mix(in srgb, var(--accent) 45%, var(--fg))", "color-mix(in srgb, var(--accent) 55%, var(--bg))", "var(--muted)"];
      const sets = r.sets.filter((x) => x.u.length).map((x, i) => ({ ...x, ghz, color: colors[i % colors.length],
        // the execution block's id, without the uid___A002_ every ALMA MS shares (full path on hover)
        name: x.ms.split("/").filter(Boolean).pop().replace(/\.ms.*$/, "").replace(/^uid___A\d+_/, "") }));
      if (!sets.length) { uvBox.replaceChildren(h("div", { class: "note" }, "no baselines found for this field")); return; }
      uvBox.replaceChildren(uvPlot(sets, 300));   // drawn small: the card is about 350 px
      uvLegend.replaceChildren(...sets.map((x) => h("span", { title: `${x.ms}\n${x.u.length} of ${x.rows} baselines (one spw), and their conjugates` },
        h("i", { style: `background:${x.color}` }), x.name)));
    } catch (e) { uvBox.replaceChildren(h("div", { class: "note" }, e.message)); }
  }
  const im = m.imaging || {};
  const kv = h("table", { class: "kv" },
    m.ms ? h("tr", {}, h("td", {}, "MS"), h("td", { class: "mono" }, [].concat(m.ms).map((x) => h("div", {}, x)))) : null,
    im.field ? h("tr", {}, h("td", {}, "field / spw"), h("td", {}, `${im.field} / ${im.spw_list ? im.spw_list.join(", ") : im.spw}`)) : null,
    im.phasecenter ? h("tr", {}, h("td", {}, "phase centre"), h("td", { class: "mono" }, im.phasecenter)) : null,
    im.restfreq_ghz ? h("tr", {}, h("td", {}, "rest frequency"), h("td", {}, `${im.restfreq_ghz} GHz, ${im.weighting} weighting`)) : null,
    m.imported_from ? h("tr", {}, h("td", {}, "imported from"), h("td", { class: "mono" }, m.imported_from.map((x) => h("div", {}, x)))) : null,
    m.loaded_from ? h("tr", {}, h("td", {}, "loaded from"), h("td", { class: "mono" }, m.loaded_from)) : null,
    m.beam ? h("tr", {}, h("td", {}, "clean beam"), h("td", {}, `${fmt(m.beam[0])}″ × ${fmt(m.beam[1])}″ @ ${fmt(m.beam[2])}°`)) : null,
    // geometry from the CASA imaging is fixed: listed here (imported cubes edit it below)
    casa && m.cell_arcsec ? h("tr", {}, h("td", {}, "cell"), h("td", {}, `${fmt(m.cell_arcsec, 4)}″`)) : null,
    casa && m.width_kms ? h("tr", {}, h("td", {}, "channel width"), h("td", {}, `${fmt(m.width_kms, 4)} km/s`)) : null,
    m.note ? h("tr", {}, h("td", {}, "note"), h("td", {}, m.note)) : null,
    m.crop ? h("tr", {}, h("td", {}, "cropped from"), h("td", {}, `${m.crop.parent_name}: x ${m.crop.box[0]}–${m.crop.box[1]}, y ${m.crop.box[2]}–${m.crop.box[3]} of ${m.crop.parent_size.join(" × ")} px` +
      (m.crop.chans ? `, channels ${m.crop.chans[0]}–${m.crop.chans[1]} of ${m.crop.parent_nchan}` : ""))) : null,
    h("tr", {}, h("td", {}, "stored in"), h("td", { class: "mono" }, `${S.info.workdir}/datasets/${m.id}`)));
  body.append(
    h("div", { class: "card" }, h("div", { class: "ov-wrap" },
      h("div", { class: "ov-main" }, h("div", { class: "controls" }, h("label", {}, "show", plane), slider, ...displayControls(draw)),
        h("div", { class: "ov-grid" }, ...panels.map((p) => p.el),
          h("div", { class: "panel" }, h("div", { class: "ptitle" }, specTitle, specUnit), specBox))), cropEl)),
    h("div", { class: "ds-row" },
      h("div", { class: "card ds-card" }, h("h3", {}, "Dataset"), kv,
        casa ? null : h("h4", { class: "sub-head" }, "Geometry and units"), casa ? null : metaForm,
        h("div", { class: "row" }, saveDisk)),
      uvCard));
  sync();
  load().then(loadSpectrum).catch((e) => toast(e.message, true));
  if (uvCard) {
    const dsCard = $(".ds-card", body);
    new ResizeObserver(() => { if (getComputedStyle(uvCard).position !== "static" || innerWidth > 900) uvCard.style.width = `${Math.round(dsCard.getBoundingClientRect().height)}px`; })
      .observe(dsCard);
    loadUV();
  }
}

// ---- deconvolve ----
// a run's method, from its parameters: primal-dual is "pdtau" when reweighted with the tau*lambda reference
function methodName(p) {
  if (!p || !p.backend) return "–";
  if (p.backend.endsWith("fista")) return "FISTA";
  const tau = p.backend === "gpu_pd" && p.reweight && (p.reweight_ref || "tau_lambda") === "tau_lambda";
  return tau ? "primal-dual (pdtau)" : "primal-dual";
}
const DEFAULTS = {
  // the recipe of the scripts (pdtau: scripts/experiments, FISTA: make_fista_run.py)
  gpu_fista: { n_iter: 20000, k_sigma: 2, reweight: true, burn_in_iters: 6000, reweight_every: 2000, reweight_eps: 1.0, rebin: 1 },
  gpu_pd: { n_iter: 20000, k_sigma: 3, reweight: true, burn_in_iters: 6000, reweight_every: 2000, reweight_eps: 1.0, rebin: 1 },
  pd: { n_iter: 300, k_sigma: 3, reweight: false, burn_in_iters: 100, reweight_every: 25, reweight_eps: 0.01, positivity: true, tol: 1e-4 },
  fista: { n_iter: 60, k_sigma: 4, reweight: true, burn_in_iters: 20, reweight_eps: 0.01, positivity: true },
};

function deconvolveTab(body, m) {
  const mon = h("div");
  const form = h("form", { class: "form", onsubmit: (e) => e.preventDefault() });
  const hasTorch = !!(S.devices && S.devices.torch);
  // the method; its implementation follows from the devices: the PyTorch solvers (gpu_pd.py) run on any ticked
  // device, GPU or CPU; without PyTorch, the numpy solvers (deconvolve_pd.py, deconvolve.py) on the CPU
  const backend = h("select", { name: "backend" },
    h("option", { value: "pd" }, "Primal-dual (pdtau)"),
    h("option", { value: "fista" }, "FISTA"));
  backend.value = "pd";
  const impl = () => (hasTorch ? (backend.value === "pd" ? "gpu_pd" : "gpu_fista") : backend.value);
  const devBox = h("div", { class: "devices" });
  function drawDevices() {
    const list = (S.devices && S.devices.devices) || [{ id: "cpu", name: "CPU", kind: "cpu" }];
    const gpuOK = hasTorch;
    if (!S.devSel) { const g = list.find((d) => d.kind !== "cpu"); S.devSel = new Set([g ? g.id : "cpu"]); }
    devBox.innerHTML = "";
    for (const d of list) {
      const enabled = gpuOK || d.kind === "cpu";
      const on = gpuOK ? S.devSel.has(d.id) : d.kind === "cpu";
      const cb = h("input", { type: "checkbox", checked: on, disabled: !enabled, "data-dev": d.id,
        onchange: (e) => { if (e.target.checked) S.devSel.add(d.id); else S.devSel.delete(d.id); } });
      devBox.append(h("label", { class: "check dev" + (enabled ? "" : " off") }, cb,
        h("span", {}, h("b", {}, d.id), " ", h("span", { class: "note" }, d.name + (d.mem_gb ? `, ${d.mem_gb} GB` : "")))));
    }
    devBox.append(h("div", { class: "hint" }, gpuOK
      ? "The solver runs (PyTorch) on the device ticked, GPU or CPU. One run uses one device; tick several to spread a sweep (or several queued runs) across them, each device taking one run at a time."
      : "PyTorch is not installed: the numpy versions run, on the CPU only."));
  }
  function fill() {
    const b = impl(), d = DEFAULTS[b];
    form.innerHTML = "";
    drawDevices();
    const num = (name, label, val, hint, step = "any") => h("label", {}, label, h("input", { name, type: "number", step, value: val ?? "" }), hint ? h("span", { class: "hint" }, hint) : null);
    const chk = (name, label, val) => h("label", { class: "check" }, h("input", { name, type: "checkbox", checked: !!val }), label);
    form.append(...[
      h("label", {}, "solver", backend),
      num("n_iter", "iterations (max)", d.n_iter, null, 1),
      h("label", {}, "threshold k (× noise σ)", h("input", { name: "k_sigma", value: d.k_sigma }), h("span", { class: "hint" }, "comma-separated for a sweep, e.g. 2, 3, 4")),
      b.startsWith("gpu_") ? num("rebin", "rebin factor", d.rebin, "b×b block binning before solving; the model is upsampled back", 1) : null,
      num("num_scales_2d", "spatial starlet scales", "", "empty = deepest", 1),
      b.startsWith("gpu_") ? h("label", {}, "2D dictionary", h("input", { name: "dict_2d", list: "dict2d-names", placeholder: "starlet" }),
        h("datalist", { id: "dict2d-names" }, ["starlet", "learnlet", "learnlet:16"].map((v) => h("option", { value: v }))),
        h("span", { class: "hint" }, "starlet, or any dictionaries/<name>.npz (learnlet: ImageNet-trained filters); <name>:N keeps N filters per scale. A learnt bank costs ~(1 + filters)× the starlet's memory on the scales it covers")) : null,
      num("num_levels_1d", "spectral CDF 9/7 levels", "", "empty = deepest", 1),
      h("label", {}, "line-free channels", h("input", { name: "noise_channels", placeholder: "e.g. 0-9, 80-89" }), h("span", { class: "hint" }, b.startsWith("gpu_") ? "set the thresholds from the real correlated noise; empty = detected automatically (see the log)" : "measures the real correlated noise per scale")),
      b === "fista" ? num("lrs_arcsec", "largest recoverable scale (″)", "", "zeroes larger starlet planes") : null,
      b === "pd" ? num("tol", "stop when relative change <", d.tol) : null,
      b === "gpu_pd" ? h("label", {}, "thresholds λ_b from", h("select", { name: "lambda_calibration" },
        h("option", { value: "noise" }, "noise coefficients (gpu_pd.noise_lambdas, as the scripts)"),
        h("option", { value: "dual" }, "dual certificate of noise (experimental, larger λ)")),
        h("span", { class: "hint" }, "primal-dual only; see the note below the form")) : null,
      b === "gpu_pd" ? h("label", {}, "reweighting reference", h("select", { name: "reweight_ref" },
        h("option", { value: "tau_lambda" }, "τλ, the threshold in model units (pdtau, recommended)"),
        h("option", { value: "lambda" }, "λ (plain pd: ~β× too weak, almost no reweighting)"))) : null,
      b.startsWith("gpu_") ? chk("null_test", "null test: run on pure noise instead of the data", false) : null,
      b.startsWith("gpu_") ? num("tol", "stop when relative change <", "", "per iteration, checked ~200 times; empty = run all iterations") : null,
      num("frame_every", "snapshot every N iterations", "", "saved for the GIF and the scrubber, not the live view (that shows every iteration); empty = ~115 log-spaced snapshots (as the scripts' GIFs); at most 1000 are kept"),
      b.startsWith("gpu_") ? null : chk("positivity", "positivity", d.positivity),
      chk("reweight", "reweighted L1", d.reweight),
      num("burn_in_iters", "reweighting starts at iteration", d.burn_in_iters, null, 1),
      b !== "fista" ? num("reweight_every", "reweight every", d.reweight_every, null, 1) : null,
      num("reweight_eps", "reweighting ε", d.reweight_eps),
      h("label", {}, "restoring beam (″, ″, °)", h("input", { name: "restore_beam", placeholder: "bmaj, bmin, pa  (or one FWHM)" }),
        h("span", { class: "hint" }, "for the restored cube; empty = the dataset's clean beam, which hides any super-resolution: use a smaller beam (e.g. the longest-baseline resolution) to see it")),
    ].filter(Boolean));
  }
  backend.onchange = fill; fill();
  const run = h("button", { class: "primary" }, "Run deconvolution");
  run.onclick = async () => {
    const p = { backend: impl() };
    for (const el of form.elements) {
      if (!el.name || el.name === "backend") continue;
      if (el.type === "checkbox") p[el.name] = el.checked;
      else if (el.value !== "") p[el.name] = el.type === "number" ? +el.value : el.value;
    }
    const ks = String(p.k_sigma).split(/[,\s]+/).filter(Boolean).map(Number);
    if (!ks.length || ks.some(Number.isNaN)) return toast("threshold k: a number, or several separated by commas", true);
    p.k_sigma = ks.length > 1 ? ks : ks[0];
    p.devices = hasTorch ? [...S.devSel] : ["cpu"];
    if (!p.devices.length) return toast("choose at least one device", true);
    run.disabled = true;
    try {
      const r = await api(`/api/datasets/${m.id}/deconvolve`, { method: "POST", body: { params: p } });
      if (r.runs.length === 1) showMon(r.job, r.result, p);
      else showSweep(r.runs);
      run.disabled = false;      // more runs can be queued while these run
      loadDatasets();
    } catch (e) { toast(e.message, true); run.disabled = false; }
  };
  function showSweep(runs) {
    mon.innerHTML = "";
    mon.append(h("div", { class: "row", style: "margin-bottom:10px" }, h("b", {}, `${runs.length} runs queued`),
      h("span", { class: "note" }, "each takes the next free device; open Results when they finish")));
    for (const r of runs) mon.append(h("div", { style: "margin-bottom:10px" }, jobMonitor(r.job, { onDone: () => loadDatasets() })));
  }
  function showMon(job, resId, params) {
    mon.innerHTML = "";
    const dv = Math.abs(m.width_kms || 1);
    mon.append(jobMonitor(job, { live: true, cell: m.cell_arcsec, dv, marks: reweightMarks(params), onDone: (j) => {
      loadDatasets();
      if (j.status === "done") { const b = h("button", { class: "primary", onclick: () => selectDataset(m.id, "results", resId) }, "Open result"); mon.prepend(h("div", { class: "row", style: "margin-bottom:10px" }, h("b", {}, "Finished."), b)); }
    } }));
  }
  body.append(h("div", { class: "card" }, h("h3", {}, "Deconvolution parameters"),
    h("div", { class: "note" }, "Every run starts a fresh worker that imports simple/ from the repository at that moment, so your latest edits are used. Parameters a solver does not accept are ignored and reported in the log."),
    h("h3", {}, "Devices"), devBox,
    h("h3", {}, "Solver"), form,
    h("div", { class: "note" }, "Null test: the cube is replaced by a noise realization with the power spectrum measured in the line-free channels. A good threshold setting gives a model of (nearly) zero flux. For the primal-dual solver, λ_b set from the noise's own wavelet coefficients does not pass this test: the optimum keeps x = 0 only if the noise can be written as Wᵀu with |u| ≤ λ, and for the redundant 2D-1D frame those u are larger and spread differently over the sub-bands than W n. 'Dual certificate' sets λ_b = k × MAD of u = W(WᵀW)⁻¹n per sub-band."),
    h("div", { class: "row" }, run)), h("div", { style: "margin-top:14px" }, mon));
  // A run on this dataset that is still going (e.g. started before switching tab): show its live view.
  // The live job list is the authority: a result records its job id only when the run ends.
  api("/api/jobs", { timeout: 10000 }).then((jobs) => {
    const live = jobs.filter((j) => j.kind === "deconvolve" && j.ds_id === m.id && (j.status === "running" || j.status === "queued"));
    if (!live.length || mon.childElementCount) return;   // nothing going, or a run was just started here
    const paramsOf = (j) => ((m.results || []).find((r) => r.id === j.res_id) || {}).params;
    const show = (j) => {
      showMon({ id: j.id, label: `${methodName(paramsOf(j))} ${j.res_id || ""} (${j.status})` }, j.res_id, paramsOf(j));
      const others = live.filter((o) => o !== j);
      if (others.length) mon.prepend(h("div", { class: "row", style: "margin-bottom:10px" }, h("span", { class: "note" }, "Also running:"),
        others.map((o) => h("button", { onclick: () => show(o) }, `${o.res_id || o.id}${paramsOf(o) && paramsOf(o).k_sigma != null ? ` (k = ${paramsOf(o).k_sigma})` : ""}`))));
    };
    show(live.find((j) => j.res_id === S.resId) || live.find((j) => j.status === "running") || live[0]);
  }).catch(() => {});
}

// ---- results ----
// The deconvolution results of a dataset as a table to pick one from (Results and Analysis tabs). A running
// one opens its live progress; `tab` is where picking a finished one goes.
function resultTable(m, res, tab) {
  const live = (r) => r.status === "running" || r.status === "queued";
  return h("table", { class: "list" }, h("tr", {}, ["result", "solver", "device", "k", "iterations", "flux (Jy)", "status", "started", ""].map((t) => h("th", {}, t))),
    res.map((r) => h("tr", { class: "click" + (r.id === S.resId ? " sel" : ""), title: live(r) ? "show its live progress" : "",
      onclick: () => selectDataset(m.id, live(r) ? "deconvolve" : tab, r.id) },
      h("td", { class: "mono" }, r.id), h("td", {}, methodName(r.params)), h("td", { class: "mono" }, r.device || "–"), h("td", {}, r.params.k_sigma ?? "–"),
      h("td", {}, r.summary ? r.summary.iterations : "–"), h("td", {}, r.summary ? fmt(r.summary.flux, 4) : "–"),
      h("td", {}, h("span", { class: "chip " + (r.status === "done" ? "ok" : live(r) ? "run" : "err") }, r.status)),
      h("td", {}, ago(r.created)),
      h("td", {}, h("button", { class: "ghost danger", title: "delete this result", onclick: async (e) => { e.stopPropagation(); await api(`/api/datasets/${m.id}/results/${r.id}`, { method: "DELETE" }); S.resId = null; selectDataset(m.id, tab); } }, "✕")))));
}

async function resultsTab(body, m) {
  const res = (m.results || []).slice().reverse();
  if (!res.length) { body.append(h("p", { class: "note" }, "No results yet. Run a deconvolution first.")); return; }
  if (!S.resId || !res.find((r) => r.id === S.resId)) S.resId = (res.find((r) => r.status === "done") || res[0]).id;
  body.append(h("div", { class: "card scroll-x" }, resultTable(m, res, "results")));
  const r = await api(`/api/datasets/${m.id}/results/${S.resId}`);
  if (r.status !== "done") {
    body.append(h("div", { class: "card", style: "margin-top:14px" }, h("div", { class: "row" }, h("b", {}, `status: ${r.status}`),
      r.status === "running" || r.status === "queued" ? h("button", { class: "primary", onclick: () => selectDataset(m.id, "deconvolve", r.id) }, "Show live progress") : null),
      r.error ? h("div", { class: "err-box" }, r.error) : null));
    return;
  }
  const cell = m.cell_arcsec, hasV = !!m.width_kms, hasTruth = m.source === "toy", hasRestored = r.files.includes("restored.npy");
  const regionCb = (b) => setRegion(b);
  const summ = r.summary || {}, rbeam = summ.restored_beam || null;
  const opt = { cell, onRegion: regionCb, cbar: "bottom" };
  const P = {
    dirty: new ImagePanel({ title: "dirty", ...opt }),
    model: new ImagePanel({ title: "model", ...opt }),
    restored: hasRestored ? new ImagePanel({ title: "restored", ...opt }) : null,
    truth: hasTruth ? new ImagePanel({ title: "true sky", ...opt }) : null,
  };
  P.dirty.setBeam(m.beam);
  if (P.restored) P.restored.setBeam(rbeam);
  // all maps in surface brightness, per arcsec² (per pixel² without a cell): Jy/pixel maps divided by the
  // pixel area, Jy/beam maps by their beam's area (π bmaj bmin / 4 ln 2), so the four share one unit
  const beamArea = (b) => Math.PI * b[0] * b[1] / (4 * Math.LN2);
  const perPix = cell ? 1 / (cell * cell) : 1;
  const sbScale = { model: perPix, truth: perPix, dirty: cell && m.beam ? 1 / beamArea(m.beam) : null,
    restored: cell && rbeam ? 1 / beamArea(rbeam) : null };
  const sbUnit = (kind, isCh) => {
    const v = isCh ? "" : hasV ? " km/s" : " ch";
    return sbScale[kind] === null ? `Jy/beam${v}` : `Jy${v} ${cell ? "arcsec⁻²" : "px⁻²"}`;
  };
  const scaled = (arr, f) => (!arr || f === null || f === 1) ? arr : { ...arr, data: Float32Array.from(arr.data, (v) => v * f) };
  const plane = h("select", {}, h("option", { value: "mom0" }, "moment 0"), h("option", { value: "chan" }, "channel"));
  const slider = h("input", { type: "range", min: 0, max: (m.nchan || 1) - 1, value: Math.floor((m.nchan || 1) / 2), disabled: true });
  const link = h("input", { type: "checkbox", checked: true });
  const zoomBox = h("div", { class: "card", style: "margin-top:14px" }, h("div", { class: "note" }, "Drag a box on any image to zoom in and get integrated spectra of that region."));
  let region = null;

  const get = arrayCache(m.imsize), kinds = Object.keys(P).filter((k2) => P[k2]);
  const urls = (isCh, k) => kinds.map((kind) => `/api/datasets/${m.id}/array?kind=${kind}&res=${S.resId}&${isCh ? `plane=chan&chan=${k}` : "plane=mom0"}`);
  const label = () => { const isCh = plane.value === "chan"; slider.disabled = !isCh; P.dirty.setTag(isCh ? chanLabel(m, +slider.value) : ""); };
  const load = latestOnly(async () => {
    const isCh = plane.value === "chan", k = +slider.value;
    const arrs = await Promise.all(urls(isCh, k).map((u) => get(u).catch(() => null)));
    kinds.forEach((kind, i) => {
      const p = P[kind]; p.set(scaled(arrs[i], sbScale[kind]));
      p.setUnit(sbUnit(kind, isCh));
      p.setTitle(`${kind === "truth" ? "true sky" : kind}${isCh ? "" : ", moment 0"}`);
    });
    await draw();
    if (!spec) loadSpectrum();
    if (isCh) for (const j of [k + 1, k - 1]) if (j >= 0 && j < (m.nchan || 1)) urls(true, j).forEach((u) => get(u).catch(() => {}));
  });
  async function draw() {
    const L = await lut(S.disp.cmap);
    let shared = null;
    // common scale shares the top only (brightness comparable); each panel keeps its own floor, so the
    // restored cube is not dragged below 0 by the dirty image's negative noise and sidelobes
    if (link.checked && P.dirty.arr && P.restored && P.restored.arr)
      shared = Math.max(P.dirty.autoLimits(S.disp.clip)[1], P.restored.autoLimits(S.disp.clip)[1]);
    for (const [kind, p] of Object.entries(P)) {
      if (!p || !p.arr) continue;
      let [lo, hi] = p.autoLimits(S.disp.clip);
      if (shared !== null && (kind === "dirty" || kind === "restored")) hi = shared;
      // model * beam: starts at 0, unless the model itself goes negative (beyond FFT round-off)
      if (kind === "restored") lo = lo < -1e-6 * Math.abs(hi) ? lo : 0;
      p.render(L, S.disp.stretch, lo, hi);
    }
    if (region) drawZoom();
  }

  // ---- spectrum panel (in place of the residual): dirty vs recovered over the field shown; in channel
  // mode a dashed line at the channel shown, and pressing or dragging on the plot picks the channel
  const specBox = h("div", { class: "spec" });
  const specPanel = h("div", { class: "panel spec-panel" }, h("div", { class: "ptitle" }, h("b", {}, "spectrum, field shown")), specBox);
  let spec = null;
  async function loadSpectrum() {
    const [x0, x1, y0, y1] = P.dirty.view();
    spec = {};
    try { spec = await api(`/api/datasets/${m.id}/region`, { method: "POST", body: { res: S.resId, x0, x1, y0, y1 } }); }
    catch (e) { specBox.textContent = e.message; return; }
    drawSpectrum();
  }
  function drawSpectrum() {
    if (!spec || !spec.axis) return;
    const x = spec.axis, isCh = plane.value === "chan";
    const series = [
      { name: "dirty", color: cssVar("--s-dirty"), x, y: spec.dirty, width: 1.6 },
      spec.model ? { name: "recovered (model)", color: cssVar("--s-model"), x, y: spec.model, width: 1.8 } : null,
      spec.truth ? { name: "true sky", color: cssVar("--s-truth"), x, y: spec.truth, dash: "1 3", width: 2 } : null].filter(Boolean);
    const svg = linePlot(series, { xlabel: spec.axis_label, ylabel: `flux density (${spec.unit})`, width: 270, height: 290,
      vline: isCh ? x[+slider.value] : null, left: 50 });
    if (isCh) scrubSpec(svg);
    specBox.replaceChildren(svg, legend(series));
  }
  function scrubSpec(svg) {
    svg.classList.add("scrub");
    svg.addEventListener("pointerdown", (e) => {
      svg.setPointerCapture(e.pointerId); e.preventDefault();
      const go = (ev) => {
        const v = svg.fromX(new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse()).x);
        let k = 0;
        spec.axis.forEach((a, i) => { if (Math.abs(a - v) < Math.abs(spec.axis[k] - v)) k = i; });
        if (k === +slider.value) return;
        slider.value = k; paintSlider(slider); label(); load();
        // move the line in place: redrawing the plot would end the drag
        const ln = $(".vline", svg), xx = svg.X(spec.axis[k]);
        if (ln) { ln.setAttribute("x1", xx); ln.setAttribute("x2", xx); }
      };
      go(e);
      svg.onpointermove = go;
      svg.onpointerup = svg.onpointercancel = () => { svg.onpointermove = svg.onpointerup = svg.onpointercancel = null; };
    });
  }
  plane.onchange = slider.oninput = () => { label(); load(); drawSpectrum(); };

  // ---- region: zoomed crops + spectra ----
  let zoomPanels = [];
  async function setRegion(b) {
    region = b;
    for (const p of Object.values(P)) if (p) p.setBox(b);
    zoomBox.innerHTML = "";
    const head = h("div", { class: "row" }, h("h3", {}, "Region"),
      h("span", { class: "mono note" }, `x ${b.x0}–${b.x1}, y ${b.y0}–${b.y1}` + (cell ? `  (${((b.x1 - b.x0 + 1) * cell).toFixed(2)}″ × ${((b.y1 - b.y0 + 1) * cell).toFixed(2)}″)` : "")),
      h("div", { style: "flex:1" }), h("button", { onclick: () => { region = null; for (const p of Object.values(P)) if (p) p.setBox(null); zoomBox.innerHTML = ""; zoomBox.append(h("div", { class: "note" }, "Drag a box on any image to zoom in and get integrated spectra of that region.")); } }, "Clear"));
    zoomPanels = Object.entries(P).filter(([, p]) => p).map(([kind, p]) => {
      const z = new ImagePanel({ title: p.title, cell, cbar: "bottom" }); z.kind = kind; z.setBeam(p.beam); return z; });
    const spec = h("div");
    zoomBox.append(head, h("div", { class: "panels zoom" }, zoomPanels.map((z) => z.el)), spec);
    drawZoom();
    try {
      const s = await api(`/api/datasets/${m.id}/region`, { method: "POST", body: { res: S.resId, ...b } });
      const col = { dirty: "--s-dirty", model: "--s-model", restored: "--s-restored", residual: "--s-residual", truth: "--s-truth" };
      const series = ["dirty", "restored", "model", "residual", "truth"].filter((k) => s[k]).map((k) => ({
        name: k === "truth" ? "true sky" : k, color: cssVar(col[k]), x: s.axis, y: s[k], dash: k === "residual" ? "4 3" : k === "truth" ? "1 3" : "", width: k === "truth" ? 2.2 : 1.6 }));
      spec.append(h("h3", { style: "margin-top:6px" }, "Integrated spectrum"), legend(series),
        linePlot(series, { xlabel: s.axis_label, ylabel: `flux density (${s.unit})`, height: 300, width: 1100 }),
        s.beam_area_px ? h("div", { class: "note" }, `Jy/beam maps divided by the beam area (${s.beam_area_px.toFixed(1)} px) to give Jy.`) : null);
    } catch (e) { toast(e.message, true); }
  }
  async function drawZoom() {
    const L = await lut(S.disp.cmap);
    for (const z of zoomPanels) {
      const src = P[z.kind];
      if (!src || !src.arr) continue;
      z.set(src.arr); z.crop = [region.x0, region.x1, region.y0, region.y1];
      z.setUnit(src.unit);
      const [lo, hi] = z.autoLimits(S.disp.clip);
      z.render(L, S.disp.stretch, lo, hi);
    }
  }

  // ---- evolution + diagnostics ----
  const gifUrl = `/api/datasets/${m.id}/results/${S.resId}/evolution.gif?v=5`;   // v: the GIF style (render.GIF_VERSION), so browsers drop old ones
  const frames = r.frames || [];
  // frames are surface brightness; older runs saved them per pixel: converted here (as the server does for their GIF)
  const oldFrames = !summ.frame_unit, frameScale = oldFrames && cell ? perPix : 1;
  const frameUnit = summ.frame_unit || (cell ? `Jy${hasV ? " km/s" : " ch"} arcsec⁻²` : `Jy/pixel${hasV ? " km/s" : " ch"}`);
  const scrub = new ImagePanel({ title: "snapshot", unit: frameUnit, cell, cbar: "right" });
  const fs = h("input", { type: "range", min: 0, max: Math.max(frames.length - 1, 0), value: Math.max(frames.length - 1, 0) });
  let finalLim = null;
  const loadFrame = latestOnly(async () => {
    const i = +fs.value, a = scaled(await get(`/api/datasets/${m.id}/results/${S.resId}/frame/${i}`), frameScale);
    scrub.set(a); scrub.setTitle(`iteration ${frames[i]}`);
    if (!finalLim) finalLim = [0, scrub.autoLimits(100)[1] || 1];       // clip 100% of the final frame, fixed for all
    scrub.render(await lut(S.disp.cmap), S.disp.stretch, ...finalLim);
  });
  fs.oninput = loadFrame;
  const hist = r.history || [];
  const hx = hist.map((d) => d.iter);
  const rw = reweightMarks(r.params);
  const diag = h("div", {}, rw.length ? markLegend() : null,
    linePlot([{ name: "residual rms", color: cssVar("--s-dirty"), x: hx, y: hist.map((d) => d.residual_rms) }], { xlabel: "iteration", ylabel: "residual rms", height: 170, logy: true, marks: unlabelled(rw) }),
    linePlot([{ name: "flux", color: cssVar("--s-model"), x: hx, y: hist.map((d) => d.flux) }], { xlabel: "iteration", ylabel: "model flux (Jy)", height: 170, marks: unlabelled(rw) }),
    hist.length && hist[0].peak_ratio !== undefined ? linePlot([{ name: "peak", color: cssVar("--s-restored"), x: hx, y: hist.map((d) => d.peak_ratio) }], { xlabel: "iteration", ylabel: "N(model)/dirty peak", height: 150, marks: unlabelled(rw) }) : null);
  const params = h("table", { class: "kv" }, Object.entries(r.params).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", { class: "mono" }, JSON.stringify(v)))),
    r.summary && r.summary.restored_beam ? h("tr", {}, h("td", {}, "restoring beam"), h("td", {}, r.summary.restored_beam.map((v) => fmt(v)).join(", "))) : null,
    h("tr", {}, h("td", {}, "files"), h("td", { class: "mono" }, `${S.info.workdir}/datasets/${m.id}/results/${S.resId}/`)));

  body.append(
    h("div", { class: "card", style: "margin-top:14px" },
      h("div", { class: "controls" }, h("label", {}, "show", plane), slider, ...displayControls(draw),
        h("label", { title: "dirty and restored share the top of the colour scale (both in surface brightness); each keeps its own bottom" }, link, "common scale")),
      h("div", { class: "panels" }, Object.values(P).filter(Boolean).map((p) => p.el), specPanel)),
    zoomBox,
    h("div", { class: "grid2", style: "margin-top:14px" },
      h("div", { class: "card" }, h("h3", {}, "Evolution"),
        h("div", { class: "gifbox" }, h("img", { src: gifUrl, alt: "moment-0 map during the deconvolution" })),
        h("div", { class: "row" }, h("a", { href: gifUrl, download: "" }, "Download GIF")),
        h("h3", { style: "margin-top:8px" }, "Scrub through snapshots"), h("div", { style: "max-width:420px" }, scrub.el), fs),
      h("div", { class: "card" }, h("h3", {}, "Convergence"), diag, h("h3", {}, "Parameters"), params)));
  link.onchange = draw;
  await load();
  if (frames.length) loadFrame();
}

// ------------------------------------------------------------------ boot --
// Same switch as Albedo: sun/moon icon plus the theme it switches to, cross-faded with View Transitions.
const SUN = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6"/></svg>';
const MOON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/></svg>';
const GEAR = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.2"/><path d="M12 2.8v2.6M12 18.6v2.6M2.8 12h2.6M18.6 12h2.6M5.5 5.5l1.8 1.8M16.7 16.7l1.8 1.8M5.5 18.5l1.8-1.8M16.7 7.3l1.8-1.8"/><circle cx="12" cy="12" r="6.6"/></svg>';

// Preferences, kept in this browser: display font, the theme to start in, and the background.
// The theme button changes the theme for this tab only; the default decides where a new tab starts.
const PREFS_KEY = "deconvgui-prefs", PREF_DEFAULTS = { font: "abaddon", theme: "system", bg: "art" };
function loadPrefs() {
  let p = {};
  try {
    p = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") || {};
    const old = localStorage.getItem("deconvgui-theme");   // the theme button used to persist the theme
    if (!p.theme && (old === "dark" || old === "light")) p.theme = old;
  } catch { }
  return { ...PREF_DEFAULTS, ...p };
}
function savePrefs() { try { localStorage.setItem(PREFS_KEY, JSON.stringify(S.prefs)); localStorage.removeItem("deconvgui-theme"); } catch { } }
function applyPrefs() {   // font and background; the theme goes through initTheme's setter
  const root = document.documentElement;
  root.dataset.font = S.prefs.font; root.dataset.bg = S.prefs.bg;
}

function initTheme() {
  const root = document.documentElement, mq = matchMedia("(prefers-color-scheme: light)"), btn = $("#theme-btn");
  const theme = () => root.dataset.theme || (mq.matches ? "light" : "dark");
  const paintBtn = () => { const dark = theme() === "dark"; btn.innerHTML = (dark ? SUN : MOON) + `<span>${dark ? "Light" : "Dark"}</span>`; };
  // start in this tab's own choice if it made one, else in the default (system: no attribute, follow the OS)
  let t = null;
  try { t = sessionStorage.getItem("deconvgui-theme"); } catch { }
  if (t !== "dark" && t !== "light") t = S.prefs.theme === "system" ? null : S.prefs.theme;
  if (t) root.dataset.theme = t; else delete root.dataset.theme;
  // canvases and plots read the colour tokens when drawn, so redraw the open view
  const redraw = () => { if (S.dsId) selectDataset(S.dsId); };
  const setTheme = (t) => {
    if (t) root.dataset.theme = t; else delete root.dataset.theme;
    try { if (t) sessionStorage.setItem("deconvgui-theme", t); else sessionStorage.removeItem("deconvgui-theme"); } catch { }
    paintBtn();
  };
  let fadeTimer;
  // switch with a cross-fade (View Transitions), or a colour fade where unsupported
  S.switchTheme = (t) => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { setTheme(t); return redraw(); }
    if (document.startViewTransition) { document.startViewTransition(() => setTheme(t)).finished.then(redraw); return; }
    root.classList.add("theme-fade"); setTheme(t); redraw();
    clearTimeout(fadeTimer); fadeTimer = setTimeout(() => root.classList.remove("theme-fade"), 450);
  };
  btn.onclick = () => S.switchTheme(theme() === "dark" ? "light" : "dark");
  mq.addEventListener?.("change", paintBtn);
  paintBtn();
  $("#prefs-btn").innerHTML = GEAR;
  $("#prefs-btn").onclick = openPrefs;
}

// Toggleable pills: one is pressed (accent); pick(value) on click.
function pills(options, current, pick) {
  const box = h("div", { class: "pills", role: "group" });
  for (const [value, label, style] of options) {
    const b = h("button", { type: "button", class: "pill", "aria-pressed": value === current ? "true" : "false", style }, label);
    b.onclick = () => { for (const x of box.children) x.setAttribute("aria-pressed", x === b ? "true" : "false"); pick(value); };
    box.append(b);
  }
  return box;
}
function openPrefs() {
  const set = (k, v) => { S.prefs[k] = v; savePrefs(); applyPrefs(); };
  const section = (title, ...kids) => h("div", { class: "pref" }, h("h3", {}, title), ...kids);
  const close = modal("Preferences",
    section("Display font", pills([
      ["abaddon", "Abaddon", "font-family:Abaddon"],
      ["calibri", "Calibri", "font-family:Calibri,Carlito,sans-serif"],
      ["spectral", "Spectral", "font-family:Spectral,Georgia,serif"]], S.prefs.font,
      (v) => { set("font", v); if (S.dsId) selectDataset(S.dsId); })),
    section("Default theme", pills([["system", "System"], ["dark", "Dark"], ["light", "Light"]], S.prefs.theme,
      (v) => { set("theme", v); S.switchTheme(v === "system" ? null : v); }),
      h("div", { class: "hint" }, "Where the app starts; the sun/moon button switches only for this tab.")),
    section("Background", pills([["art", "Animated art"], ["solid", "Solid colour"]], S.prefs.bg, (v) => set("bg", v))),
    h("div", { class: "row end" }, h("button", { class: "primary", onclick: () => close() }, "Done")));
}

// ===================================================== Analysis & comparison(s)
// A 2D1D result (picked as in Results) against the method in use: multiscale CLEAN of the same
// visibilities on the same grid. The comparison: MS-CLEAN model * clean beam, the 2D1D model (raw), and
// the 2D1D model * its target (restoring) beam, in one surface-brightness unit, with their spectra.
async function analysisTab(body, m) {
  const main = h("div");
  body.append(main, hiresCard(m));
  return analysisMain(main, m);
}

// Comparison with a known high-resolution observation: does this galaxy have a better-resolved (extended)
// configuration? If yes, load it (as a measurement set, imaged on this dataset's grid; saved data; or cubes),
// and it is linked to this dataset as its reference.
function hiresCard(m) {
  const card = h("div", { class: "card", style: "margin-top:14px" }, h("h3", {}, "Comparison with known high-resolution observation"));
  const body = h("div", { class: "pref" });
  const ref = m.hires_ref && S.datasets.find((d) => d.id === m.hires_ref);
  const set = async (v) => {
    try { await api(`/api/datasets/${m.id}/meta`, { method: "POST", body: { hires: v } }); m.hires = v; draw(); }
    catch (e) { toast(e.message, true); }
  };
  const go = (fn) => () => {
    const im = m.imaging || {};
    S.linkTo = { ds: m.id, name: m.name, grid: im.phasecenter ? { field: im.field, phasecenter: im.phasecenter, restfreq_ghz: m.restfreq_ghz ?? im.restfreq_ghz,
      start_kms: m.start_kms, width_kms: m.width_kms, nchan: m.nchan, imsize: m.imsize, cell_arcsec: m.cell_arcsec } : null };
    fn();
  };
  function draw() {
    body.replaceChildren(
      h("div", { class: "note" }, "Does this galaxy have an extended configuration, better resolved than this data?"),
      pills([["yes", "Yes"], ["no", "No"]], m.hires || null, set));
    if (m.hires !== "yes") return;
    if (ref) {
      body.append(h("div", { class: "row" }, h("span", {}, "Reference: ", h("b", {}, ref.name)),
        h("button", { onclick: () => selectDataset(ref.id, "overview") }, "Open"),
        h("button", { class: "ghost danger", onclick: async () => { await api(`/api/datasets/${m.id}/meta`, { method: "POST", body: { hires_ref: null } }); selectDataset(m.id, "analysis"); } }, "Unlink")));
      return;
    }
    body.append(h("div", { class: "note" }, "Load it; it is linked here as this dataset's high-resolution reference. Load MS images it on this dataset's grid."),
      h("div", { class: "row" },
        h("button", { class: "primary", onclick: go(() => viewMS()) }, "Load as MS"),
        h("button", { class: "primary", onclick: go(() => loadSavedDialog()) }, "Load saved data"),
        h("button", { class: "primary", onclick: go(() => viewImport()) }, "Import cubes")));
  }
  draw();
  card.append(body);
  return card;
}

async function analysisMain(body, m) {
  const res = (m.results || []).slice().reverse();
  const done = res.filter((r) => r.status === "done");
  if (res.length) {
    if (!S.resId || !res.find((r) => r.id === S.resId)) S.resId = (done[0] || res[0]).id;
    body.append(h("div", { class: "card scroll-x" }, h("h3", {}, "2D1D result"), resultTable(m, res, "analysis")));
  } else body.append(h("div", { class: "card" }, h("h3", {}, "2D1D result"), h("div", { class: "note" }, "No results yet: run a deconvolution first.")));

  // ---- MS-CLEAN: parameters, run, runs
  const im = m.imaging || {}, fromMS = !!im.ms;
  const card = h("div", { class: "card", style: "margin-top:14px" }, h("h3", {}, "Comparison with presently used method: MS-CLEAN"));
  body.append(card);
  if (!fromMS) { card.append(h("div", { class: "note" }, "MS-CLEAN images the visibilities themselves: it needs a dataset made with Load MS.")); return; }
  const bpx = m.beam && m.cell_arcsec ? m.beam[0] / m.cell_arcsec : null;
  const field = (name, label, value, hint, attrs = {}) => h("label", {}, label, h("input", { name, value: value ?? "", ...attrs }), hint ? h("span", { class: "hint" }, hint) : null);
  const maskSel = h("select", { name: "mask" }, h("option", { value: "auto-multithresh" }, "auto-multithresh"), h("option", { value: "none" }, "none (components anywhere)"));
  const AUTO = [["sidelobethreshold", 3.0], ["noisethreshold", 5.0], ["lownoisethreshold", 1.5], ["negativethreshold", 7.0], ["minbeamfrac", 0.3], ["growiterations", 75]];
  const autoBox = h("div", { class: "form", style: "grid-column:1/-1" }, AUTO.map(([k, v]) => field(k, k, v, null, { type: "number", step: "any" })));
  const form = h("form", { class: "form", onsubmit: (e) => e.preventDefault() },
    field("scales", "scales (px)", "0, 10, 30, 60", `point + extended scales${bpx ? `; the clean beam is ≈ ${fmt(bpx, 3)} px across` : ""}`),
    field("nsigma", "nsigma", 3, "stop at nsigma × the residual rms", { type: "number", step: "any" }),
    field("niter", "niter (max)", 1000000, "upper limit; nsigma normally stops it first", { type: "number", step: 1 }),
    field("smallscalebias", "smallscalebias", "", "empty = CASA default (0); > 0 favours small scales", { type: "number", step: "any" }),
    h("label", {}, "mask", maskSel, h("span", { class: "hint" }, "ALMA pipeline auto-masking, as scripts/msclean_cube.py")),
    autoBox);
  maskSel.onchange = () => { autoBox.hidden = maskSel.value !== "auto-multithresh"; };
  const run = h("button", { class: "primary" }, "Run MS-CLEAN"), mon = h("div");
  run.onclick = async () => {
    const f = Object.fromEntries(new FormData(form).entries()), num = (v) => v === "" || v === undefined ? null : +v;
    const scales = String(f.scales).split(/[,\s]+/).filter(Boolean).map(Number);
    if (!scales.length || scales.some((x) => !Number.isInteger(x) || x < 0)) return toast("scales: whole numbers of pixels, e.g. 0, 10, 30, 60", true);
    const params = { scales, nsigma: num(f.nsigma), niter: num(f.niter), smallscalebias: num(f.smallscalebias), mask: f.mask,
      ...(f.mask === "auto-multithresh" ? Object.fromEntries(AUTO.map(([k]) => [k, num(f[k])])) : {}) };
    run.disabled = true;
    try {
      const r = await api(`/api/datasets/${m.id}/msclean`, { method: "POST", body: { params } });
      S.cleanId = r.run;
      mon.replaceChildren(jobMonitor(r.job, { onDone: () => selectDataset(m.id, "analysis") }));
    } catch (e) { toast(e.message, true); run.disabled = false; }
  };
  card.append(h("div", { class: "note" }, `Imaged exactly like this dataset's dirty cube (same measurement sets, field, spws, phase centre, channels, ${im.imsize} px × ${im.cell_arcsec}″, ${im.weighting || "natural"} weighting) and restored with one common beam. The comparison uses the CLEAN model convolved with that beam, without the residual, as for the 2D1D restored cube.`),
    form, h("div", { class: "row" }, run), mon);

  const runs = (await api(`/api/datasets/${m.id}/msclean`).catch(() => [])).slice().reverse();
  const cdone = runs.filter((r) => r.status === "done");
  if (!S.cleanId || !runs.find((r) => r.id === S.cleanId)) S.cleanId = (cdone[0] || runs[0] || {}).id || null;
  if (runs.length) {
    const liveRun = (r) => r.status === "running" || r.status === "queued";
    card.append(h("div", { class: "scroll-x" }, h("table", { class: "list" },
      h("tr", {}, ["MS-CLEAN run", "scales (px)", "nsigma", "mask", "clean beam", "iterations", "flux (Jy)", "status", "started", ""].map((t) => h("th", {}, t))),
      runs.map((r) => h("tr", { class: "click" + (r.id === S.cleanId ? " sel" : ""), onclick: () => { S.cleanId = r.id; selectDataset(m.id, "analysis"); } },
        h("td", { class: "mono" }, r.id), h("td", { class: "mono" }, (r.params.scales || []).join(", ")), h("td", {}, r.params.nsigma ?? "–"), h("td", {}, r.params.mask || "–"),
        h("td", {}, r.summary && r.summary.beam ? `${fmt(r.summary.beam[0], 3)}″ × ${fmt(r.summary.beam[1], 3)}″` : "–"),
        h("td", {}, r.summary ? r.summary.iterations : "–"), h("td", {}, r.summary ? fmt(r.summary.flux, 4) : "–"),
        h("td", {}, h("span", { class: "chip " + (r.status === "done" ? "ok" : liveRun(r) ? "run" : "err"), title: r.error || "" }, r.status)),
        h("td", {}, ago(r.created)),
        h("td", {}, h("button", { class: "ghost danger", title: "delete this MS-CLEAN run", onclick: async (e) => { e.stopPropagation(); await api(`/api/datasets/${m.id}/msclean/${r.id}`, { method: "DELETE" }); S.cleanId = null; selectDataset(m.id, "analysis"); } }, "✕")))))));
    // a run still going (e.g. started before switching tab): its progress
    const going = runs.find(liveRun);
    if (going && going.job && !mon.childElementCount) mon.append(jobMonitor({ id: going.job, label: `MS-CLEAN ${going.id} (running)` }, { onDone: () => selectDataset(m.id, "analysis") }));
  }

  // ---- the comparison
  const cmp = h("div", { class: "card", style: "margin-top:14px" }, h("h3", {}, "MS-CLEAN vs 2D1D"));
  body.append(cmp);
  const crun = runs.find((r) => r.id === S.cleanId && r.status === "done");
  const r = S.resId && done.find((x) => x.id === S.resId) ? await api(`/api/datasets/${m.id}/results/${S.resId}`) : null;
  if (!crun || !r) {
    cmp.append(h("div", { class: "note" }, !r ? "Pick a finished 2D1D result above." : "Run MS-CLEAN (or pick a finished run) to compare."));
    return;
  }
  const cell = m.cell_arcsec, hasV = !!m.width_kms, cb = crun.summary.beam, rb = (r.summary || {}).restored_beam || null;
  const hasRestored = r.files.includes("restored.npy");
  const beamArea = (b) => Math.PI * b[0] * b[1] / (4 * Math.LN2), perPix = cell ? 1 / (cell * cell) : 1;
  const K = [
    { key: "clean", title: "MS-CLEAN ∗ clean beam", q: `kind=clean&mrun=${crun.id}`, scale: cell && cb ? 1 / beamArea(cb) : null, beam: cb },
    { key: "model", title: "2D1D deconvolved (raw)", q: `kind=model&res=${r.id}`, scale: perPix, beam: null },
    // the target beam: the result's restoring beam, else the clean beam; editable below (model * that beam, on the server)
    (rb || m.beam) && cell ? { key: "restored", title: "2D1D deconvolved ∗ target beam", beam: (rb || m.beam).slice(0, 3) } : null,
  ].filter(Boolean);
  const T = K.find((k) => k.key === "restored");
  if (T) {
    Object.defineProperty(T, "q", { get: () => `kind=model_conv&res=${r.id}&beam=${T.beam.map((v) => +(+v).toPrecision(6)).join(",")}` });
    Object.defineProperty(T, "scale", { get: () => 1 / beamArea(T.beam) });
  }
  for (const k of K) { k.panel = new ImagePanel({ title: k.title, cell, cbar: "bottom" }); k.panel.setBeam(k.beam); }
  if (T) {   // three boxes in place of the beam's label: bmaj, bmin (arcsec), PA (deg); changing them redraws map, ellipse, spectrum
    const box = (i, label, step) => h("input", { type: "number", step, min: i < 2 ? 0 : null, value: +(+T.beam[i]).toPrecision(4), title: label, "aria-label": label });
    const ins = [box(0, "target beam major axis (arcsec)", "any"), box(1, "target beam minor axis (arcsec)", "any"), box(2, "target beam position angle (deg)", "any")];
    let timer;
    const apply = () => {
      const b = ins.map((x) => +x.value);
      if (!(b[0] > 0 && b[1] > 0) || b.some((v) => !Number.isFinite(v))) return;
      if (b[1] > b[0]) { [b[0], b[1]] = [b[1], b[0]]; b[2] += 90; }      // major >= minor
      T.beam = b; T.panel.setBeam(b);
      load(); loadSpectrum();
    };
    for (const x of ins) {
      x.oninput = () => { clearTimeout(timer); timer = setTimeout(apply, 450); };
      x.onkeydown = (e) => { if (e.key === "Enter") { clearTimeout(timer); apply(); } };
    }
    T.panel.setBeamControl(h("div", {}, ins[0], h("span", {}, "″ ×"), ins[1], h("span", {}, "″,"), ins[2], h("span", {}, "°")));
  }
  const unit = (k, isCh) => { const v = isCh ? "" : hasV ? " km/s" : " ch"; return k.scale === null ? `Jy/beam${v}` : `Jy${v} ${cell ? "arcsec⁻²" : "px⁻²"}`; };
  const scaled = (arr, f) => (!arr || f === null || f === 1) ? arr : { ...arr, data: Float32Array.from(arr.data, (v) => v * f) };
  const plane = h("select", {}, h("option", { value: "mom0" }, "moment 0"), h("option", { value: "chan" }, "channel"));
  const nchan = m.nchan || 1;
  const slider = h("input", { type: "range", min: 0, max: nchan - 1, value: Math.floor(nchan / 2), disabled: true });
  const link = h("input", { type: "checkbox", checked: true });
  // the comparison inside MS-CLEAN's own clean mask: spectra summed over its pixels (channel by channel), its outline on the target-beam map
  // spectra over the field shown, or only inside a 3D mask, channel by channel: MS-CLEAN's own clean mask, or an astrodendro
  // mask made on MS-CLEAN's image (model * beam + residual). Channel view: the mask outlined on the target-beam map;
  // moment 0: that map is the moment 0 of only what lies inside the mask (a rotating disc's mask moves from channel to channel)
  const get = arrayCache(m.imsize);
  let dinfo = await api(`/api/datasets/${m.id}/msclean/${crun.id}/dendro`).catch(() => null);
  const maskKind = () => specMask.value === "dendro" ? "clean_dmask" : "clean_mask";
  const maskVer = () => specMask.value === "dendro" && dinfo ? `&v=${dinfo.created}` : "";
  const maskUrl = (isCh, c, kind = maskKind()) => `/api/datasets/${m.id}/array?kind=${kind}&mrun=${crun.id}&${isCh ? `plane=chan&chan=${c}` : "plane=mom0"}${kind === "clean_dmask" && dinfo ? `&v=${dinfo.created}` : ""}`;
  const hasClean = crun.params.mask !== "none" && !!(await get(maskUrl(false, 0, "clean_mask")).catch(() => null));
  const specMask = h("select", { title: "spectra summed over the field shown, or only over the pixels inside a 3D mask (each channel's own)" },
    h("option", { value: "" }, "field shown"),
    h("option", { value: "clean", disabled: !hasClean }, hasClean ? "CLEAN auto-mask" : crun.params.mask === "none" ? "CLEAN mask (run had none)" : "CLEAN mask (not saved: re-run)"),
    h("option", { value: "dendro" }, "astrodendro mask"));
  specMask.value = hasClean ? "clean" : "";
  const active = () => specMask.value === "clean" || (specMask.value === "dendro" && !!dinfo);
  const maskName = () => specMask.value === "dendro" ? "astrodendro mask" : "CLEAN mask";
  // the astrodendro mask's thresholds (in units of the image noise) and the button that (re)makes it
  const DP = [["min_value", "min value (σ)", 3, "voxels below this are never in the mask"], ["min_delta", "min Δ (σ)", 1, "a peak must rise this far above where it merges to stand as its own structure"],
    ["peak", "peak (σ)", 5, "keep only structures whose peak reaches this"], ["min_beams", "min size (beams)", 2, "structures smaller than this many clean beams of voxels are dropped"]];
  const dIns = Object.fromEntries(DP.map(([k, , v, t]) => [k, h("input", { type: "number", step: "any", min: 0, value: dinfo ? dinfo.params[k] : v, title: t, style: "width:5.5em" })]));
  const dStat = h("span", { class: "hint" });
  const dBtn = h("button", {}, "Make mask");
  const dRow = h("div", { class: "controls" }, h("span", { class: "hint" }, "astrodendro on MS-CLEAN's image (model ∗ beam + residual):"),
    DP.map(([k, lbl]) => h("label", {}, lbl, dIns[k])), dBtn, dStat);
  const showDinfo = () => {
    dRow.hidden = specMask.value !== "dendro";
    dStat.textContent = dinfo ? `σ = ${fmt(dinfo.sigma * 1e3, 3)} mJy/beam (line-free channels) · ${dinfo.structures} structure${dinfo.structures === 1 ? "" : "s"} · ${fmt(100 * dinfo.fraction, 3)} % of the cube · ${fmt(dinfo.seconds, 2)} s`
      : "no mask yet: set the thresholds and make one";
  };
  dBtn.onclick = async () => {
    dBtn.disabled = true; dStat.textContent = "computing the dendrogram…";
    try {
      dinfo = await api(`/api/datasets/${m.id}/msclean/${crun.id}/dendro`, { method: "POST", body: { params: Object.fromEntries(DP.map(([k]) => [k, +dIns[k].value])) }, timeout: 1800000 });
      showDinfo(); load(); loadSpectrum();
    } catch (e) { dStat.textContent = e.message; }
    finally { dBtn.disabled = false; }
  };
  const specTitle = h("b", {}, "spectra, field shown");
  // moment 0 with the mask on: the target-beam map is the moment 0 of only what lies inside the mask, channel by channel
  // (a rotating disc's mask moves from channel to channel: no single outline stands for it)
  const urls = (isCh, c) => K.map((k) => `/api/datasets/${m.id}/array?${k.q}&${isCh ? `plane=chan&chan=${c}` : "plane=mom0"}` +
    (k === T && !isCh && active() ? `&inmask=1&mrun=${crun.id}&mask=${specMask.value}${maskVer()}` : ""));
  const label = () => { const isCh = plane.value === "chan"; slider.disabled = !isCh; K[0].panel.setTag(isCh ? chanLabel(m, +slider.value) : ""); };
  const load = latestOnly(async () => {
    const isCh = plane.value === "chan", c = +slider.value;
    const withMask = T && active() && isCh;
    const arrs = await Promise.all([...urls(isCh, c), ...(withMask ? [maskUrl(isCh, c)] : [])].map((u) => get(u).catch(() => null)));
    K.forEach((k, i) => { k.panel.set(scaled(arrs[i], k.scale)); k.panel.setUnit(unit(k, isCh));
      k.panel.setTitle(k.title + (isCh ? "" : ", moment 0")); });
    if (T) T.panel.setTag(!isCh && active() ? `inside ${maskName()}` : "");
    if (T) T.panel.setContour(withMask ? arrs[K.length] : null);   // channel: that channel's mask
    await draw();
    if (isCh) for (const j of [c + 1, c - 1]) if (j >= 0 && j < nchan) [...urls(true, j), ...(withMask ? [maskUrl(true, j)] : [])].forEach((u) => get(u).catch(() => {}));
  });
  async function draw() {
    const L = await lut(S.disp.cmap);
    // common scale: all three share the top (one unit); each keeps its own floor, which is 0 for the
    // beam-convolved maps unless their model goes negative (CLEAN components can)
    const top = link.checked ? Math.max(...K.filter((k) => k.panel.arr).map((k) => k.panel.autoLimits(S.disp.clip)[1])) : null;
    for (const k of K) {
      if (!k.panel.arr) continue;
      let [lo, hi] = k.panel.autoLimits(S.disp.clip);
      if (top !== null) hi = top;
      if (k.key !== "model") lo = lo < -1e-6 * Math.abs(hi) ? lo : 0;
      k.panel.render(L, S.disp.stretch, lo, hi);
    }
  }

  // ---- spectra over the field, each in Jy (beam-convolved maps divided by their beam's area)
  const specBox = h("div", { class: "spec" });
  let spec = null;
  async function loadSpectrum() {
    const [x0, x1, y0, y1] = K[0].panel.view();
    const key = () => (T ? T.beam.join(",") : "") + "|" + (active() ? specMask.value + maskVer() : "");
    const masked = active() ? specMask.value : "", name = maskName(), want = key();
    specBox.style.opacity = 0.45;   // until the new spectra arrive (the masked/full-field sums take a few seconds)
    try {
      const sp = await api(`/api/datasets/${m.id}/region`, { method: "POST", body: { res: r.id, mrun: crun.id, x0, x1, y0, y1, ...(T ? { rbeam: T.beam } : {}), ...(masked ? { clean_mask: masked } : {}) } });
      if (key() !== want) return;   // the beam or mask changed meanwhile: a newer request is on its way
      spec = sp;
      specTitle.textContent = masked ? `spectra, inside ${name}` : "spectra, field shown";
    }
    catch (e) { specBox.textContent = e.message; return; }
    finally { if (key() === want) specBox.style.opacity = ""; }
    drawSpectrum();
  }
  function drawSpectrum() {
    if (!spec || !spec.axis) return;
    const x = spec.axis, isCh = plane.value === "chan", dv = Math.abs(m.width_kms || 1);
    const tot = (y) => ` (${fmt(y.reduce((a, b) => a + b, 0) * dv, 3)} Jy${hasV ? " km/s" : " ch"})`;
    const series = [
      spec.clean ? { name: "MS-CLEAN ∗ clean beam" + tot(spec.clean), color: cssVar("--s-dirty"), x, y: spec.clean, width: 1.8 } : null,
      spec.clean_image ? { name: "MS-CLEAN ∗ clean beam + residual" + tot(spec.clean_image), color: cssVar("--s-dirty"), x, y: spec.clean_image, dash: "2 3", width: 1.4 } : null,
      spec.model ? { name: "2D1D (raw)" + tot(spec.model), color: cssVar("--s-model"), x, y: spec.model, width: 1.8 } : null,
      spec.restored ? { name: "2D1D ∗ target beam" + tot(spec.restored), color: cssVar("--s-restored"), x, y: spec.restored, dash: "5 4", width: 1.8 } : null].filter(Boolean);
    const svg = linePlot(series, { xlabel: spec.axis_label, ylabel: `flux density (${spec.unit})`, width: 300, height: 300, vline: isCh ? x[+slider.value] : null, left: 50 });
    if (isCh) {   // press or drag on the spectrum to pick the channel
      svg.classList.add("scrub");
      svg.addEventListener("pointerdown", (e) => {
        svg.setPointerCapture(e.pointerId); e.preventDefault();
        const go = (ev) => {
          const v = svg.fromX(new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse()).x);
          let c = 0; x.forEach((a, i) => { if (Math.abs(a - v) < Math.abs(x[c] - v)) c = i; });
          if (c === +slider.value) return;
          slider.value = c; paintSlider(slider); label(); load();
          const ln = $(".vline", svg), xx = svg.X(x[c]);
          if (ln) { ln.setAttribute("x1", xx); ln.setAttribute("x2", xx); }
        };
        go(e);
        svg.onpointermove = go;
        svg.onpointerup = svg.onpointercancel = () => { svg.onpointermove = svg.onpointerup = svg.onpointercancel = null; };
      });
    }
    specBox.replaceChildren(svg, legend(series));
  }
  plane.onchange = slider.oninput = () => { label(); load(); drawSpectrum(); };
  link.onchange = draw;
  specMask.onchange = () => { showDinfo(); load(); loadSpectrum(); };
  showDinfo();
  cmp.append(h("div", { class: "note" }, `MS-CLEAN ${crun.id} (clean beam ${fmt(cb[0], 3)}″ × ${fmt(cb[1], 3)}″) against 2D1D ${r.id}` +
      ". The target beam (its boxes, on the third map) can be changed: the map, its ellipse and its spectrum follow. Spectra: summed over the field shown" +
      " or only over the pixels inside a 3D mask, each channel's own: MS-CLEAN's clean mask, or an astrodendro mask made on MS-CLEAN's image (channel view: outlined on the target-beam map; moment 0: that map is the moment 0 of only what lies inside it)" +
      "; beam-convolved maps divided by their beam's area."),
    h("div", { class: "controls" }, h("label", {}, "show", plane), slider, ...displayControls(draw),
      h("label", { title: "the three maps share the top of the colour scale (one unit); each keeps its own bottom" }, link, "common scale"),
      h("label", {}, "spectra over", specMask)),
    dRow,
    h("div", { class: "panels" }, K.map((k) => k.panel.el),
      h("div", { class: "panel spec-panel" }, h("div", { class: "ptitle" }, specTitle), specBox)));
  await load();
  loadSpectrum();
}

// Sliders: WebKit has no ::range-progress, so the filled part of the track is a gradient up to --b.
// (call paintSlider after setting a slider's value from code: that fires no input event)
const paintSlider = (r) => { const lo = +r.min || 0, hi = r.max === "" ? 100 : +r.max; r.style.setProperty("--b", `${hi > lo ? (r.value - lo) / (hi - lo) * 100 : 0}%`); };
function initSliders() {
  const paint = paintSlider;
  document.addEventListener("input", (e) => { if (e.target.type === "range") paint(e.target); });
  new MutationObserver(() => document.querySelectorAll("input[type=range]").forEach(paint))
    .observe(document.body, { childList: true, subtree: true });
}

async function boot() {
  S.prefs = loadPrefs(); applyPrefs();
  initTheme();
  initSliders();
  try { S.devices = await api("/api/devices"); } catch { S.devices = null; }
  await loadInfo();
  await loadDatasets();
  // the sidebar starts plain flows (only the Analysis tab's buttons add a high-resolution reference)
  $("#new-ms").onclick = () => { S.linkTo = null; viewMS(); };
  $("#load-saved").onclick = () => { S.linkTo = null; loadSavedDialog(); };
  $("#new-import").onclick = () => { S.linkTo = null; viewImport(); };
  $("#new-toy").onclick = async () => {
    $("#new-toy").disabled = true;
    try { const m = await api("/api/datasets/toy", { method: "POST", body: {} }); await loadDatasets(); selectDataset(m.id); }
    catch (e) { toast(e.message, true); } finally { $("#new-toy").disabled = false; }
  };
  setInterval(pollJobs, 2500); pollJobs();
  setInterval(() => loadInfo().catch(() => { /* server restarting */ }), 60000);
  if (!restoreView()) { const first = S.datasets[0]; if (first) selectDataset(first.id); }
}
boot().catch((e) => toast(e.message, true));
