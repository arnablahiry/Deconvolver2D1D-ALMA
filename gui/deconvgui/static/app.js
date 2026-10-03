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

async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.body && typeof o.body !== "string") {
    o.body = JSON.stringify(o.body);
    o.headers = { "content-type": "application/json" };
  }
  const r = await fetch(path, o);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch { msg = await r.text().catch(() => msg); }
    throw new Error(msg);
  }
  return r.json();
}

async function getArray(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const [ny, nx] = r.headers.get("X-Shape").split(",").map(Number);
  const it = r.headers.get("X-Iter");
  return { data: new Float32Array(await r.arrayBuffer()), ny, nx, iter: it === null ? null : +it };
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
  t.style.cssText = `position:fixed;bottom:18px;right:18px;max-width:520px;padding:10px 14px;border-radius:6px;z-index:9;
    background:var(--surface);border:1px solid ${isErr ? "var(--err)" : "var(--line)"};color:${isErr ? "var(--err)" : "var(--fg)"};white-space:pre-wrap;font-size:13px`;
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
  constructor({ title, unit = "", onRegion = null, cell = null }) {
    this.title = title; this.unit = unit; this.onRegion = onRegion; this.cell = cell;
    this.arr = null; this.crop = null; this.box = null; this.limits = null;
    this.canvas = h("canvas");
    this.overlay = h("canvas", { style: "pointer-events:none" });
    this.readout = h("div", { class: "readout" }, " ");
    this.unitEl = h("span", { class: "punit" }, unit);
    this.titleEl = h("b", {}, title);
    this.cv = h("div", { class: "cv" }, this.canvas, this.overlay);
    this.el = h("div", { class: "panel" }, h("div", { class: "ptitle" }, this.titleEl, this.unitEl), this.cv, this.readout);
    this._mouse();
    new ResizeObserver(() => this.drawOverlay()).observe(this.cv);
  }
  setUnit(u) { this.unit = u; this.unitEl.textContent = u; }
  setTitle(t) { this.title = t; this.titleEl.textContent = t; }
  set(arr) { this.arr = arr; this._auto = null; }
  view() {   // [x0, x1, y0, y1] inclusive, in data pixels
    if (!this.arr) return [0, 0, 0, 0];
    return this.crop || [0, this.arr.nx - 1, 0, this.arr.ny - 1];
  }
  autoLimits(clip) {
    if (!this.arr) return [0, 1];
    if (!this._auto || this._auto.clip !== clip) {
      let src = this.arr.data;
      if (this.crop) src = this._cropData();
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
  }
  setBox(box) { this.box = box; this.drawOverlay(); }
  _toScreen(x, y) {   // data pixel corner -> overlay px
    const [x0, x1, y0, y1] = this.view(), W = this.overlay.width, H = this.overlay.height;
    return [(x - x0) / (x1 - x0 + 1) * W, (y1 + 1 - y) / (y1 - y0 + 1) * H];
  }
  drawOverlay(tmp) {
    const r = this.cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    this.overlay.width = Math.max(1, Math.round(r.width * dpr)); this.overlay.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = this.overlay.getContext("2d");
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    const b = tmp || this.box;
    if (!b || !this.arr || this.crop) return;
    const [ax, ay] = this._toScreen(b.x0, b.y1 + 1), [bx, by] = this._toScreen(b.x1 + 1, b.y0);
    ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = "#7fd3ff"; ctx.setLineDash([5 * dpr, 3 * dpr]);
    ctx.strokeRect(ax, ay, bx - ax, by - ay);
  }
  _pix(e) {
    const r = this.canvas.getBoundingClientRect(), [x0, x1, y0, y1] = this.view();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const x = Math.min(x1, Math.max(x0, x0 + Math.floor(fx * (x1 - x0 + 1))));
    const y = Math.min(y1, Math.max(y0, y1 - Math.floor(fy * (y1 - y0 + 1))));
    return [x, y];
  }
  _mouse() {
    let start = null;
    this.canvas.addEventListener("mousemove", (e) => {
      if (!this.arr) return;
      const [x, y] = this._pix(e), v = this.arr.data[y * this.arr.nx + x];
      let off = "";
      if (this.cell) {
        const dx = -(x - this.arr.nx / 2) * this.cell, dy = (y - this.arr.ny / 2) * this.cell;
        off = `  Δα ${dx.toFixed(2)}″ Δδ ${dy.toFixed(2)}″`;
      }
      this.readout.textContent = `x ${x}  y ${y}  ${fmt(v, 4)}${off}`;
      if (start) {
        const b = { x0: Math.min(start[0], x), x1: Math.max(start[0], x), y0: Math.min(start[1], y), y1: Math.max(start[1], y) };
        this.drawOverlay(b);
      }
    });
    this.canvas.addEventListener("mouseleave", () => { this.readout.textContent = " "; });
    if (!this.onRegion) return;
    this.canvas.addEventListener("mousedown", (e) => { if (this.arr && !this.crop) { start = this._pix(e); e.preventDefault(); } });
    window.addEventListener("mouseup", (e) => {
      if (!start) return;
      const [x, y] = this._pix(e), s = start; start = null;
      const b = { x0: Math.min(s[0], x), x1: Math.max(s[0], x), y0: Math.min(s[1], y), y1: Math.max(s[1], y) };
      if (b.x1 - b.x0 < 2 || b.y1 - b.y0 < 2) { this.drawOverlay(); return; }
      this.onRegion(b);
    });
  }
}

// ------------------------------------------------------------ line plots --
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function linePlot(series, { xlabel = "", ylabel = "", width = 520, height = 220, logy = false } = {}) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.setAttribute("class", "plot");
  const m = { l: 58, r: 12, t: 10, b: 36 }, W = width - m.l - m.r, H = height - m.t - m.b;
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
  el("line", { x1: m.l, x2: m.l + W, y1: m.t + H, y2: m.t + H, class: "axis" });
  el("line", { x1: m.l, x2: m.l, y1: m.t, y2: m.t + H, class: "axis" });
  if (y0 < 0 && y1 > 0 && !logy) el("line", { x1: m.l, x2: m.l + W, y1: Y(0), y2: Y(0), class: "axis" });
  el("text", { x: m.l + W / 2, y: height - 4, "text-anchor": "middle" }).textContent = xlabel;
  const yl = el("text", { x: 12, y: m.t + H / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + H / 2})` });
  yl.textContent = ylabel;
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
const legend = (series) => h("div", { class: "legend" }, series.map((s) => h("span", {}, h("i", { style: `background:${s.color}` }), s.name)));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ------------------------------------------------------------------ state --
const S = {
  info: null, datasets: [], dsId: null, tab: "overview", resId: null,
  disp: { cmap: "inferno", stretch: "asinh", clip: 99.5 },
  timers: new Set(),
};
function clearTimers() { for (const t of S.timers) clearInterval(t); S.timers.clear(); }
function every(ms, fn) { const t = setInterval(fn, ms); S.timers.add(t); return t; }

// -------------------------------------------------------------- top bar --
async function loadInfo() {
  S.info = await api("/api/info");
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

async function pollJobs() {
  try {
    const jobs = await api("/api/jobs"), running = jobs.filter((j) => j.status === "running" || j.status === "queued"), el = $("#jobs-indicator");
    el.hidden = !running.length; el.innerHTML = "";
    if (running.length) {
      const j = running[0];
      el.append(h("span", { class: "dot" }), h("span", {}, running.length > 1 ? `${jobs.filter((j) => j.status === "running").length} running, ${jobs.filter((j) => j.status === "queued").length} queued` : `${j.label}: ${j.fraction != null ? Math.round(j.fraction * 100) + "%" : "…"}`));
      el.style.cursor = j.ds_id ? "pointer" : "";
      el.onclick = () => j.ds_id && selectDataset(j.ds_id, j.kind === "deconvolve" ? "deconvolve" : "overview");
    }
  } catch { /* server restarting */ }
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

function setMain(...nodes) { clearTimers(); const m = $("#main"); m.innerHTML = ""; m.append(...nodes); m.scrollTop = 0; }

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

// ------------------------------------------------------- job monitor --
function jobMonitor(job, { onDone, live = false, cell = null, dv = 1 } = {}) {
  const bar = h("div", { class: "bar" }, h("span"));
  const msg = h("div", { class: "progress-msg" }, job.message || "");
  const log = h("pre", { class: "log" });
  const cancel = h("button", { class: "danger", onclick: async () => { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); cancel.disabled = true; } }, "Cancel");
  const err = h("div", { class: "err-box" });
  const devEl = h("span", { class: "chip run" }, job.device ? `on ${job.device}` : job.status === "queued" ? "queued" : "");
  const card = h("div", { class: "card" }, h("div", { class: "row" }, h("h2", {}, job.label), devEl, h("div", { class: "spacer", style: "flex:1" }), cancel), bar, msg);
  let panel = null, plotBox = null, stats = [], seq = -1, busy = false;
  if (live) {
    panel = new ImagePanel({ title: "current iterate, moment 0", unit: "Jy/pixel km/s", cell });
    plotBox = h("div");
    card.append(h("div", { class: "grid2" }, h("div", { style: "max-width:460px" }, panel.el), plotBox));
  }
  card.append(h("details", { open: !live }, h("summary", { class: "note" }, "log"), log), err);
  const tick = async () => {
    if (busy) return; busy = true;
    try {
      const j = await api(`/api/jobs/${job.id}?since=${stats.length}`);
      stats.push(...j.stats);
      bar.classList.toggle("indet", j.fraction === null && j.status === "running");
      if (j.device) devEl.textContent = `on ${j.device}`;
      else if (j.status === "queued") devEl.textContent = `queued for ${(j.devices || []).join(" / ")}`;
      $("span", bar).style.width = j.fraction === null ? "" : `${Math.round(j.fraction * 100)}%`;
      msg.textContent = j.message || "";
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
      log.textContent = j.log.join("\n");
      if (atBottom) log.scrollTop = log.scrollHeight;
      if (live && j.frame_seq !== seq && j.frame_seq > 0) {
        seq = j.frame_seq;
        const a = await getArray(`/api/jobs/${job.id}/frame`);
        panel.set(a);
        panel.setTitle(`iteration ${a.iter}`);
        const [lo, hi] = panel.autoLimits(S.disp.clip);
        panel.render(await lut(S.disp.cmap), S.disp.stretch, Math.min(lo, 0), hi);
      }
      if (live && stats.length) {
        const x = stats.map((s) => s.iter);
        plotBox.innerHTML = "";
        const a = [{ name: "residual rms", color: cssVar("--s-dirty"), x, y: stats.map((s) => s.residual_rms) }];
        const b = [{ name: "model flux (Jy)", color: cssVar("--s-model"), x, y: stats.map((s) => s.flux) }];
        plotBox.append(linePlot(a, { xlabel: "iteration", ylabel: "residual rms", height: 170, logy: true }),
          linePlot(b, { xlabel: "iteration", ylabel: "model flux (Jy)", height: 170 }));
      }
      if (!["running", "queued"].includes(j.status)) {
        clearInterval(t); S.timers.delete(t);
        cancel.disabled = true; bar.classList.remove("indet");
        if (j.status === "error") { err.textContent = j.error || "failed"; msg.textContent = "failed"; }
        if (j.status === "cancelled") msg.textContent = "cancelled";
        onDone && onDone(j);
      }
    } catch (e) {
      msg.textContent = `lost contact with the job (${e.message})`;
      clearInterval(t); S.timers.delete(t);
    } finally { busy = false; }
  };
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
  S.dsId = null; loadDatasets();
  const noCasa = !S.info.casa ? h("div", { class: "card", style: "border-color:var(--warn)" },
    h("b", {}, "CASA is not available in this Python environment."),
    h("div", { class: "note" }, "You can browse, but inspecting an MS and dirty imaging need casatools + casatasks (pip install casatools casatasks). Start the app from your CASA environment, or use Import cubes.")) : null;
  const sel = h("div", { class: "note" }, "Select a measurement set (marked MS). Double-click a folder to open it.");
  const inspectBtn = h("button", { class: "primary", disabled: true }, "Inspect");
  const out = h("div", { style: "display:grid;gap:14px" });
  let ms = null;
  const br = fileBrowser({
    start: S.lastMSDir || "",
    badge: (e) => e.is_ms ? h("span", { class: "chip ok badge" }, "MS") : null,
    onSelect: (e) => {
      ms = e.is_ms ? e.path : null;
      sel.textContent = ms ? ms : "Not a measurement set. Double-click to open the folder.";
      inspectBtn.disabled = !ms || !S.info.casa;
    },
  });
  inspectBtn.onclick = async () => {
    S.lastMSDir = ms.replace(/\/[^/]+\/?$/, "");
    out.innerHTML = "";
    try {
      const job = await api("/api/ms/inspect", { method: "POST", body: { path: ms } });
      out.append(jobMonitor(job, { onDone: (j) => { if (j.status === "done") { out.innerHTML = ""; out.append(imagingForm(ms, j.result)); } } }));
    } catch (e) { toast(e.message, true); }
  };
  setMain(h("div", { class: "view-head" }, h("h1", {}, "Load a measurement set")), noCasa,
    h("div", { class: "card" }, br, h("div", { class: "row" }, sel, h("div", { style: "flex:1" }), inspectBtn)), out);
}

function imagingForm(ms, info) {
  const spws = info.spws || [], fields = info.target_fields || [];
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
  spwSel.onchange = suggest;
  fieldSel.onchange = () => { const f = fields.find((x) => x.name === fieldSel.value); if (f) F("phasecenter").value = f.phasecenter; F("name").value = fieldSel.value; };
  suggest();

  const summary = h("table", { class: "kv" },
    h("tr", {}, h("td", {}, "MS"), h("td", { class: "mono" }, ms)),
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
    p.ms = ms;
    run.disabled = true;
    try {
      const r = await api("/api/datasets/from_ms", { method: "POST", body: { name, params: p } });
      loadDatasets();
      mon.append(jobMonitor(r.job, { onDone: (j) => { loadDatasets(); if (j.status === "done") selectDataset(r.dataset); else run.disabled = false; } }));
    } catch (e) { toast(e.message, true); run.disabled = false; }
  };
  return h("div", { style: "display:grid;gap:14px" },
    h("div", { class: "grid2" }, h("div", { class: "card" }, h("h3", {}, "Measurement set"), summary), h("div", { class: "card scroll-x" }, h("h3", {}, "Science spectral windows"), spwTable)),
    h("div", { class: "card" }, h("h3", {}, "Dirty imaging (tclean, niter = 0)"),
      h("div", { class: "note" }, "Suggested values come from the spectral setup and the longest baseline: cell ≈ λ/Bmax / 5, image ≈ 1.2 × the primary beam. The PSF is made at twice the image size."),
      form, h("div", { class: "row" }, run)), mon);
}

// ============================================================ Import view
function viewImport() {
  S.dsId = null; loadDatasets();
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
    try { const m = await api("/api/datasets/import", { method: "POST", body }); await loadDatasets(); selectDataset(m.id); }
    catch (e) { toast(e.message, true); btn.disabled = false; }
  };
  setMain(h("div", { class: "view-head" }, h("h1", {}, "Import existing cubes")),
    h("div", { class: "card" }, h("div", { class: "note" }, "Click a folder marked cubes, or click the dirty and PSF files. Without cell and beam the app still deconvolves, but cannot make a restored cube or show offsets in arcsec. A beam_arcsec_deg.npy next to the cubes is picked up automatically."), br),
    h("div", { class: "card" }, form, h("div", { class: "row" }, btn)));
}

// ============================================================ Dataset view
async function selectDataset(id, tab, resId) {
  S.dsId = id; if (tab) S.tab = tab; if (resId !== undefined) S.resId = resId;
  await loadDatasets();
  let m;
  try { m = await api(`/api/datasets/${id}`); } catch (e) { toast(e.message, true); return; }
  if (m.status !== "ready" && S.tab !== "overview") S.tab = "overview";
  const tabs = h("div", { class: "tabs" }, [["overview", "Overview"], ["deconvolve", "Deconvolve"], ["results", `Results (${m.results.length})`]]
    .map(([k, label]) => h("button", { class: S.tab === k ? "on" : "", disabled: k !== "overview" && m.status !== "ready", onclick: () => selectDataset(id, k) }, label)));
  const body = h("div");
  const head = h("div", { class: "view-head" }, h("h1", {}, m.name),
    h("span", { class: "meta" }, `${m.nchan ?? "?"} channels · ${m.imsize ?? "?"} px · ${m.source === "ms" ? m.ms : m.source}`),
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
  const cell = m.cell_arcsec, unitD = m.width_kms ? "Jy/beam km/s" : "Jy/beam · ch";
  const pD = new ImagePanel({ title: "dirty, moment 0", unit: unitD, cell });
  const pP = new ImagePanel({ title: "PSF, central channel", unit: "peak 1", cell });
  const hasTruth = m.source === "toy";
  const pT = hasTruth ? new ImagePanel({ title: "true sky, moment 0", unit: "Jy/pixel km/s", cell }) : null;
  const plane = h("select", {}, h("option", { value: "mom0" }, "moment 0"), h("option", { value: "chan" }, "channel"));
  const slider = h("input", { type: "range", min: 0, max: (m.nchan || 1) - 1, value: Math.floor((m.nchan || 1) / 2), disabled: true });
  const vel = h("span", { class: "vel" }, "");
  let token = 0;
  async function load() {
    const my = ++token, isCh = plane.value === "chan", k = +slider.value;
    slider.disabled = !isCh; vel.textContent = isCh ? chanLabel(m, k) : "";
    const q = isCh ? `plane=chan&chan=${k}` : "plane=mom0";
    const [d, p, t] = await Promise.all([getArray(`/api/datasets/${m.id}/array?kind=dirty&${q}`),
      getArray(`/api/datasets/${m.id}/array?kind=psf&plane=chan&chan=${isCh ? k : Math.floor((m.nchan || 1) / 2)}`),
      hasTruth ? getArray(`/api/datasets/${m.id}/array?kind=truth&${q}`) : null]);
    if (my !== token) return;
    pD.set(d); pP.set(p); if (pT) pT.set(t);
    pD.setTitle(isCh ? "dirty, channel" : "dirty, moment 0"); pD.setUnit(isCh ? "Jy/beam" : unitD);
    if (pT) { pT.setTitle(isCh ? "true sky, channel" : "true sky, moment 0"); pT.setUnit(isCh ? "Jy/pixel" : "Jy/pixel km/s"); }
    draw();
  }
  async function draw() {
    const L = await lut(S.disp.cmap);
    for (const p of [pD, pT]) if (p && p.arr) { const [lo, hi] = p.autoLimits(S.disp.clip); p.render(L, S.disp.stretch, lo, hi); }
    if (pP.arr) pP.render(L, "linear", -0.2, 1);
  }
  plane.onchange = load; slider.oninput = load;

  const metaForm = h("form", { class: "form", onsubmit: (e) => e.preventDefault() },
    h("label", {}, "cell (arcsec)", h("input", { name: "cell_arcsec", type: "number", step: "any", value: m.cell_arcsec ?? "" })),
    h("label", {}, "beam major (″)", h("input", { name: "bmaj", type: "number", step: "any", value: m.beam ? m.beam[0] : "" })),
    h("label", {}, "beam minor (″)", h("input", { name: "bmin", type: "number", step: "any", value: m.beam ? m.beam[1] : "" })),
    h("label", {}, "beam PA (deg)", h("input", { name: "bpa", type: "number", step: "any", value: m.beam ? m.beam[2] : "" })),
    h("label", {}, "start velocity (km/s)", h("input", { name: "start_kms", type: "number", step: "any", value: m.start_kms ?? "" })),
    h("label", {}, "channel width (km/s)", h("input", { name: "width_kms", type: "number", step: "any", value: m.width_kms ?? "" })));
  const save = h("button", {}, "Save");
  save.onclick = async () => {
    const p = Object.fromEntries(new FormData(metaForm).entries()), num = (v) => v === "" ? null : +v;
    await api(`/api/datasets/${m.id}/meta`, { method: "POST", body: {
      cell_arcsec: num(p.cell_arcsec), start_kms: num(p.start_kms), width_kms: num(p.width_kms),
      beam: p.bmaj && p.bmin ? [+p.bmaj, +p.bmin, +(p.bpa || 0)] : null } });
    toast("Saved"); selectDataset(m.id);
  };
  const im = m.imaging || {};
  const kv = h("table", { class: "kv" },
    m.ms ? h("tr", {}, h("td", {}, "MS"), h("td", { class: "mono" }, m.ms)) : null,
    im.field ? h("tr", {}, h("td", {}, "field / spw"), h("td", {}, `${im.field} / ${im.spw}`)) : null,
    im.phasecenter ? h("tr", {}, h("td", {}, "phase centre"), h("td", { class: "mono" }, im.phasecenter)) : null,
    im.restfreq_ghz ? h("tr", {}, h("td", {}, "rest frequency"), h("td", {}, `${im.restfreq_ghz} GHz, ${im.weighting} weighting`)) : null,
    m.imported_from ? h("tr", {}, h("td", {}, "imported from"), h("td", { class: "mono" }, m.imported_from.join("\n"))) : null,
    m.beam ? h("tr", {}, h("td", {}, "clean beam"), h("td", {}, `${fmt(m.beam[0])}″ × ${fmt(m.beam[1])}″ @ ${fmt(m.beam[2])}°`)) : null,
    m.note ? h("tr", {}, h("td", {}, "note"), h("td", {}, m.note)) : null,
    h("tr", {}, h("td", {}, "stored in"), h("td", { class: "mono" }, `${S.info.workdir}/datasets/${m.id}`)));
  body.append(
    h("div", { class: "card" }, h("div", { class: "controls" }, h("label", {}, "show", plane), slider, vel, ...displayControls(draw)),
      h("div", { class: "panels" }, pD.el, pP.el, pT ? pT.el : null)),
    h("div", { class: "grid2", style: "margin-top:14px" }, h("div", { class: "card" }, h("h3", {}, "Dataset"), kv),
      h("div", { class: "card" }, h("h3", {}, "Geometry and units"), metaForm, h("div", { class: "row" }, save))));
  load().catch((e) => toast(e.message, true));
}

// ---- deconvolve ----
const DEFAULTS = {
  gpu_fista: { n_iter: 2000, k_sigma: 3, reweight: false, burn_in_iters: 1000, reweight_every: 500, reweight_eps: 1.0, rebin: 1 },
  gpu_pd: { n_iter: 5000, k_sigma: 3, reweight: false, burn_in_iters: 3000, reweight_every: 1000, reweight_eps: 1.0, rebin: 1, dirac: 0 },
  pd: { n_iter: 300, k_sigma: 3, reweight: false, burn_in_iters: 100, reweight_every: 25, reweight_eps: 0.01, positivity: true, tol: 1e-4 },
  fista: { n_iter: 60, k_sigma: 4, reweight: true, burn_in_iters: 20, reweight_eps: 0.01, positivity: true },
};

function deconvolveTab(body, m) {
  const running = (m.results || []).find((r) => (r.status === "running" || r.status === "queued") && r.job);
  const mon = h("div");
  const form = h("form", { class: "form", onsubmit: (e) => e.preventDefault() });
  const hasTorch = !!(S.devices && S.devices.torch);
  const backend = h("select", { name: "backend" },
    h("option", { value: "gpu_pd", disabled: !hasTorch }, `primal-dual, PyTorch (gpu_pd.py)${hasTorch ? "" : ": torch not installed"}`),
    h("option", { value: "gpu_fista", disabled: !hasTorch }, `FISTA, PyTorch (gpu_pd.py blocks)${hasTorch ? "" : ": torch not installed"}`),
    h("option", { value: "pd" }, "primal-dual, numpy (deconvolve_pd.py)"),
    h("option", { value: "fista" }, "FISTA + reweighting, numpy (deconvolve.py)"));
  backend.value = hasTorch ? "gpu_pd" : "pd";
  const devBox = h("div", { class: "devices" });
  function drawDevices() {
    const list = (S.devices && S.devices.devices) || [{ id: "cpu", name: "CPU", kind: "cpu" }];
    const gpuOK = backend.value.startsWith("gpu_");
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
      ? "One run uses one device. Tick several to spread a sweep (or several queued runs) across them; each device takes one run at a time."
      : "The numpy solvers run on the CPU only. Choose the PyTorch solver for CUDA or Apple MPS."));
  }
  function fill() {
    const b = backend.value, d = DEFAULTS[b];
    form.innerHTML = "";
    drawDevices();
    const num = (name, label, val, hint, step = "any") => h("label", {}, label, h("input", { name, type: "number", step, value: val ?? "" }), hint ? h("span", { class: "hint" }, hint) : null);
    const chk = (name, label, val) => h("label", { class: "check" }, h("input", { name, type: "checkbox", checked: !!val }), label);
    form.append(...[
      h("label", {}, "solver", backend),
      num("n_iter", "iterations (max)", d.n_iter, null, 1),
      h("label", {}, "threshold k (× noise σ)", h("input", { name: "k_sigma", value: d.k_sigma }), h("span", { class: "hint" }, "comma-separated for a sweep, e.g. 2, 3, 4")),
      b.startsWith("gpu_") ? num("rebin", "rebin factor", d.rebin, "b×b block binning before solving; the model is upsampled back", 1) : null,
      b === "gpu_pd" ? num("dirac", "pixel-domain L1 (λ_D)", d.dirac, "0 = off") : null,
      num("num_scales_2d", "spatial starlet scales", "", "empty = deepest", 1),
      num("num_levels_1d", "spectral CDF 9/7 levels", "", "empty = deepest", 1),
      h("label", {}, "line-free channels", h("input", { name: "noise_channels", placeholder: "e.g. 0-9, 80-89" }), h("span", { class: "hint" }, b.startsWith("gpu_") ? "strongly recommended: sets the thresholds from the real correlated noise" : "measures the real correlated noise per scale")),
      b === "fista" ? num("lrs_arcsec", "largest recoverable scale (″)", "", "zeroes larger starlet planes") : null,
      b === "pd" ? num("tol", "stop when relative change <", d.tol) : null,
      b === "gpu_pd" ? h("label", {}, "thresholds λ_b from", h("select", { name: "lambda_calibration" },
        h("option", { value: "dual" }, "dual certificate of noise (recommended)"),
        h("option", { value: "noise" }, "noise coefficients (gpu_pd.noise_lambdas)")),
        h("span", { class: "hint" }, "primal-dual only; see the note below the form")) : null,
      b.startsWith("gpu_") ? chk("null_test", "null test: run on pure noise instead of the data", false) : null,
      b.startsWith("gpu_") ? num("tol", "stop when relative change <", "", "per iteration, checked ~200 times; empty = run all iterations") : null,
      num("frame_every", "snapshot every N iterations", "", "empty = ~60 frames"),
      b.startsWith("gpu_") ? null : chk("positivity", "positivity", d.positivity),
      chk("reweight", "reweighted L1", d.reweight),
      num("burn_in_iters", "reweighting starts at iteration", d.burn_in_iters, null, 1),
      b !== "fista" ? num("reweight_every", "reweight every", d.reweight_every, null, 1) : null,
      num("reweight_eps", "reweighting ε", d.reweight_eps),
    ].filter(Boolean));
  }
  backend.onchange = fill; fill();
  const run = h("button", { class: "primary" }, "Run deconvolution");
  run.onclick = async () => {
    const p = { backend: backend.value };
    for (const el of form.elements) {
      if (!el.name || el.name === "backend") continue;
      if (el.type === "checkbox") p[el.name] = el.checked;
      else if (el.value !== "") p[el.name] = el.type === "number" ? +el.value : el.value;
    }
    const ks = String(p.k_sigma).split(/[,\s]+/).filter(Boolean).map(Number);
    if (!ks.length || ks.some(Number.isNaN)) return toast("threshold k: a number, or several separated by commas", true);
    p.k_sigma = ks.length > 1 ? ks : ks[0];
    p.devices = backend.value.startsWith("gpu_") ? [...S.devSel] : ["cpu"];
    if (!p.devices.length) return toast("choose at least one device", true);
    run.disabled = true;
    try {
      const r = await api(`/api/datasets/${m.id}/deconvolve`, { method: "POST", body: { params: p } });
      if (r.runs.length === 1) showMon(r.job, r.result);
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
  function showMon(job, resId) {
    mon.innerHTML = "";
    const dv = Math.abs(m.width_kms || 1);
    mon.append(jobMonitor(job, { live: true, cell: m.cell_arcsec, dv, onDone: (j) => {
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
  if (running) { showMon({ id: running.job, label: `${running.params.backend} (running)` }, running.id); }
}

// ---- results ----
async function resultsTab(body, m) {
  const res = (m.results || []).slice().reverse();
  if (!res.length) { body.append(h("p", { class: "note" }, "No results yet. Run a deconvolution first.")); return; }
  if (!S.resId || !res.find((r) => r.id === S.resId)) S.resId = (res.find((r) => r.status === "done") || res[0]).id;
  const tbl = h("table", { class: "list" }, h("tr", {}, ["result", "solver", "device", "k", "iterations", "flux (Jy)", "status", "started", ""].map((t) => h("th", {}, t))),
    res.map((r) => h("tr", { class: "click" + (r.id === S.resId ? " sel" : ""), onclick: () => selectDataset(m.id, "results", r.id) },
      h("td", { class: "mono" }, r.id), h("td", {}, r.params.backend), h("td", { class: "mono" }, r.device || "–"), h("td", {}, r.params.k_sigma ?? "–"),
      h("td", {}, r.summary ? r.summary.iterations : "–"), h("td", {}, r.summary ? fmt(r.summary.flux, 4) : "–"),
      h("td", {}, h("span", { class: "chip " + (r.status === "done" ? "ok" : (r.status === "running" || r.status === "queued") ? "run" : "err") }, r.status)),
      h("td", {}, ago(r.created)),
      h("td", {}, h("button", { class: "ghost danger", title: "delete this result", onclick: async (e) => { e.stopPropagation(); await api(`/api/datasets/${m.id}/results/${r.id}`, { method: "DELETE" }); S.resId = null; selectDataset(m.id, "results"); } }, "✕")))));
  body.append(h("div", { class: "card scroll-x" }, tbl));
  const r = await api(`/api/datasets/${m.id}/results/${S.resId}`);
  if (r.status !== "done") {
    body.append(h("div", { class: "card", style: "margin-top:14px" }, h("b", {}, `status: ${r.status}`), r.error ? h("div", { class: "err-box" }, r.error) : null));
    return;
  }
  const cell = m.cell_arcsec, hasV = !!m.width_kms, hasTruth = m.source === "toy", hasRestored = r.files.includes("restored.npy");
  const regionCb = (b) => setRegion(b);
  const P = {
    dirty: new ImagePanel({ title: "dirty", cell, onRegion: regionCb }),
    model: new ImagePanel({ title: "model", cell, onRegion: regionCb }),
    restored: hasRestored ? new ImagePanel({ title: "restored", cell, onRegion: regionCb }) : null,
    residual: new ImagePanel({ title: "residual", cell, onRegion: regionCb }),
    truth: hasTruth ? new ImagePanel({ title: "true sky", cell, onRegion: regionCb }) : null,
  };
  const units = { dirty: "Jy/beam", model: "Jy/pixel", restored: "Jy/beam", residual: "Jy/beam", truth: "Jy/pixel" };
  const plane = h("select", {}, h("option", { value: "mom0" }, "moment 0"), h("option", { value: "chan" }, "channel"));
  const slider = h("input", { type: "range", min: 0, max: (m.nchan || 1) - 1, value: Math.floor((m.nchan || 1) / 2), disabled: true });
  const vel = h("span", { class: "vel" });
  const link = h("input", { type: "checkbox", checked: true });
  const zoomBox = h("div", { class: "card", style: "margin-top:14px" }, h("div", { class: "note" }, "Drag a box on any image to zoom in and get integrated spectra of that region."));
  let token = 0, region = null;

  async function load() {
    const my = ++token, isCh = plane.value === "chan", k = +slider.value;
    slider.disabled = !isCh; vel.textContent = isCh ? chanLabel(m, k) : "";
    const q = isCh ? `plane=chan&chan=${k}` : "plane=mom0";
    const kinds = Object.keys(P).filter((k2) => P[k2]);
    const arrs = await Promise.all(kinds.map((kind) => getArray(`/api/datasets/${m.id}/array?kind=${kind}&res=${S.resId}&${q}`).catch(() => null)));
    if (my !== token) return;
    kinds.forEach((kind, i) => {
      const p = P[kind]; p.set(arrs[i]);
      p.setUnit(units[kind] + (isCh ? "" : hasV ? " km/s" : " · ch"));
      p.setTitle(`${kind === "truth" ? "true sky" : kind}${isCh ? "" : ", moment 0"}`);
    });
    draw();
  }
  async function draw() {
    const L = await lut(S.disp.cmap), R = await lut("RdBu_r");
    let shared = null;
    if (link.checked && P.dirty.arr && P.restored && P.restored.arr) {
      const a = P.dirty.autoLimits(S.disp.clip), b = P.restored.autoLimits(S.disp.clip);
      shared = [Math.min(a[0], b[0]), Math.max(a[1], b[1])];
    }
    for (const [kind, p] of Object.entries(P)) {
      if (!p || !p.arr) continue;
      let [lo, hi] = p.autoLimits(S.disp.clip);
      if (shared && (kind === "dirty" || kind === "restored")) [lo, hi] = shared;
      if (kind === "residual") {
        // residual: diverging map, symmetric about zero; on the dirty image's scale when linked
        const a = shared ? Math.max(Math.abs(shared[0]), Math.abs(shared[1])) : Math.max(Math.abs(lo), Math.abs(hi));
        p.render(R, "linear", -a, a);
        continue;
      }
      p.render(L, S.disp.stretch, lo, hi);
    }
    if (region) drawZoom();
  }
  plane.onchange = load; slider.oninput = load;

  // ---- region: zoomed crops + spectra ----
  let zoomPanels = [];
  async function setRegion(b) {
    region = b;
    for (const p of Object.values(P)) if (p) p.setBox(b);
    zoomBox.innerHTML = "";
    const head = h("div", { class: "row" }, h("h3", {}, "Region"),
      h("span", { class: "mono note" }, `x ${b.x0}–${b.x1}, y ${b.y0}–${b.y1}` + (cell ? `  (${((b.x1 - b.x0 + 1) * cell).toFixed(2)}″ × ${((b.y1 - b.y0 + 1) * cell).toFixed(2)}″)` : "")),
      h("div", { style: "flex:1" }), h("button", { onclick: () => { region = null; for (const p of Object.values(P)) if (p) p.setBox(null); zoomBox.innerHTML = ""; zoomBox.append(h("div", { class: "note" }, "Drag a box on any image to zoom in and get integrated spectra of that region.")); } }, "Clear"));
    zoomPanels = Object.entries(P).filter(([, p]) => p).map(([kind, p]) => { const z = new ImagePanel({ title: p.title, cell }); z.kind = kind; return z; });
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
      let [lo, hi] = z.autoLimits(S.disp.clip);
      if (z.kind === "residual") { const a = Math.max(Math.abs(lo), Math.abs(hi)); z.render(await lut("RdBu_r"), "linear", -a, a); continue; }
      z.render(L, S.disp.stretch, lo, hi);
    }
  }

  // ---- evolution + diagnostics ----
  const gifUrl = `/api/datasets/${m.id}/results/${S.resId}/evolution.gif`;
  const frames = r.frames || [];
  const scrub = new ImagePanel({ title: "snapshot", unit: hasV ? "Jy/pixel km/s" : "Jy/pixel · ch", cell });
  const fs = h("input", { type: "range", min: 0, max: Math.max(frames.length - 1, 0), value: Math.max(frames.length - 1, 0) });
  let finalLim = null;
  async function loadFrame() {
    const a = await getArray(`/api/datasets/${m.id}/results/${S.resId}/frame/${fs.value}`);
    scrub.set(a); scrub.setTitle(`iteration ${frames[+fs.value]}`);
    if (!finalLim) finalLim = [0, scrub.autoLimits(S.disp.clip)[1]];
    scrub.render(await lut(S.disp.cmap), S.disp.stretch, ...finalLim);
  }
  fs.oninput = loadFrame;
  const hist = r.history || [];
  const hx = hist.map((d) => d.iter);
  const diag = h("div", {},
    linePlot([{ name: "residual rms", color: cssVar("--s-dirty"), x: hx, y: hist.map((d) => d.residual_rms) }], { xlabel: "iteration", ylabel: "residual rms", height: 170, logy: true }),
    linePlot([{ name: "flux", color: cssVar("--s-model"), x: hx, y: hist.map((d) => d.flux) }], { xlabel: "iteration", ylabel: "model flux (Jy)", height: 170 }),
    hist.length && hist[0].peak_ratio !== undefined ? linePlot([{ name: "peak", color: cssVar("--s-restored"), x: hx, y: hist.map((d) => d.peak_ratio) }], { xlabel: "iteration", ylabel: "N(model)/dirty peak", height: 150 }) : null);
  const params = h("table", { class: "kv" }, Object.entries(r.params).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", { class: "mono" }, JSON.stringify(v)))),
    r.summary && r.summary.restored_beam ? h("tr", {}, h("td", {}, "restoring beam"), h("td", {}, r.summary.restored_beam.map((v) => fmt(v)).join(", "))) : null,
    h("tr", {}, h("td", {}, "files"), h("td", { class: "mono" }, `${S.info.workdir}/datasets/${m.id}/results/${S.resId}/`)));

  body.append(
    h("div", { class: "card", style: "margin-top:14px" },
      h("div", { class: "controls" }, h("label", {}, "show", plane), slider, vel, ...displayControls(draw),
        h("label", { title: "dirty and restored share one colour scale; residual uses the same scale, symmetric about zero" }, link, "common scale")),
      h("div", { class: "panels" }, Object.values(P).filter(Boolean).map((p) => p.el))),
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
function initTheme() {
  let t = null;
  try { t = localStorage.getItem("deconvgui-theme"); } catch { }
  if (t) document.documentElement.dataset.theme = t;
  $("#theme-btn").onclick = () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const nxt = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = nxt;
    try { localStorage.setItem("deconvgui-theme", nxt); } catch { }
    if (S.dsId) selectDataset(S.dsId);
  };
}

async function boot() {
  initTheme();
  try { S.devices = await api("/api/devices"); } catch { S.devices = null; }
  await loadInfo();
  await loadDatasets();
  $("#new-ms").onclick = () => viewMS();
  $("#new-import").onclick = () => viewImport();
  $("#new-toy").onclick = async () => {
    $("#new-toy").disabled = true;
    try { const m = await api("/api/datasets/toy", { method: "POST", body: {} }); await loadDatasets(); selectDataset(m.id); }
    catch (e) { toast(e.message, true); } finally { $("#new-toy").disabled = false; }
  };
  setInterval(pollJobs, 2500); pollJobs();
  setInterval(loadInfo, 60000);
  const first = S.datasets[0];
  if (first) selectDataset(first.id);
}
boot().catch((e) => toast(e.message, true));
