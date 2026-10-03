"""
Job bodies. Each runs in a FRESH worker process (multiprocessing "spawn"),
which imports the repository's `simple/` modules at that moment -- so a
change to the solver (or a `git pull`) is used by the next job with no
restart, and a CASA crash can never take the server down.

Messages to the server go through `q` as (kind, payload) tuples:
    ("log", str)                       a line for the job log
    ("progress", dict)                 fraction 0..1 (or None = indeterminate), message
    ("stat", dict)                     per-iteration diagnostics for the live plot
    ("frame", (iteration, float32 2D)) a moment-0 snapshot of the current iterate
    ("result", dict)                   final payload (job-specific)
    ("error", str)                     traceback text
"""

import inspect
import json
import os
import sys
import time
import traceback

import numpy as np


class Cancelled(Exception):
    pass


def _setup_paths(repo_root):
    for sub in ("simple", "scripts"):
        p = os.path.join(repo_root, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    # CASA writes timestamped logs into the CWD: keep them out of the repo root.
    logs = os.path.join(repo_root, "casa_logs")
    os.makedirs(logs, exist_ok=True)
    os.chdir(repo_root)


def run_job(kind, args, q, cancel, repo_root):
    try:
        _setup_paths(repo_root)
        fn = {"inspect": inspect_ms_job, "image": image_job, "deconvolve": deconvolve_job,
              "devices": devices_job}[kind]
        result = fn(q=q, cancel=cancel, **args)
        q.put(("result", result))
    except Cancelled:
        q.put(("cancelled", "cancelled by user"))
    except BaseException:
        q.put(("error", traceback.format_exc()))


def _kwargs_for(fn, kw):
    """Only pass the keyword arguments `fn` actually accepts, so the GUI keeps
    working when the solver's signature changes in the repo."""
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in sig.parameters}


# ======================================================================
# Inspect an MS
# ======================================================================
def inspect_ms_job(ms_path, q, cancel):
    q.put(("progress", dict(fraction=None, message="reading MS metadata (casatools)")))
    from casatools import msmetadata, measures, quanta  # noqa: F401

    info = None
    try:
        import inspect_ms  # scripts/inspect_ms.py
        info = inspect_ms.describe(ms_path)
    except ImportError:
        pass

    md = msmetadata()
    md.open(ms_path)
    try:
        fields = list(md.fieldnames())
        try:
            target_ids = list(md.fieldsforintent("OBSERVE_TARGET#ON_SOURCE"))
        except Exception:
            target_ids = list(range(len(fields)))
        targets = []
        for fid in target_ids:
            pc = md.phasecenter(int(fid))
            ra = pc["m0"]["value"]
            dec = pc["m1"]["value"]
            targets.append(dict(id=int(fid), name=fields[fid], phasecenter=_fmt_j2000(ra, dec)))
        if info is None:
            spws = []
            for s in range(md.nspw()):
                f = md.chanfreqs(s) / 1e9
                if f.size < 2:
                    continue
                spws.append(dict(spw=s, nchan=int(f.size), f_lo=float(f.min()), f_hi=float(f.max()),
                                 chanwidth_khz=float(np.mean(np.abs(md.chanwidths(s))) / 1e3)))
            info = dict(path=ms_path, fields=fields, spws=spws, nant=int(md.nantennas()))
    finally:
        md.close()
    info["target_fields"] = targets
    info["mosaic"] = len(targets) > 1 and len({t["name"] for t in targets}) == 1
    return info


def _fmt_j2000(ra_rad, dec_rad):
    ra_h = (np.degrees(ra_rad) % 360.0) / 15.0
    h = int(ra_h); m = int((ra_h - h) * 60); s = ((ra_h - h) * 60 - m) * 60
    sign = "-" if dec_rad < 0 else "+"
    d_abs = abs(np.degrees(dec_rad))
    d = int(d_abs); dm = int((d_abs - d) * 60); ds = ((d_abs - d) * 60 - dm) * 60
    return f"J2000 {h:02d}:{m:02d}:{s:07.4f} {sign}{d:02d}.{dm:02d}.{ds:06.3f}"


# ======================================================================
# MS -> dirty cube + PSF (CASA tclean, niter=0)
# ======================================================================
def image_job(ds_dir, params, q, cancel):
    import grid_with_casa as g

    p = params
    q.put(("progress", dict(fraction=None, message="CASA: gridding PSF and dirty cube (tclean niter=0)")))
    q.put(("log", f"[image] tclean_dirty_cube on {p['ms']} spw={p['spw']} field={p['field']}"))
    t0 = time.time()
    out_dir = os.path.join(ds_dir, "casa")
    call = dict(ms_list=[p["ms"]], spw_list=[p["spw"]], field=p["field"], phasecenter=p["phasecenter"],
                out_dir=out_dir, imsize=int(p["imsize"]), cell_arcsec=float(p["cell_arcsec"]),
                start_kms=float(p["start_kms"]), width_kms=float(p["width_kms"]), nchan=int(p["nchan"]),
                restfreq_ghz=float(p["restfreq_ghz"]), weighting=p.get("weighting", "natural"),
                log_name=f"gui_tclean_{os.path.basename(ds_dir)}.log")
    out = g.tclean_dirty_cube(**_kwargs_for(g.tclean_dirty_cube, call))
    q.put(("log", f"[image] tclean done in {time.time() - t0:.0f} s"))

    dirty = _read_fits_cube(out["dirty"])
    psf = _read_fits_cube(out["psf"])
    np.save(os.path.join(ds_dir, "dirty.npy"), dirty)
    np.save(os.path.join(ds_dir, "psf.npy"), psf)
    bmaj, bmin, bpa = (float(v) for v in out["beam"])
    q.put(("log", f"[image] dirty {dirty.shape}, psf {psf.shape}, beam {bmaj:.3f}\" x {bmin:.3f}\" @ {bpa:.1f} deg"))
    return dict(shape=list(dirty.shape), psf_shape=list(psf.shape), beam=[bmaj, bmin, bpa])


def _read_fits_cube(path):
    from astropy.io import fits
    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 2:
        data = data[None]
    return np.nan_to_num(data, nan=0.0)


# ======================================================================
# Deconvolution
# ======================================================================
class _WatchedOperator:
    """
    Wraps the fast operator. Both solvers call `operator.gradient(...)`
    exactly once per iteration on the current iterate, so this is where the
    GUI hooks in -- progress, live moment-0 frames, per-iteration stats and
    cancellation -- without any change to the solver code.
    """

    def __init__(self, op, lipschitz, n_iter, frame_every, dv, q, cancel):
        self._op = op
        self.lipschitz = lipschitz
        self.n_iter = n_iter
        self.frame_every = max(1, int(frame_every))
        self.dv = dv
        self.q = q
        self.cancel = cancel
        self.it = 0
        self.frames, self.frame_iters = [], []
        self.t0 = time.time()

    def __getattr__(self, name):
        return getattr(self._op, name)

    def gradient(self, x):
        if self.cancel.is_set():
            raise Cancelled()
        g = self._op.gradient(x)
        it = self.it
        self.it += 1
        if it % self.frame_every == 0:
            m0 = (np.asarray(x).sum(axis=0) * self.dv).astype(np.float32)
            self.frames.append(m0)
            self.frame_iters.append(it)
            self.q.put(("frame", (it, m0)))
        el = time.time() - self.t0
        eta = el / max(self.it, 1) * max(self.n_iter - self.it, 0)
        self.q.put(("stat", dict(iter=it, residual_rms=float(g.std()), flux=float(np.asarray(x).sum()))))
        self.q.put(("progress", dict(fraction=min(self.it / self.n_iter, 1.0),
                                     message=f"iteration {self.it}/{self.n_iter}  ({el:.0f} s, ~{eta:.0f} s left)")))
        return g


def deconvolve_job(ds_dir, res_dir, params, meta, q, cancel, device="cpu"):
    from uv_to_image import ShiftInvariantOperator

    p = dict(params)
    backend = p.pop("backend", "pd")
    p.pop("devices", None)
    if backend in ("gpu_pd", "gpu_fista"):
        return torch_job(ds_dir, res_dir, p, meta, q, cancel, device, algo="pd" if backend == "gpu_pd" else "fista")
    if device != "cpu":
        q.put(("log", f"[warn] the {backend} solver is numpy-only; running on the CPU, not {device}"))
    frame_every = p.pop("frame_every", None)
    lrs_arcsec = p.pop("lrs_arcsec", None)
    noise_spec = p.pop("noise_channels", None)

    q.put(("progress", dict(fraction=None, message="loading cubes")))
    dirty = np.load(os.path.join(ds_dir, "dirty.npy")).astype(np.float64)
    psf = np.load(os.path.join(ds_dir, "psf.npy")).astype(np.float64)
    q.put(("log", f"[setup] dirty {dirty.shape}, psf {psf.shape}, backend={backend}"))
    if psf.shape[1] < 2 * dirty.shape[1] or psf.shape[2] < 2 * dirty.shape[2]:
        q.put(("log", "[warn] the PSF is smaller than twice the image: the normal operator may be "
                      "indefinite (see uv_to_image.ShiftInvariantOperator)"))

    fast = ShiftInvariantOperator(psf, dirty)
    if backend == "pd":
        import deconvolve_pd as mod
        q.put(("progress", dict(fraction=None, message="power iteration for the step size")))
        lip = mod.field_lipschitz(fast, dirty.shape)
    elif backend == "fista":
        import deconvolve as mod
        lip = fast.lipschitz
    else:
        raise ValueError(f"unknown backend {backend}")

    n_iter = int(p.get("n_iter") or getattr(mod, "N_ITER", 100))
    p["n_iter"] = n_iter
    if frame_every in (None, "", 0):
        frame_every = max(1, n_iter // 60)
    dv = abs(float(meta.get("width_kms") or 1.0))

    if noise_spec:
        p["noise_channels"] = parse_channels(noise_spec, dirty.shape[0])
    if lrs_arcsec and meta.get("cell_arcsec"):
        p["max_scale_px"] = float(lrs_arcsec) / float(meta["cell_arcsec"])
    for k in ("num_scales_2d", "num_levels_1d"):
        if p.get(k) in ("", None):
            p.pop(k, None)

    op = _WatchedOperator(fast, lip, n_iter, frame_every, dv, q, cancel)
    q.put(("progress", dict(fraction=0.0, message="computing thresholds")))
    kw = _kwargs_for(mod.deconvolve, dict(p, operator=op, verbose=True, log=lambda m: q.put(("log", m))))
    dropped = sorted(set(p) - set(kw) - {"operator"})
    if dropped:
        q.put(("log", f"[warn] {mod.__name__}.deconvolve() does not take: {', '.join(dropped)} (ignored)"))
    x, history = mod.deconvolve(psf, dirty, **kw)
    return _finalize(np.asarray(x, dtype=np.float32), history, op.frames, op.frame_iters, op.it,
                     dirty, fast, meta, params, dv, res_dir, q, device="cpu")


def _finalize(x, history, frames, frame_iters, n_it, dirty, fast, meta, params, dv, res_dir, q, device):
    """Save model, residual, restored cube, frames, history and the GIF (all solvers)."""
    q.put(("progress", dict(fraction=1.0, message="saving model, residual, restored cube")))
    np.save(os.path.join(res_dir, "model.npy"), x)
    residual = (dirty - fast.apply(x.astype(np.float64))).astype(np.float32)
    np.save(os.path.join(res_dir, "residual.npy"), residual)

    restored_beam = None
    beam = meta.get("beam")
    cell = meta.get("cell_arcsec")
    if beam and cell:
        from restore import convolve_with_beam
        rb = params.get("restore_beam") or beam
        restored = convolve_with_beam(x, float(cell), *[float(v) for v in rb]).astype(np.float32)
        np.save(os.path.join(res_dir, "restored.npy"), restored)
        restored_beam = [float(v) for v in rb]

    frames = list(frames) + [(x.sum(axis=0) * dv).astype(np.float32)]
    frame_iters = list(frame_iters) + [n_it - 1]
    np.save(os.path.join(res_dir, "frames.npy"), np.stack(frames))
    with open(os.path.join(res_dir, "frames.json"), "w") as f:
        json.dump(frame_iters, f)
    with open(os.path.join(res_dir, "history.json"), "w") as f:
        json.dump(history, f, default=float)

    from .render import make_gif
    make_gif(np.stack(frames), frame_iters, os.path.join(res_dir, "evolution.gif"))
    q.put(("log", f"[done] {n_it} iterations on {device}, model flux {float(x.sum()):.3f} Jy"))
    return dict(iterations=n_it, flux=float(x.sum()), restored_beam=restored_beam, device=device)


def parse_channels(spec, nz):
    """'0-9, 80-89' -> [0..9, 80..89] (clipped to the cube)."""
    out = []
    for part in str(spec).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted({c for c in out if 0 <= c < nz})


# ======================================================================
# Devices
# ======================================================================
def devices_job(q=None, cancel=None):
    """What this machine offers: always the CPU, plus every CUDA GPU visible to
    this process (respects CUDA_VISIBLE_DEVICES / the SLURM allocation) and
    Apple's Metal GPU (MPS) when PyTorch is built with it."""
    out = [dict(id="cpu", name=f"CPU ({os.cpu_count()} cores)", kind="cpu")]
    try:
        import torch
    except ImportError:
        return dict(devices=out, torch=None)
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        out.append(dict(id=f"cuda:{i}", name=pr.name, mem_gb=round(pr.total_memory / 1e9, 1), kind="cuda"))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        out.append(dict(id="mps", name="Apple GPU (Metal, MPS)", kind="mps"))
    return dict(devices=out, torch=torch.__version__)


class _CPUGeneratorTorch:
    """Stand-in for the `torch` module inside gpu_pd when running on MPS:
    random numbers are drawn with a CPU generator (seeded, reproducible on
    any device) and moved to the device. Everything else is plain torch."""

    def __init__(self, torch):
        self._t = torch

    def __getattr__(self, name):
        return getattr(self._t, name)

    def Generator(self, device=None):
        return self._t.Generator("cpu")

    def randn(self, *size, device=None, dtype=None, generator=None, **kw):
        if len(size) == 1 and isinstance(size[0], (tuple, list, self._t.Size)):
            size = tuple(size[0])
        a = self._t.randn(size, dtype=dtype or self._t.float32, generator=generator, **kw)
        return a.to(device) if device is not None else a


def _use_device(G, torch, device, q):
    """Point gpu_pd at `device` (it picks cuda-or-cpu by itself at import)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{device} requested but CUDA is not available to this process")
        torch.cuda.set_device(dev)
    if dev.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but this PyTorch build / macOS does not provide it")
    G.DEV = dev
    if dev.type == "mps":
        _apply_mps_compat(G, torch, dev)
    q.put(("log", f"[setup] gpu_pd on {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else "")))


def _apply_mps_compat(G, torch, dev):
    """Metal has no float64: build the PSF transfer function in float64 on the
    CPU (as gpu_pd does on CUDA) and move the complex64 result; draw random
    numbers on the CPU (see _CPUGeneratorTorch). Only gpu_pd's module state in
    this worker process is changed; the file on disk is not touched."""
    G.torch = _CPUGeneratorTorch(torch)
    Base = G.Op

    class OpMPS(Base):
        def __init__(self, psf, dirty):
            nz, self.ny, self.nx = dirty.shape
            psf = np.asarray(psf, dtype=np.float64)
            psf = psf / psf.max(axis=(1, 2), keepdims=True)
            py, px = np.unravel_index(np.argmax(psf[0]), psf[0].shape)
            self.gy, self.gx = max(psf.shape[1], 2 * self.ny), max(psf.shape[2], 2 * self.nx)
            big = np.zeros((nz, self.gy, self.gx))
            big[:, :psf.shape[1], :psf.shape[2]] = psf
            big = np.roll(big, (-py, -px), axis=(1, 2))
            otf = torch.fft.rfft2(torch.tensor(big, dtype=torch.float64))
            self.otf = otf.to(torch.complex64).to(dev)
            self.d = G.to_t(dirty)
            self.bound = float(self.otf.abs().max())

    G.Op = OpMPS


def _auto_scales(ny, nx, nz, J=None, L=None):
    """J: deepest starlet whose reflect padding (2**J px) still fits the image;
    L: deepest CDF 9/7 level that divides the channel count (gpu_pd packs the
    bands without padding, so nz must be a multiple of 2**L)."""
    jmax = max(1, int(np.floor(np.log2(min(ny, nx)))) - 2)
    J = min(int(J), jmax) if J else min(7, jmax)
    lmax = 0
    while lmax < 6 and nz % (2 ** (lmax + 1)) == 0:
        lmax += 1
    L = min(int(L), lmax) if L not in (None, "") else min(5, lmax)
    return J, L


class _Converged(Exception):
    pass


def torch_pd_job(ds_dir, res_dir, p, meta, q, cancel, device):
    return torch_job(ds_dir, res_dir, p, meta, q, cancel, device, algo="pd")


def torch_job(ds_dir, res_dir, p, meta, q, cancel, device, algo="pd"):
    """PyTorch solvers on CPU / CUDA / MPS, both on gpu_pd's building blocks
    (operator, 2D-1D dictionary, noise lambdas, step size):

      algo="pd"     gpu_pd.solve_pd (Condat-Vu primal-dual), as scripts/make_pd_evolution_gif.py
      algo="fista"  x = max(0, W^-1 soft(W(z - tau grad), tau lambda)) with FISTA momentum,
                    the loop of scripts/make_fista_run.py (reweighting: same convention)
    """
    import torch
    import gpu_pd as G
    from uv_to_image import ShiftInvariantOperator

    _use_device(G, torch, device, q)
    q.put(("progress", dict(fraction=None, message=f"loading cubes onto {device}")))
    dirty = np.load(os.path.join(ds_dir, "dirty.npy")).astype(np.float64)
    psf = np.load(os.path.join(ds_dir, "psf.npy")).astype(np.float64)
    b = int(p.get("rebin") or 1)
    d_s, psf_s = G.rebin(dirty, psf, b) if b > 1 else (dirty, psf)
    nz, ny, nx = d_s.shape
    J, L = _auto_scales(ny, nx, nz, p.get("num_scales_2d"), p.get("num_levels_1d"))
    if p.get("num_levels_1d") not in (None, "") and int(p["num_levels_1d"]) != L:
        q.put(("log", f"[warn] {nz} channels allow at most {L} CDF 9/7 levels (nz must be a multiple of 2**L)"))
    q.put(("log", f"[setup] {algo} on {device}; cube {d_s.shape}" + (f" after {b}x{b} rebinning" if b > 1 else "") +
           f", {J} starlet scales x {L} CDF 9/7 levels"))

    op = G.Op(psf_s, d_s)
    D = G.Dict2D1D(d_s.shape, J, L)
    k = float(p.get("k_sigma", 3.0))
    noise = parse_channels(p["noise_channels"], nz) if p.get("noise_channels") else []
    q.put(("progress", dict(fraction=None, message="noise model and thresholds")))
    if p.get("null_test"):
        if len(noise) < 2:
            raise ValueError("the null test needs line-free channels (to measure the noise power spectrum)")
        ps = G.measured_ps(G.noise_planes(op.d, noise))
        if b > 1:
            raise ValueError("run the null test without rebinning")
        op.d = G.noise_with_ps(ps, tuple(op.d.shape), seed=int(p.get("null_seed") or 12345))
        dirty = op.d.float().cpu().numpy().astype(np.float64)       # residual / dirty panels refer to the noise cube
        q.put(("log", "[null test] the dirty cube is REPLACED by a pure-noise realization with the measured "
                      "power spectrum: the model should come out (close to) zero"))
    calib = p.get("lambda_calibration") or "noise"
    if len(noise) >= 2:
        lam, _, _, s_pix = G.noise_lambdas(D, op.d, noise, k=k)
        q.put(("log", f"[setup] lambda = {k} x max(real, white) noise MAD per sub-band, from {len(noise)} line-free channels; pixel sigma {s_pix:.3e}"))
        if calib == "dual":
            lam = _dual_lambdas(G, D, op, noise, k, lam, q)
    else:
        s_pix = 1.4826 * float((op.d - op.d.median()).abs().median())
        white = G.torch.randn(tuple(op.d.shape), device=G.DEV, dtype=G.DT,
                              generator=G.torch.Generator(G.DEV).manual_seed(101)) * s_pix
        lam = k * D.band_mad(D.W(white))
        q.put(("log", "[warn] no line-free channels given: thresholds from WHITE noise only, which "
                      "underestimates the correlated interferometric noise (set line-free channels)"))
    beta = G.field_lipschitz(op)

    n_iter = int(p.get("n_iter") or 5000)
    tol = float(p.get("tol") or 0.0)
    reweight = None
    if p.get("reweight"):
        reweight = dict(start=int(p.get("burn_in_iters") or n_iter // 3), every=int(p.get("reweight_every") or 1000),
                        eps=float(p.get("reweight_eps") or 1.0))
    frame_every = int(p.get("frame_every") or max(1, n_iter // 60))
    stat_every = max(1, n_iter // 400)
    hist_every = max(1, n_iter // 200)
    dv = abs(float(meta.get("width_kms") or 1.0))
    frames, frame_iters, history = [], [], []
    st = dict(last=0.0, prev=None, x=None, it=0)
    t0 = time.time()
    dmax = float(op.d.max())

    def cb(it, x, g):
        if cancel.is_set():
            raise Cancelled()
        st["x"], st["it"] = x, it
        if it % frame_every == 0 or it == 1:
            m0 = (x.sum(0) * dv).float().cpu().numpy()
            if b > 1:
                m0 = G.upsample(m0[None], b)[0]
            m0 = m0.astype(np.float32)
            frames.append(m0)
            frame_iters.append(it)
            q.put(("frame", (it, m0)))
        if it % stat_every == 0:
            q.put(("stat", dict(iter=it, residual_rms=float(g.std()), flux=float(x.sum()))))
        # relative change per iteration, measured between iterations it-1 and it
        if it % hist_every == 0 and st["prev"] is not None:
            rel = float((x - st["prev"]).norm() / max(float(x.norm()), 1e-30))
            history.append(dict(iter=it, residual_rms=float(g.std()), flux=float(x.sum()),
                                peak_ratio=float((op.d + g).max()) / dmax, rel_change=rel))
            if it % max(hist_every, 10) == 0:
                q.put(("log", f"iter {it:6d}  flux {float(x.sum()):9.3f}  change {rel:.2e}  resid rms {float(g.std()):.3e}"))
            if tol and it > 50 and rel < tol:
                raise _Converged()
        if (it + 1) % hist_every == 0:
            st["prev"] = x.clone()
        now = time.time()
        if now - st["last"] > 0.5 or it == n_iter:
            st["last"] = now
            el = now - t0
            q.put(("progress", dict(fraction=it / n_iter,
                                    message=f"iteration {it}/{n_iter} on {device}  ({el:.0f} s, ~{el / it * (n_iter - it):.0f} s left)")))

    q.put(("progress", dict(fraction=0.0, message=f"iterating on {device}")))
    try:
        if algo == "pd":
            x, _, _, _ = G.solve_pd(op, D, lam, n_iter=n_iter, beta=beta, reweight=reweight,
                                    dirac=float(p.get("dirac") or 0.0), print_every=10 ** 9,
                                    log=lambda m: None, callback=cb)
        else:
            x = _torch_fista(G, torch, op, D, lam, beta, n_iter, reweight, cb)
        n_done = n_iter
    except _Converged:
        x, n_done = st["x"], st["it"]
        q.put(("log", f"converged at iteration {n_done}: relative change < {tol:g}"))
    x = x.float().cpu().numpy().astype(np.float64)
    if b > 1:
        x = G.upsample(x, b)
    fast = ShiftInvariantOperator(psf, dirty)
    return _finalize(x.astype(np.float32), history, frames, frame_iters, n_done, dirty, fast, meta,
                     dict(p), dv, res_dir, q, device=device)


def _dual_lambdas(G, D, op, noise, k, lam_noise, q, seed=7, max_cg=300):
    """lambda_b from the DUAL certificate of pure noise instead of the noise's own
    coefficients. The primal-dual optimum satisfies d - N x = W^T u with |u| <= lambda,
    so x = 0 is the answer for a noise-only cube n exactly when n = W^T u has a solution
    with |u| <= lambda. For the redundant 2D-1D frame W^T W is not the identity, and the
    coefficients u of such a certificate are larger (and distributed differently over the
    sub-bands) than those of W n. Here u = W (W^T W)^-1 n for a noise realization n with
    the measured power spectrum (conjugate gradient), and lambda_b = k x max(MAD_b(u),
    MAD_b(W n)), per spatial scale and spectral band."""
    torch = G.torch
    ps = G.measured_ps(G.noise_planes(op.d, noise))
    n = G.noise_with_ps(ps, tuple(op.d.shape), seed=seed)
    A = lambda v: D.WT(D.W(v))
    x = torch.zeros_like(n)
    r = n.clone()
    pdir = r.clone()
    rs = float((r * r).sum())
    nn = float((n * n).sum())
    for i in range(max_cg):
        Ap = A(pdir)
        a = rs / float((pdir * Ap).sum())
        x += a * pdir
        r -= a * Ap
        rn = float((r * r).sum())
        if rn < 1e-10 * nn:
            break
        pdir = r + (rn / rs) * pdir
        rs = rn
    lam = torch.maximum(k * D.band_mad(D.W(x)), lam_noise)
    ratio = [round(float((lam[j] / lam_noise[j]).median()), 2) for j in range(lam.shape[0])]
    q.put(("log", f"[setup] dual-certificate calibration ({i + 1} CG steps, residual {np.sqrt(rn / nn):.1e}): "
                  f"lambda / noise-coefficient lambda per spatial scale (median over bands) = {ratio}"))
    return lam


def _torch_fista(G, torch, op, D, lam, beta, n_iter, reweight, cb):
    """FISTA on the 2D-1D dictionary, exactly the loop of scripts/make_fista_run.py:
    threshold T0 = tau * lambda (the same lambda_b as the primal-dual solve),
    x = max(0, W^-1 soft(W(z - tau grad(z)), T)), momentum on z; reweighting
    T = T0 / (|W x| / T0 + eps) at the scheduled iterations."""
    tau = 1.0 / beta
    T0 = tau * lam
    T = T0
    soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)
    x = torch.zeros_like(op.d)
    z = x.clone()
    t = 1.0
    for it in range(1, n_iter + 1):
        if reweight and it > reweight["start"] and (it - 1 - reweight["start"]) % reweight["every"] == 0:
            T = T0 / (D.W(x).abs() / T0 + reweight["eps"])
        g = op.grad(z)
        xn = D.Winv(soft(D.W(z - tau * g), T)).clamp_(min=0)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = xn + ((t - 1) / tn) * (xn - x)
        x, t = xn, tn
        cb(it, x, g)
    return x
