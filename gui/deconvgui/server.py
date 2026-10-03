"""
HTTP API + static front end. Arrays go to the browser as raw little-endian
float32 (header `X-Shape: ny,nx`), so it can colour-map, stretch, read pixel
values and select regions locally; everything heavy stays on this machine.
"""

import importlib.util
import os
import secrets
import json
import socket
import subprocess
import sys
import threading

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .jobs import JobManager
from .render import colormap_lut
from .repoinfo import repo_info
from .store import REPO_ROOT, Store, read_json, write_json

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def create_app(workdir, roots, token):
    store = Store(workdir)
    jobs = JobManager(store)
    roots = [os.path.realpath(os.path.expanduser(r)) for r in roots]
    app = FastAPI(title="deconvgui", docs_url=None, redoc_url=None)
    cache_lock = threading.Lock()

    # ------------------------------------------------------------ auth --
    @app.middleware("http")
    async def check_token(request: Request, call_next):
        if token:
            q = request.query_params.get("token")
            if q is not None:
                if not secrets.compare_digest(q, token):
                    return Response("bad token", status_code=401)
                resp = RedirectResponse(request.url.path)
                resp.set_cookie("deconvgui_token", token, httponly=True, samesite="strict")
                return resp
            c = request.cookies.get("deconvgui_token") or request.headers.get("x-token")
            if not c or not secrets.compare_digest(c, token):
                return Response("Open the URL printed in the terminal (it carries the access token).",
                                status_code=401, media_type="text/plain")
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------ misc --
    @app.get("/api/info")
    def info():
        return dict(version=__version__, casa=importlib.util.find_spec("casatools") is not None,
                    torch=importlib.util.find_spec("torch") is not None,
                    repo=repo_info(), workdir=store.workdir, roots=roots, host=socket.gethostname())

    @app.get("/api/colormap/{name}")
    def colormap(name: str):
        try:
            return colormap_lut(name).tolist()
        except KeyError:
            raise HTTPException(404, f"unknown colormap {name}")

    # --------------------------------------------------------- browsing --
    def _allowed(path):
        rp = os.path.realpath(os.path.expanduser(path))
        if not any(rp == r or rp.startswith(r + os.sep) for r in roots):
            raise HTTPException(403, f"{rp} is outside the browsable roots ({', '.join(roots)}); "
                                     f"start the app with --root to add more")
        return rp

    @app.get("/api/browse")
    def browse(path: str = ""):
        rp = _allowed(path or roots[0])
        if not os.path.isdir(rp):
            raise HTTPException(404, "not a directory")
        entries = []
        try:
            names = sorted(os.listdir(rp), key=str.lower)
        except PermissionError:
            raise HTTPException(403, "permission denied")
        for n in names:
            if n.startswith("."):
                continue
            p = os.path.join(rp, n)
            is_dir = os.path.isdir(p)
            e = dict(name=n, is_dir=is_dir)
            if is_dir:
                e["is_ms"] = os.path.isfile(os.path.join(p, "table.dat")) and os.path.isdir(os.path.join(p, "ANTENNA"))
                e["is_cubes"] = (os.path.isfile(os.path.join(p, "dirty.npy")) and
                                 os.path.isfile(os.path.join(p, "psf.npy")))
            else:
                e["ext"] = os.path.splitext(n)[1].lower()
                try:
                    e["size"] = os.path.getsize(p)
                except OSError:
                    pass
            entries.append(e)
        parent = os.path.dirname(rp)
        return dict(path=rp, parent=parent if any(parent == r or parent.startswith(r + os.sep) for r in roots) else None,
                    entries=entries, roots=roots)

    # --------------------------------------------------------- datasets --
    @app.get("/api/datasets")
    def datasets():
        return store.list_datasets()

    def _ds(ds_id):
        try:
            return store.meta(ds_id)
        except KeyError:
            raise HTTPException(404, "no such dataset")

    @app.get("/api/datasets/{ds_id}")
    def dataset(ds_id: str):
        m = _ds(ds_id)
        m["results"] = store.list_results(ds_id)
        return m

    @app.post("/api/datasets/{ds_id}/meta")
    async def dataset_meta(ds_id: str, request: Request):
        _ds(ds_id)
        body = await request.json()
        allowed = {"name", "cell_arcsec", "beam", "width_kms", "start_kms", "example", "note"}
        return store.update_meta(ds_id, **{k: v for k, v in body.items() if k in allowed})

    @app.delete("/api/datasets/{ds_id}")
    def dataset_delete(ds_id: str):
        _ds(ds_id)
        store.delete_dataset(ds_id)
        _cache_clear()
        return dict(ok=True)

    @app.post("/api/ms/inspect")
    async def ms_inspect(request: Request):
        body = await request.json()
        ms = _allowed(body["path"])
        job = jobs.start("inspect", f"inspect {os.path.basename(ms)}", dict(ms_path=ms))
        return job.summary()

    @app.post("/api/datasets/from_ms")
    async def from_ms(request: Request):
        body = await request.json()
        p = body["params"]
        p["ms"] = _allowed(p["ms"])
        name = body.get("name") or os.path.basename(p["ms"].rstrip("/"))
        meta = dict(source="ms", ms=p["ms"], imaging=p, cell_arcsec=float(p["cell_arcsec"]),
                    width_kms=float(p["width_kms"]), start_kms=float(p["start_kms"]),
                    nchan=int(p["nchan"]), imsize=int(p["imsize"]), restfreq_ghz=float(p["restfreq_ghz"]),
                    status="imaging")
        ds_id = store.create_dataset(name, meta)

        def done(job):
            if job.status == "done":
                store.update_meta(ds_id, status="ready", beam=job.result["beam"])
            else:
                store.update_meta(ds_id, status=job.status, error=job.error)

        job = jobs.start("image", f"dirty imaging: {name}", dict(ds_dir=store.ds_dir(ds_id), params=p),
                         ds_id=ds_id, on_done=done)
        store.update_meta(ds_id, job=job.id)
        return dict(dataset=ds_id, job=job.summary())

    @app.post("/api/datasets/import")
    async def import_cubes(request: Request):
        """Register an existing dirty/PSF pair: a directory holding dirty.npy +
        psf.npy (e.g. simple/data), or two FITS / .npy files."""
        body = await request.json()
        src = _allowed(body["path"])
        dirty_p = psf_p = None
        if os.path.isdir(src):
            for cand_d, cand_p in (("dirty.npy", "psf.npy"), ("dirty_cube.fits", "dirty_beam_cube.fits")):
                if os.path.isfile(os.path.join(src, cand_d)) and os.path.isfile(os.path.join(src, cand_p)):
                    dirty_p, psf_p = os.path.join(src, cand_d), os.path.join(src, cand_p)
                    break
        if body.get("dirty") and body.get("psf"):
            dirty_p, psf_p = _allowed(body["dirty"]), _allowed(body["psf"])
        if not dirty_p:
            raise HTTPException(400, "no dirty.npy + psf.npy (or dirty_cube.fits + dirty_beam_cube.fits) found")
        dirty, psf = _load_any(dirty_p), _load_any(psf_p)
        if dirty.shape[0] != psf.shape[0]:
            raise HTTPException(400, f"channel mismatch: dirty {dirty.shape} vs psf {psf.shape}")
        beam = body.get("beam")
        bfile = os.path.join(os.path.dirname(dirty_p), "beam_arcsec_deg.npy")
        if not beam and os.path.isfile(bfile):
            beam = [float(v) for v in np.load(bfile)]
        name = body.get("name") or os.path.basename(src.rstrip("/"))
        meta = dict(source="import", imported_from=[dirty_p, psf_p], beam=beam,
                    cell_arcsec=_f(body.get("cell_arcsec")), width_kms=_f(body.get("width_kms")),
                    start_kms=_f(body.get("start_kms")), nchan=int(dirty.shape[0]), imsize=int(dirty.shape[1]),
                    status="ready")
        ds_id = store.create_dataset(name, meta)
        d = store.ds_dir(ds_id)
        np.save(os.path.join(d, "dirty.npy"), dirty.astype(np.float32))
        np.save(os.path.join(d, "psf.npy"), psf.astype(np.float32))
        return store.meta(ds_id)

    @app.post("/api/datasets/toy")
    async def toy(request: Request):
        from .toy import make_toy
        body = await request.json() if (await request.body()) else {}
        dirty, psf, truth, meta = make_toy(ny=int(body.get("size", 128)), nz=int(body.get("nchan", 40)))
        ds_id = store.create_dataset(body.get("name") or "synthetic disc", dict(meta, status="ready", example=True))
        d = store.ds_dir(ds_id)
        np.save(os.path.join(d, "dirty.npy"), dirty)
        np.save(os.path.join(d, "psf.npy"), psf)
        np.save(os.path.join(d, "truth.npy"), truth)
        return store.meta(ds_id)

    # ---------------------------------------------------- deconvolution --
    @app.post("/api/datasets/{ds_id}/deconvolve")
    async def deconvolve(ds_id: str, request: Request):
        """Queue one run, or one run per value when k_sigma is a list (a sweep).
        `devices`: list of devices the runs may use; each device takes one run at
        a time, so a sweep over several devices runs in parallel."""
        meta = _ds(ds_id)
        if meta.get("status") != "ready":
            raise HTTPException(409, f"dataset is {meta.get('status')}, not ready")
        params = (await request.json())["params"]
        devices = params.pop("devices", None) or ["cpu"]
        known = {d["id"] for d in _devices()["devices"]}
        bad = [d for d in devices if d not in known]
        if bad:
            raise HTTPException(400, f"unknown device(s) {bad}; available: {sorted(known)}")
        if params.get("backend") not in ("gpu_pd", "gpu_fista"):
            devices = ["cpu"]           # the numpy solvers only run on the CPU
        ks = params.get("k_sigma")
        ks = ks if isinstance(ks, list) else [ks]
        out = []
        for k in ks:
            p = dict(params, k_sigma=k)
            res_id = store.create_result(ds_id, dict(p, devices=devices))

            def started(job, res_id=res_id):
                store.update_result(ds_id, res_id, status="running", device=job.device, started=job.started)

            def done(job, res_id=res_id):
                kw = dict(status=job.status, job=job.id, finished=job.finished, started=job.started, device=job.device)
                if job.result:
                    kw.update(summary=job.result)
                if job.error:
                    kw.update(error=job.error)
                store.update_result(ds_id, res_id, **kw)
                _cache_clear()

            job = jobs.start("deconvolve", f"{p.get('backend')} k={k} on {meta['name']}",
                             dict(ds_dir=store.ds_dir(ds_id), res_dir=store.res_dir(ds_id, res_id),
                                  params=p, meta=meta),
                             ds_id=ds_id, res_id=res_id, on_done=done, on_start=started, devices=devices)
            store.update_result(ds_id, res_id, job=job.id, status=job.status)
            out.append(dict(result=res_id, job=job.summary()))
        return dict(runs=out, result=out[0]["result"], job=out[0]["job"])

    _dev_cache = {}

    def _devices(refresh=False):
        if refresh or "v" not in _dev_cache:
            code = "import json; from deconvgui.worker import devices_job; print(json.dumps(devices_job()))"
            try:
                r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
                _dev_cache["v"] = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception as e:
                _dev_cache["v"] = dict(devices=[dict(id="cpu", name="CPU", kind="cpu")], torch=None, error=repr(e))
        return _dev_cache["v"]

    @app.get("/api/devices")
    def devices(refresh: bool = False):
        d = dict(_devices(refresh))
        d["load"] = jobs.device_load()
        return d

    @app.get("/api/datasets/{ds_id}/results/{res_id}")
    def result(ds_id: str, res_id: str):
        _ds(ds_id)
        try:
            info = store.result_info(ds_id, res_id)
        except KeyError:
            raise HTTPException(404, "no such result")
        d = store.res_dir(ds_id, res_id)
        for name in ("history", "frames"):
            p = os.path.join(d, f"{name}.json")
            info[name] = read_json(p) if os.path.isfile(p) else None
        return info

    @app.delete("/api/datasets/{ds_id}/results/{res_id}")
    def result_delete(ds_id: str, res_id: str):
        try:
            store.delete_result(ds_id, res_id)
        except KeyError:
            raise HTTPException(404, "no such result")
        _cache_clear()
        return dict(ok=True)

    @app.get("/api/datasets/{ds_id}/results/{res_id}/evolution.gif")
    def gif(ds_id: str, res_id: str):
        p = os.path.join(store.res_dir(ds_id, res_id), "evolution.gif")
        if not os.path.isfile(p):
            raise HTTPException(404, "no GIF yet")
        return FileResponse(p, media_type="image/gif", filename=f"{ds_id}_{res_id}_evolution.gif")

    @app.get("/api/datasets/{ds_id}/results/{res_id}/frame/{i}")
    def result_frame(ds_id: str, res_id: str, i: int):
        p = os.path.join(store.res_dir(ds_id, res_id), "frames.npy")
        if not os.path.isfile(p):
            raise HTTPException(404, "no frames")
        f = np.load(p, mmap_mode="r")
        i = max(0, min(i, f.shape[0] - 1))
        return _arr(np.asarray(f[i]))

    # ----------------------------------------------------------- arrays --
    _cache = {}

    def _cache_clear():
        with cache_lock:
            _cache.clear()

    def _cube(ds_id, kind, res_id):
        if kind == "truth":
            p = os.path.join(store.ds_dir(ds_id), "truth.npy")
            if not os.path.isfile(p):
                raise HTTPException(404, "no truth cube")
            return np.load(p, mmap_mode="r")
        try:
            return store.load(ds_id, kind, res_id)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, f"no {kind} cube")

    def _mom0(ds_id, kind, res_id, dv):
        key = (ds_id, kind, res_id)
        with cache_lock:
            if key in _cache:
                return _cache[key]
        cube = _cube(ds_id, kind, res_id)
        if kind == "psf":
            out = np.asarray(cube[cube.shape[0] // 2], dtype=np.float32)
        else:
            out = (np.asarray(cube, dtype=np.float64).sum(axis=0) * dv).astype(np.float32)
        with cache_lock:
            if len(_cache) > 64:
                _cache.clear()
            _cache[key] = out
        return out

    @app.get("/api/datasets/{ds_id}/array")
    def array(ds_id: str, kind: str, plane: str = "mom0", chan: int = 0, res: str = None):
        meta = _ds(ds_id)
        dv = abs(meta.get("width_kms") or 1.0)
        if plane == "mom0":
            a = _mom0(ds_id, kind, res, dv)
        else:
            cube = _cube(ds_id, kind, res)
            chan = max(0, min(chan, cube.shape[0] - 1))
            a = np.asarray(cube[chan], dtype=np.float32)
        if kind == "psf":
            # show the central image-sized part of the (2x) PSF
            ny, nx = meta.get("imsize") or a.shape[0], meta.get("imsize") or a.shape[1]
            py, px = np.unravel_index(np.argmax(a), a.shape)
            a = a[max(0, py - ny // 2): py + ny // 2, max(0, px - nx // 2): px + nx // 2]
        return _arr(a)

    @app.post("/api/datasets/{ds_id}/region")
    async def region(ds_id: str, request: Request):
        """Integrated spectra inside a pixel box (x0, x1, y0, y1 inclusive)."""
        meta = _ds(ds_id)
        b = await request.json()
        res = b.get("res")
        x0, x1 = sorted((int(b["x0"]), int(b["x1"])))
        y0, y1 = sorted((int(b["y0"]), int(b["y1"])))
        beam, cell = meta.get("beam"), meta.get("cell_arcsec")
        beam_px = (1.1331 * beam[0] * beam[1] / cell ** 2) if beam and cell else None
        out = dict(box=[x0, x1, y0, y1], beam_area_px=beam_px, unit="Jy" if beam_px else "sum of pixels")
        kinds = ["dirty"] + (["model", "restored", "residual"] if res else [])
        if os.path.isfile(os.path.join(store.ds_dir(ds_id), "truth.npy")):
            kinds.append("truth")
        for kind in kinds:
            try:
                cube = _cube(ds_id, kind, res)
            except HTTPException:
                continue
            s = np.asarray(cube[:, y0:y1 + 1, x0:x1 + 1], dtype=np.float64).sum(axis=(1, 2))
            if kind in ("dirty", "restored", "residual") and beam_px:
                s = s / beam_px                       # Jy/beam summed over pixels -> Jy
            out[kind] = s.tolist()
        nz = len(out["dirty"])
        if meta.get("start_kms") is not None and meta.get("width_kms"):
            out["axis"] = [meta["start_kms"] + k * meta["width_kms"] for k in range(nz)]
            out["axis_label"] = "velocity (km/s)"
        else:
            out["axis"] = list(range(nz))
            out["axis_label"] = "channel"
        return out

    # ------------------------------------------------------------- jobs --
    @app.get("/api/jobs")
    def job_list():
        return jobs.list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, since: int = 0):
        try:
            return jobs.get(job_id).summary(stats_since=since)
        except KeyError:
            raise HTTPException(404, "unknown job (the server may have restarted)")

    @app.get("/api/jobs/{job_id}/frame")
    def job_frame(job_id: str):
        try:
            j = jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, "unknown job")
        if j.frame is None:
            raise HTTPException(404, "no frame yet")
        return _arr(j.frame, extra={"X-Iter": str(j.frame_iter)})

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        try:
            return jobs.cancel(job_id).summary()
        except KeyError:
            raise HTTPException(404, "unknown job")

    # mark results that were running when the server last stopped
    for m in store.list_datasets():
        for r in m.get("results", []):
            if r.get("status") in ("running", "queued"):
                store.update_result(m["id"], r["id"], status="interrupted")
        if m.get("status") == "imaging":
            store.update_meta(m["id"], status="interrupted")

    app.state.store, app.state.jobs = store, jobs
    return app


def _arr(a, extra=None):
    a = np.ascontiguousarray(a, dtype="<f4")
    finite = a[np.isfinite(a)]
    headers = {"X-Shape": f"{a.shape[0]},{a.shape[1]}", "Cache-Control": "no-store",
               "Access-Control-Expose-Headers": "X-Shape, X-Iter"}
    if finite.size:
        headers["X-Range"] = f"{float(finite.min())},{float(finite.max())}"
    if extra:
        headers.update(extra)
    return Response(np.nan_to_num(a).tobytes(), media_type="application/octet-stream", headers=headers)


def _load_any(path):
    if path.endswith(".npy"):
        a = np.load(path)
    elif path.endswith((".fits", ".fit", ".fits.gz")):
        from astropy.io import fits
        with fits.open(path) as h:
            a = np.asarray(h[0].data, dtype=np.float32)
    else:
        raise HTTPException(400, f"unsupported file type: {path}")
    a = np.squeeze(np.asarray(a, dtype=np.float32))
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise HTTPException(400, f"{path}: expected a 3D cube, got shape {a.shape}")
    return np.nan_to_num(a)


def _f(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
