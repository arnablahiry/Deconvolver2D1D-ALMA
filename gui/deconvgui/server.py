"""
HTTP API + static front end. Arrays go to the browser as raw little-endian
float32 (header `X-Shape: ny,nx`), so it can colour-map, stretch, read pixel
values and select regions locally; everything heavy stays on this machine.
"""

import importlib.util
import urllib.parse
import os
import pickle
import secrets
import json
import socket
import subprocess
import sys
import threading

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .jobs import JobManager
from .render import GIF_VERSION, colormap_lut, make_gif
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
                    return Response("bad token: this link is from another deconvgui run. "
                                    "Open the URL printed in the terminal of the running server.",
                                    status_code=401, media_type="text/plain")
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
        allowed = {"name", "cell_arcsec", "beam", "width_kms", "start_kms", "distance_mpc", "example", "note",
                   "hires", "hires_ref"}   # hires: "yes"/"no", a better-resolved observation exists; hires_ref: its dataset
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
        paths = [_allowed(x) for x in (body.get("paths") or [body["path"]])]
        label = os.path.basename(paths[0].rstrip("/")) + (f" + {len(paths) - 1} more" if len(paths) > 1 else "")
        job = jobs.start("inspect", f"inspect {label}", dict(ms_paths=paths))
        return job.summary()

    @app.post("/api/datasets/from_ms")
    async def from_ms(request: Request):
        body = await request.json()
        p = body["params"]
        p["ms"] = [_allowed(x) for x in (p["ms"] if isinstance(p["ms"], list) else [p["ms"]])]
        spw_list = p.get("spw_list") or [p["spw"]] * len(p["ms"])
        if len(spw_list) != len(p["ms"]):
            raise HTTPException(400, "spw_list needs one entry per MS")
        p["spw_list"] = [str(x) for x in spw_list]
        name = body.get("name") or os.path.basename(p["ms"][0].rstrip("/"))
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

            b = p.get("backend") or ""
            method = "FISTA" if b.endswith("fista") else (
                "primal-dual (pdtau)" if b == "gpu_pd" and p.get("reweight") and (p.get("reweight_ref") or "tau_lambda") == "tau_lambda"
                else "primal-dual")
            dev = "" if b.startswith("gpu_") else " (numpy, CPU)"
            job = jobs.start("deconvolve", f"{method}{dev}, k={k} on {meta['name']}",
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

    gif_lock = threading.Lock()

    def _gif_current(p):
        try:
            with open(p + ".v") as fh:
                return int(fh.read().strip() or 0) >= GIF_VERSION
        except (OSError, ValueError):
            return False

    @app.get("/api/datasets/{ds_id}/results/{res_id}/evolution.gif")
    def gif(ds_id: str, res_id: str):
        """the evolution GIF; one made by an older version of the app is re-rendered in the current look
        from the saved frames (no new solve). Older frames were per pixel: they are converted to arcsec^-2."""
        d = store.res_dir(ds_id, res_id)
        p, fp, ip = (os.path.join(d, f) for f in ("evolution.gif", "frames.npy", "frames.json"))
        if not _gif_current(p) and os.path.isfile(fp) and os.path.isfile(ip):
            with gif_lock:
                if not _gif_current(p):
                    meta, info = _ds(ds_id), store.result_info(ds_id, res_id)
                    summ = info.get("summary") or {}
                    frames, cell = np.load(fp).astype(np.float32), meta.get("cell_arcsec")
                    v = "km/s" if meta.get("width_kms") else "ch"
                    unit = summ.get("frame_unit")
                    if not unit:                                  # an older run: frames in Jy/pixel
                        if cell:
                            frames, unit = frames / float(cell) ** 2, f"Jy {v} arcsec\u207b\u00b2"
                        else:
                            unit = f"Jy/pixel {v}"
                    make_gif(frames, read_json(ip), p, unit=unit, cell=cell)
        if not os.path.isfile(p):
            raise HTTPException(404, "no GIF yet")
        # revalidate every time (ETag): a GIF re-rendered in a new look must not be hidden by a cached copy
        return FileResponse(p, media_type="image/gif", filename=f"{ds_id}_{res_id}_evolution.gif",
                            headers={"Cache-Control": "no-cache"})

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

    def _mom0(ds_id, kind, res_id, dv, c0=None, c1=None):
        key = (ds_id, kind, res_id, c0, c1)
        with cache_lock:
            if key in _cache:
                return _cache[key]
        cube = _cube(ds_id, kind, res_id)
        lo = 0 if c0 is None else max(0, c0)
        hi = cube.shape[0] - 1 if c1 is None else min(cube.shape[0] - 1, c1)
        if kind == "psf":
            out = np.asarray(cube[(lo + hi) // 2], dtype=np.float32)
        else:
            out = (np.asarray(cube[lo:hi + 1], dtype=np.float64).sum(axis=0) * dv).astype(np.float32)
        with cache_lock:
            if len(_cache) > 64:
                _cache.clear()
            _cache[key] = out
        return out

    def _conv(cube, cell, beam):
        """cube (nz, ny, nx), Jy/pixel, * a peak-1 Gaussian beam (bmaj", bmin", pa deg) -> Jy/beam"""
        if os.path.join(REPO_ROOT, "simple") not in sys.path:
            sys.path.insert(0, os.path.join(REPO_ROOT, "simple"))
        from restore import convolve_with_beam
        return convolve_with_beam(np.asarray(cube, dtype=np.float64), float(cell), *beam)

    _conv_cache = {}

    def _mask_cube(ds_id, mrun, which):
        """MS-CLEAN run mrun's mask as a boolean cube: "clean" its own clean mask, "dendro" the astrodendro mask made for it"""
        kind = "clean_dmask" if which == "dendro" else "clean_mask"
        try:
            return np.asarray(_cube(ds_id, kind, mrun)) > 0
        except HTTPException:
            raise HTTPException(404, "no astrodendro mask yet for this MS-CLEAN run: make one" if which == "dendro"
                                else "this MS-CLEAN run has no clean mask saved (made before masks were kept, or no mask): run it again")

    def _model_conv_cube(ds_id, res, beam):
        """the whole 2D1D model cube * a beam (Jy/beam), the last one kept (a beam's boxes and the masked
        moment 0 / spectra ask for the same cube again and again)"""
        key = (ds_id, res, tuple(beam))
        with cache_lock:
            if key in _conv_cache:
                return _conv_cache[key]
        out = _conv(np.asarray(_cube(ds_id, "model", res)), _ds(ds_id)["cell_arcsec"], beam)
        with cache_lock:
            _conv_cache.clear()
            _conv_cache[key] = out
        return out

    def _beam(text):
        try:
            b = [float(v) for v in str(text).replace(" ", "").split(",")]
        except ValueError:
            b = []
        if len(b) != 3 or b[0] <= 0 or b[1] <= 0:
            raise HTTPException(400, "beam: bmaj,bmin,pa (arcsec, arcsec, deg), both axes > 0")
        return b

    @app.get("/api/datasets/{ds_id}/array")
    def array(ds_id: str, kind: str, plane: str = "mom0", chan: int = 0, res: str = None,
              c0: int = None, c1: int = None, mrun: str = None, beam: str = None, inmask: bool = False, mask: str = "clean"):
        """One plane: a channel, or moment 0 (over channels c0..c1 inclusive, default all). MS-CLEAN cubes
        (kind clean, clean_model, clean_residual) are those of run `mrun`."""
        meta = _ds(ds_id)
        if kind.startswith("clean"):
            res = mrun
        if kind in ("clean_mask", "clean_dmask"):   # 1 where the mask is set in this channel; mom0: the fraction of the masked
            cube = _cube(ds_id, kind, res)   # channels of c0..c1 in which it is set (the union fills most of a busy field)
            if plane == "mom0":
                lo, hi = 0 if c0 is None else max(0, c0), cube.shape[0] - 1 if c1 is None else min(cube.shape[0] - 1, c1)
                mk = np.asarray(cube[lo:hi + 1]) > 0
                n = max(1, int(mk.any(axis=(1, 2)).sum()))
                return _arr(mk.sum(axis=0) / n)
            return _arr(np.asarray(cube[max(0, min(chan, cube.shape[0] - 1))]) > 0)
        if kind == "model_conv":   # the 2D1D model * a beam of the caller's choosing (Analysis: target beam)
            bm = _beam(beam)
            if plane == "mom0" and inmask:   # moment 0 of only what lies inside MS-CLEAN run mrun's mask, channel by channel
                mk = _mask_cube(ds_id, mrun, mask)
                cube = _model_conv_cube(ds_id, res, bm)
                if mk.shape != cube.shape:
                    raise HTTPException(409, f"clean mask {list(mk.shape)} does not match the cube {list(cube.shape)}")
                lo, hi = 0 if c0 is None else max(0, c0), cube.shape[0] - 1 if c1 is None else min(cube.shape[0] - 1, c1)
                return _arr((cube[lo:hi + 1] * mk[lo:hi + 1]).sum(axis=0) * abs(meta.get("width_kms") or 1.0))
            if plane == "mom0":
                m = _mom0(ds_id, "model", res, abs(meta.get("width_kms") or 1.0), c0, c1)
            else:
                cube = _cube(ds_id, "model", res)
                m = np.asarray(cube[max(0, min(chan, cube.shape[0] - 1))], dtype=np.float32)
            return _arr(_conv(m[None], meta["cell_arcsec"], bm)[0])
        dv = abs(meta.get("width_kms") or 1.0)
        if plane == "mom0":
            a = _mom0(ds_id, kind, res, dv, c0, c1)
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

    # ---------------------------------------------------------- MS-CLEAN --
    @app.post("/api/datasets/{ds_id}/msclean")
    def msclean_start(ds_id: str, body: dict = Body(...)):
        """Run multiscale CLEAN on the visibilities this dataset was imaged from, on its grid."""
        meta = _ds(ds_id)
        if meta.get("status") != "ready" or not (meta.get("imaging") or {}).get("ms"):
            raise HTTPException(409, "MS-CLEAN needs a ready dataset imaged from measurement sets")
        ms = meta["imaging"]["ms"]
        gone = [x for x in (ms if isinstance(ms, list) else [ms]) if not os.path.isdir(x)]
        if gone:
            raise HTTPException(404, f"measurement set not found (moved?): {gone[0]}")
        p = body.get("params") or {}
        run_id = store.create_clean(ds_id, p)

        def started(job):
            store.update_clean(ds_id, run_id, status="running", started=job.started)

        def done(job):
            kw = dict(status=job.status, job=job.id, finished=job.finished)
            if job.result:
                kw.update(summary=job.result)
            if job.error:
                kw.update(error=job.error)
            store.update_clean(ds_id, run_id, **kw)
            _cache_clear()

        job = jobs.start("msclean", f"MS-CLEAN {run_id} on {meta['name']}",
                         dict(ds_dir=store.ds_dir(ds_id), run_dir=store.clean_dir(ds_id, run_id), params=p, meta=meta),
                         ds_id=ds_id, res_id=run_id, on_done=done, on_start=started)
        store.update_clean(ds_id, run_id, job=job.id, status=job.status)
        return dict(run=run_id, job=job.summary())

    @app.get("/api/datasets/{ds_id}/msclean")
    def msclean_list(ds_id: str):
        _ds(ds_id)
        return store.list_clean(ds_id)

    @app.get("/api/datasets/{ds_id}/msclean/{run_id}/dendro")
    def dendro_get(ds_id: str, run_id: str):
        try:
            path = os.path.join(store.clean_dir(ds_id, run_id), "dendro_mask.json")
        except KeyError:
            raise HTTPException(404, "no such MS-CLEAN run")
        if not os.path.isfile(path):
            raise HTTPException(404, "no astrodendro mask yet")
        return read_json(path)

    @app.post("/api/datasets/{ds_id}/msclean/{run_id}/dendro")
    def dendro_make(ds_id: str, run_id: str, body: dict = Body(default={})):
        """an astrodendro 3D mask on this MS-CLEAN run's image (worker.dendro_mask), in a separate process"""
        meta = _ds(ds_id)
        try:
            run_dir, run = store.clean_dir(ds_id, run_id), store.list_clean_one(ds_id, run_id)
        except KeyError:
            raise HTTPException(404, "no such MS-CLEAN run")
        cb, cell = (run.get("summary") or {}).get("beam"), meta.get("cell_arcsec")
        if run.get("status") != "done" or not cb or not cell:
            raise HTTPException(409, "the MS-CLEAN run is not finished")
        code = ("import json, sys; from deconvgui.worker import dendro_mask; "
                "print(json.dumps(dendro_mask(sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3]))))")
        try:
            r = subprocess.run([sys.executable, "-c", code, run_dir, str(1.1331 * cb[0] * cb[1] / cell ** 2), json.dumps(body.get("params") or {})],
                               capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "the dendrogram took over 30 minutes: raise min_value")
        if r.returncode:
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            raise HTTPException(500, f"astrodendro failed: {tail[-1] if tail else r.returncode}")
        with cache_lock:
            _cache.clear()
        return read_json(os.path.join(run_dir, "dendro_mask.json"))

    @app.delete("/api/datasets/{ds_id}/msclean/{run_id}")
    def msclean_delete(ds_id: str, run_id: str):
        try:
            store.delete_clean(ds_id, run_id)
        except KeyError:
            raise HTTPException(404, "no such MS-CLEAN run")
        _cache_clear()
        return dict(ok=True)

    @app.get("/api/datasets/{ds_id}/uv")
    def dataset_uv(ds_id: str, refresh: bool = False):
        """uv coverage of the measurement sets a dataset was imaged from: about 12 000 baselines per MS
        (target field, one spw), read with casatools in a separate process (CASA is not thread-safe),
        then kept in the dataset's uv.json."""
        meta = _ds(ds_id)
        ms = [meta["ms"]] if isinstance(meta.get("ms"), str) else list(meta.get("ms") or [])
        if not ms:
            raise HTTPException(404, "not imaged from a measurement set")
        path = os.path.join(store.ds_dir(ds_id), "uv.json")
        if os.path.isfile(path) and not refresh:
            return read_json(path)
        gone = [p for p in ms if not os.path.isdir(p)]
        if gone:
            raise HTTPException(404, f"measurement set not found (moved?): {gone[0]}")
        field = (meta.get("imaging") or {}).get("field") or ""
        code = "import json, sys; from deconvgui.worker import uv_coverage; uv_coverage(json.loads(sys.argv[1]), sys.argv[2], sys.argv[3])"
        try:
            r = subprocess.run([sys.executable, "-c", code, json.dumps(ms), field, path],
                               capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "reading the visibilities took over 10 minutes")
        if r.returncode or not os.path.isfile(path):
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            raise HTTPException(500, f"reading the visibilities failed: {tail[-1] if tail else r.returncode}")
        return read_json(path)

    @app.post("/api/datasets/{ds_id}/region")
    async def region(ds_id: str, request: Request):
        """Integrated spectra inside a pixel box (x0, x1, y0, y1 inclusive); with clean_mask (and mrun), only
        over the pixels inside MS-CLEAN run mrun's clean mask (true or "clean") or its astrodendro mask ("dendro"),
        channel by channel."""
        meta = _ds(ds_id)
        b = await request.json()
        res, mrun = b.get("res"), b.get("mrun")
        x0, x1 = sorted((int(b["x0"]), int(b["x1"])))
        y0, y1 = sorted((int(b["y0"]), int(b["y1"])))
        win = None
        if b.get("clean_mask"):   # true / "clean": the run's clean mask; "dendro": its astrodendro mask
            if not mrun:
                raise HTTPException(400, "clean_mask needs an MS-CLEAN run (mrun)")
            win = _mask_cube(ds_id, mrun, b["clean_mask"])[:, y0:y1 + 1, x0:x1 + 1]

        def total(cube):   # (nz, ny, nx) -> per-channel sum over the box (and the mask)
            a = np.asarray(cube[:, y0:y1 + 1, x0:x1 + 1], dtype=np.float64)
            if win is None:
                return a.sum(axis=(1, 2))
            if win.shape != a.shape:
                raise HTTPException(409, f"clean mask {list(win.shape)} does not match the cube {list(a.shape)}")
            return (a * win).sum(axis=(1, 2))
        beam, cell = meta.get("beam"), meta.get("cell_arcsec")
        beam_px = (1.1331 * beam[0] * beam[1] / cell ** 2) if beam and cell else None
        out = dict(box=[x0, x1, y0, y1], beam_area_px=beam_px, unit="Jy" if beam_px else "sum of pixels")
        kinds = ["dirty"] + (["model", "restored", "residual"] if res else [])
        rb = ((store.result_info(ds_id, res).get("summary") or {}).get("restored_beam")) if res else None
        rb_px = (1.1331 * rb[0] * rb[1] / cell ** 2) if rb and cell else beam_px     # the restoring beam
        if os.path.isfile(os.path.join(store.ds_dir(ds_id), "truth.npy")):
            kinds.append("truth")
        for kind in kinds:
            try:
                cube = _cube(ds_id, kind, res)
            except HTTPException:
                continue
            s = total(cube)
            if kind in ("dirty", "restored", "residual") and beam_px:
                s = s / (rb_px if kind == "restored" else beam_px)      # Jy/beam summed over pixels -> Jy
            out[kind] = s.tolist()
        if res and b.get("rbeam"):   # the 2D1D model * a chosen target beam, in place of the saved restored cube
            tb = _beam(",".join(str(v) for v in b["rbeam"]))
            s = total(_model_conv_cube(ds_id, res, tb))
            out["restored"] = (s / (1.1331 * tb[0] * tb[1] / cell ** 2)).tolist()
            out["restored_beam"] = tb
        if mrun:   # MS-CLEAN model * its clean beam, summed and divided by that beam's area -> Jy
            try:
                cb = (store.list_clean_one(ds_id, mrun).get("summary") or {}).get("beam")
                cube = _cube(ds_id, "clean", mrun)
                s = total(cube)
                out["clean"] = (s / (1.1331 * cb[0] * cb[1] / cell ** 2) if cb and cell else s).tolist()
                # + its residual: CLEAN's image (CASA .image); an early-stopped CLEAN leaves real emission there
                rs = total(_cube(ds_id, "clean_residual", mrun))
                out["clean_image"] = ((s + rs) / (1.1331 * cb[0] * cb[1] / cell ** 2) if cb and cell else s + rs).tolist()
            except (HTTPException, KeyError):
                pass
        if win is not None:
            out["clean_mask_px"] = win.sum(axis=(1, 2)).tolist()
        nz = len(out["dirty"])
        if meta.get("start_kms") is not None and meta.get("width_kms"):
            out["axis"] = [meta["start_kms"] + k * meta["width_kms"] for k in range(nz)]
            out["axis_label"] = "velocity (km/s)"
        else:
            out["axis"] = list(range(nz))
            out["axis_label"] = "channel"
        return out

    # ------------------------------------------------------ crop / save --
    GEOMETRY = ("cell_arcsec", "beam", "width_kms")

    def _extract(ds_id, req):
        """Cut a dataset to req["box"] (x0, x1, y0, y1 inclusive; None = whole field) and channels
        req["c0"]..req["c1"] (inclusive; None = all), with req["geometry"] overriding the cell, beam
        and channel width. Returns (arrays, meta). The dirty cube and true sky are cut to the box;
        the PSF is cut to twice the box around its peak, wherever the box is: it is shift-invariant,
        and the solvers want it at least twice the image and find its peak themselves."""
        meta = _ds(ds_id)
        if meta.get("status") != "ready":
            raise HTTPException(409, f"dataset is {meta.get('status')}, not ready")
        dirty = store.load(ds_id, "dirty")
        nz, ny, nx = dirty.shape
        box = req.get("box") or dict(x0=0, x1=nx - 1, y0=0, y1=ny - 1)
        x0, x1 = sorted((int(box["x0"]), int(box["x1"])))
        y0, y1 = sorted((int(box["y0"]), int(box["y1"])))
        c0 = 0 if req.get("c0") is None else int(req["c0"])
        c1 = nz - 1 if req.get("c1") is None else int(req["c1"])
        c0, c1 = sorted((c0, c1))
        if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny or min(x1 - x0, y1 - y0) < 7:
            raise HTTPException(400, f"crop box {[x0, x1, y0, y1]} must be at least 8 px and inside the {nx}x{ny} image")
        if c0 < 0 or c1 >= nz or c1 - c0 < 1:
            raise HTTPException(400, f"channel range {c0}-{c1} must hold at least 2 of the {nz} channels")
        w, hgt = x1 - x0 + 1, y1 - y0 + 1
        out = dict(dirty=np.ascontiguousarray(dirty[c0:c1 + 1, y0:y1 + 1, x0:x1 + 1]))
        psf = store.load(ds_id, "psf")
        py, px = np.unravel_index(np.argmax(psf[c0]), psf.shape[1:])
        out["psf"] = np.ascontiguousarray(psf[c0:c1 + 1, max(0, py - hgt):py + hgt, max(0, px - w):px + w])
        t = os.path.join(store.ds_dir(ds_id), "truth.npy")
        if os.path.isfile(t):
            out["truth"] = np.ascontiguousarray(np.load(t, mmap_mode="r")[c0:c1 + 1, y0:y1 + 1, x0:x1 + 1])
        m = {k: v for k, v in meta.items() if k not in ("id", "name", "created", "status", "job", "error", "example", "results")}
        m.update({k: v for k, v in (req.get("geometry") or {}).items() if k in GEOMETRY})
        whole = (x0, x1, y0, y1, c0, c1) == (0, nx - 1, 0, ny - 1, 0, nz - 1)
        if not whole:
            crop = dict(parent=ds_id, parent_name=meta.get("name"), box=[x0, x1, y0, y1], parent_size=[nx, ny],
                        chans=[c0, c1], parent_nchan=nz,
                        centre_offset_px=[(x0 + x1 + 1) / 2 - nx / 2, (y0 + y1 + 1) / 2 - ny / 2])
            m.update(crop=crop, crops=meta.get("crops", []) + [crop])
        if m.get("start_kms") is not None and m.get("width_kms"):
            m["start_kms"] = m["start_kms"] + c0 * m["width_kms"]
        m.update(imsize=w, nchan=c1 - c0 + 1)
        parts = ([f"{w}px"] if (w, hgt) != (nx, ny) else []) + ([f"ch {c0}-{c1}"] if (c0, c1) != (0, nz - 1) else [])
        m["name"] = f"{meta.get('name', ds_id)} crop {' '.join(parts)}" if parts else meta.get("name", ds_id)
        return out, m

    def _register(arrays, meta):
        """A new ready dataset from arrays (dirty, psf, optional truth) and metadata."""
        new_id = store.create_dataset(meta.pop("name"), dict(meta, status="ready"))
        d = store.ds_dir(new_id)
        try:
            for k, a in arrays.items():
                np.save(os.path.join(d, f"{k}.npy"), a)
        except Exception:
            store.delete_dataset(new_id)
            raise
        return store.meta(new_id)

    @app.post("/api/datasets/{ds_id}/crop")
    def dataset_crop(ds_id: str, req: dict = Body(...)):
        """Save in memory: the cut (and geometry) as a new dataset; the original is kept, so its
        results stay valid."""
        arrays, meta = _extract(ds_id, req)
        return _register(arrays, meta)

    @app.post("/api/datasets/{ds_id}/export")
    def dataset_export(ds_id: str, req: dict = Body(...)):
        """Save on disk: the cut as a pickle at req["path"] (inside the browsable roots)."""
        path = req.get("path") or ""
        if not path.endswith(".pkl"):
            raise HTTPException(400, "the file name must end in .pkl")
        rp = _allowed(path)
        if not os.path.isdir(os.path.dirname(rp)):
            raise HTTPException(400, f"no such folder: {os.path.dirname(rp)}")
        if os.path.exists(rp) and not req.get("overwrite"):
            raise HTTPException(409, f"{rp} already exists")
        arrays, meta = _extract(ds_id, req)
        meta = {k: v for k, v in meta.items() if k != "crops"}
        tmp = rp + ".part"
        with open(tmp, "wb") as f:
            pickle.dump(dict(format="deconvgui-1", meta=meta, **arrays), f, protocol=4)
        os.replace(tmp, rp)
        return dict(path=rp, size=os.path.getsize(rp))

    @app.post("/api/datasets/from_pkl")
    def dataset_from_pkl(req: dict = Body(...)):
        """Load saved data: a pickle written by Save on disk becomes a new dataset."""
        rp = _allowed(req.get("path") or "")
        if not os.path.isfile(rp):
            raise HTTPException(404, f"no such file: {rp}")
        try:
            with open(rp, "rb") as f:
                d = _ArraysOnly(f).load()
        except pickle.UnpicklingError as e:
            raise HTTPException(400, f"not a file saved by this app: {e}")
        if not isinstance(d, dict) or not all(isinstance(d.get(k), np.ndarray) and d[k].ndim == 3 for k in ("dirty", "psf")):
            raise HTTPException(400, "the file must hold 3-D 'dirty' and 'psf' arrays (nchan, ny, nx)")
        meta = {k: v for k, v in (d.get("meta") or {}).items() if k not in ("id", "created", "status", "job", "error", "example")}
        meta.update(nchan=int(d["dirty"].shape[0]), imsize=int(d["dirty"].shape[2]), loaded_from=rp,
                    name=meta.get("name") or os.path.splitext(os.path.basename(rp))[0])
        arrays = {k: d[k] for k in ("dirty", "psf", "truth") if isinstance(d.get(k), np.ndarray)}
        return _register(arrays, meta)

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
        live = j.live   # one read: the frame and the progress of the same iteration
        if live is None:
            raise HTTPException(404, "no frame yet")
        it, frame, fraction, message = live
        return _arr(frame, extra={"X-Iter": str(it), "X-Fraction": "" if fraction is None else str(fraction),
                                  "X-Message": urllib.parse.quote(message or "")})

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


class _ArraysOnly(pickle.Unpickler):
    """Unpickles numpy arrays and plain Python data only: a pickle can otherwise run any code."""
    SAFE = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"), ("numpy.core.numeric", "_frombuffer"),
            ("numpy._core.numeric", "_frombuffer"), ("builtins", "bytearray"), ("collections", "OrderedDict")}

    def find_class(self, module, name):
        if (module, name) in self.SAFE:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def _arr(a, extra=None):
    a = np.ascontiguousarray(a, dtype="<f4")
    finite = a[np.isfinite(a)]
    headers = {"X-Shape": f"{a.shape[0]},{a.shape[1]}", "Cache-Control": "no-store",
               "Access-Control-Expose-Headers": "X-Shape, X-Iter, X-Fraction, X-Message"}
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
