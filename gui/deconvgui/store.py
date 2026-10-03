"""
On-disk store of datasets and deconvolution results.

    <workdir>/datasets/<dataset_id>/
        meta.json          name, source, imaging parameters, cell, beam, velocities
        dirty.npy          (nz, ny, nx) dirty cube, Jy/beam
        psf.npy            (nz, py, px) dirty beam, peak 1 (ideally 2x the image)
        results/<result_id>/
            params.json    deconvolution parameters + status
            model.npy      Jy/pixel
            restored.npy   model convolved with the clean beam, Jy/beam (if beam known)
            residual.npy   dirty - N(model), Jy/beam
            frames.npy     moment-0 snapshots during the solve
            frames.json    iteration of each snapshot
            history.json   per-iteration diagnostics
            evolution.gif

Everything the GUI shows is rebuilt from these files, so results survive a
server restart or a dropped SSH session.
"""

import json
import os
import re
import shutil
import time
import uuid

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMPLE_DIR = os.path.join(REPO_ROOT, "simple")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


class Store:
    def __init__(self, workdir):
        self.workdir = os.path.abspath(os.path.expanduser(workdir))
        self.ds_root = os.path.join(self.workdir, "datasets")
        os.makedirs(self.ds_root, exist_ok=True)

    # ------------------------------------------------------------ datasets --
    def new_dataset_id(self, name):
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:40] or "dataset"
        return f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}-{uuid.uuid4().hex[:4]}"

    def ds_dir(self, ds_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", ds_id or ""):
            raise KeyError(ds_id)
        d = os.path.join(self.ds_root, ds_id)
        if not os.path.isdir(d):
            raise KeyError(ds_id)
        return d

    def create_dataset(self, name, meta):
        ds_id = self.new_dataset_id(name)
        d = os.path.join(self.ds_root, ds_id)
        os.makedirs(os.path.join(d, "results"))
        meta = dict(meta, id=ds_id, name=name, created=time.time(), status=meta.get("status", "empty"))
        write_json(os.path.join(d, "meta.json"), meta)
        return ds_id

    def meta(self, ds_id):
        return read_json(os.path.join(self.ds_dir(ds_id), "meta.json"))

    def update_meta(self, ds_id, **kw):
        p = os.path.join(self.ds_dir(ds_id), "meta.json")
        m = read_json(p)
        m.update(kw)
        write_json(p, m)
        return m

    def list_datasets(self):
        out = []
        for ds_id in sorted(os.listdir(self.ds_root), reverse=True):
            p = os.path.join(self.ds_root, ds_id, "meta.json")
            if os.path.isfile(p):
                m = read_json(p)
                m["results"] = self.list_results(ds_id)
                out.append(m)
        out.sort(key=lambda m: (not m.get("example", False), -m.get("created", 0)))
        return out

    def delete_dataset(self, ds_id):
        shutil.rmtree(self.ds_dir(ds_id))

    # ------------------------------------------------------------- results --
    def res_dir(self, ds_id, res_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", res_id or ""):
            raise KeyError(res_id)
        d = os.path.join(self.ds_dir(ds_id), "results", res_id)
        if not os.path.isdir(d):
            raise KeyError(res_id)
        return d

    def create_result(self, ds_id, params):
        rdir = os.path.join(self.ds_dir(ds_id), "results")
        os.makedirs(rdir, exist_ok=True)
        n = len([x for x in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, x))]) + 1
        res_id = f"r{n:03d}-{params.get('backend', 'x')}"
        while os.path.exists(os.path.join(rdir, res_id)):
            n += 1
            res_id = f"r{n:03d}-{params.get('backend', 'x')}"
        os.makedirs(os.path.join(rdir, res_id))
        write_json(os.path.join(rdir, res_id, "params.json"),
                   dict(params=params, id=res_id, created=time.time(), status="queued"))
        return res_id

    def result_info(self, ds_id, res_id):
        d = self.res_dir(ds_id, res_id)
        info = read_json(os.path.join(d, "params.json"))
        info["files"] = sorted(f for f in os.listdir(d))
        return info

    def update_result(self, ds_id, res_id, **kw):
        p = os.path.join(self.res_dir(ds_id, res_id), "params.json")
        m = read_json(p)
        m.update(kw)
        write_json(p, m)
        return m

    def list_results(self, ds_id):
        rdir = os.path.join(self.ds_dir(ds_id), "results")
        if not os.path.isdir(rdir):
            return []
        out = []
        for r in sorted(os.listdir(rdir)):
            p = os.path.join(rdir, r, "params.json")
            if os.path.isfile(p):
                out.append(read_json(p))
        return out

    def delete_result(self, ds_id, res_id):
        shutil.rmtree(self.res_dir(ds_id, res_id))

    # -------------------------------------------------------------- arrays --
    def load(self, ds_id, kind, res_id=None, mmap=True):
        """kind: dirty | psf  (dataset) or model | restored | residual (result)."""
        if kind in ("dirty", "psf"):
            path = os.path.join(self.ds_dir(ds_id), f"{kind}.npy")
        elif kind in ("model", "restored", "residual"):
            path = os.path.join(self.res_dir(ds_id, res_id), f"{kind}.npy")
        else:
            raise KeyError(kind)
        if not os.path.isfile(path):
            raise FileNotFoundError(kind)
        return np.load(path, mmap_mode="r" if mmap else None)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
