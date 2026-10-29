#!/usr/bin/env python
"""
Training set for a galaxy-trained learnlet (2D filters for the 2D-1D deconvolution): FIRE [CII] 158 um CHANNEL MAPS
(the 2D dictionary acts on every channel, not on moment 0), from the 1x cubes (/nas/datasets/FIRE_IFU_cubes/1x_new:
1024 px of 19.5 pc over the 20 kpc box, 20 km/s channels).

Per galaxy (C1 excluded: its IFU cubes duplicate C2's, already in the raw import), snapshot (60, 65, 70, 75, 80, 85;
76 is a near-duplicate of 75) and line of sight:
  * flux: observed Jy/pixel at the snapshot's redshift (as scripts/fire_preprocess.py)
  * channels: those carrying >= CHAN_FRAC of the brightest channel's flux
  * pixel scales: block sums by 1, 2, 4 (19.5, 39, 78 pc; 1024, 512, 256 px), so every dyadic scale of an 8-scale
    learnlet sees structure (the filters are defined in pixels: scale j ~ 2^(j+1) px); crop / rescale / normalise /
    add noise in the training loader
  * a galaxy-level split (SPLIT below), recorded in index.json -- change it there or here

    python scripts/fire_learnlet_dataset.py [--src DIR] [--workers 4] [--chan-frac 0.01]
    -> data/fire_cii_learnlet/<gal>_s<snap>_los<l>.h5  datasets f1, f2, f4: (nchan, n, n) float32 Jy/pixel (gzip);
       attrs z, inclination_deg, pc_per_px_f1, velocities_kms, ...
       data/fire_cii_learnlet/index.json, m0_preview.npz (moment 0 at f = 4, for the preview grid)
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import astropy.units as u
from astropy.cosmology import LambdaCDM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALS = ["A1", "A2", "A4", "A8", "B1", "B2", "C2"]                      # never C1
SNAPS = [60, 65, 70, 75, 80, 85]
SPLIT = dict(train=["A1", "A2", "A4", "B2", "C2"], val=["A8"], test=["B1"])
COSMO = LambdaCDM(H0=69.7, Om0=0.2821, Ode0=0.7179)
CM_PER_PC = 3.0857e18

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/nas/datasets/FIRE_IFU_cubes/1x_new"); ap.add_argument("--tag", default="1x")
ap.add_argument("--out", default=os.path.join(REPO, "data/fire_cii_learnlet"))
ap.add_argument("--chan-frac", type=float, default=0.01); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--only", help="GAL_SNAP,... (testing)")
A = ap.parse_args()


def block_sum(c, f):
    n = c.shape[1] // f * f
    return c[:, :n, :n].reshape(c.shape[0], n // f, f, n // f, f).sum((2, 4))


def process(job):
    gal, snap = job
    with h5py.File(f"{A.src}/{gal}/{gal}_snap_{snap}_{A.tag}_20s.hdf5", "r") as h:
        z = float(h["header/redshift"][()]); pc = float(h["metadata/pc_per_px"][()])
        v = h["metadata/velocity_array"][()]; los_vec = h["metadata/LoS {unit}"][()]
        cube_all = h["IFU/CII_158mu"][()]                                          # (3, n, n, nv), 4 pi I_nu
    DA = COSMO.angular_diameter_distance(z).to(u.cm).value
    jy = (pc * CM_PER_PC / DA) ** 2 / (4 * np.pi * (1 + z) ** 3 * 1e-23)
    rows, prev = [], {}
    for los in range(cube_all.shape[0]):
        c = (np.moveaxis(cube_all[los], -1, 0) * jy).astype(np.float32)          # (nv, n, n) Jy/pixel
        cf = c.sum((1, 2)); keep = np.where(cf >= A.chan_frac * cf.max())[0]
        c = c[keep]
        name = f"{gal}_s{snap}_los{los}"
        with h5py.File(os.path.join(A.out, name + ".h5.partial"), "w") as o:
            for f in (1, 2, 4):
                b = c if f == 1 else block_sum(c, f)
                o.create_dataset(f"f{f}", data=b, compression="gzip", compression_opts=4, chunks=(1,) + b.shape[1:])
            meta = dict(gal=gal, snap=snap, los=los, z=z, los_vector=los_vec[los].tolist(),
                        inclination_deg=float(np.degrees(np.arccos(abs(los_vec[los][2])))), pc_per_px_f1=pc,
                        kpc_per_arcsec=float(COSMO.kpc_proper_per_arcmin(z).value / 60), dv_kms=float(v[1] - v[0]),
                        velocities_kms=v[keep].tolist(), nchan=int(len(keep)), npix_f1=int(c.shape[1]),
                        flux_jykms=float(c.sum() * (v[1] - v[0])), flux_kept_frac=float(cf[keep].sum() / cf.sum()),
                        peak_jy_per_pix=float(c.max()), split=next(k for k, g in SPLIT.items() if gal in g),
                        units="Jy/pixel (observed, per 20 km/s channel)")
            for k, val in meta.items():
                o.attrs[k] = json.dumps(val) if isinstance(val, list) else val
        os.replace(os.path.join(A.out, name + ".h5.partial"), os.path.join(A.out, name + ".h5"))
        prev[name] = (block_sum(c, 4).sum(0) * meta["dv_kms"]).astype(np.float32)
        rows.append(meta)
    return rows, prev


if __name__ == "__main__":
    os.makedirs(A.out, exist_ok=True)
    jobs = [(g, s) for g in GALS for s in SNAPS]
    if A.only:
        jobs = [j for j in jobs if f"{j[0]}_{j[1]}" in A.only.split(",")]
    index, prev = [], {}
    with ProcessPoolExecutor(A.workers) as ex:
        for rows, p in ex.map(process, jobs):
            index += rows; prev.update(p)
            for m in rows:
                print(f"{m['gal']} s{m['snap']} los{m['los']} [{m['split']}] z={m['z']:.2f} {m['nchan']} channels, "
                      f"{m['npix_f1']} px of {m['pc_per_px_f1']:.1f} pc, flux {m['flux_jykms']:.4f} Jy km/s "
                      f"({100 * m['flux_kept_frac']:.1f}% in the kept channels)", flush=True)
    np.savez_compressed(os.path.join(A.out, "m0_preview.npz"), **prev)
    json.dump(dict(source=A.src, tag=A.tag, line="CII_158mu", chan_frac=A.chan_frac, scales=[1, 2, 4], split=SPLIT,
                   excluded=dict(C1="IFU cubes are a byte copy of C2's (raw import)", snap76="near-duplicate of snap 75"),
                   cubes=index), open(os.path.join(A.out, "index.json"), "w"), indent=1)
    n = {k: sum(m["nchan"] for m in index if m["split"] == k) for k in SPLIT}
    print(f"wrote {len(index)} cubes, channel maps per split {n}, to {A.out}")
