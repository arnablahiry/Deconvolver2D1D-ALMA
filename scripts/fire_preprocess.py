#!/usr/bin/env python
"""
FIRE [CII] 158 um IFU cubes (/nas/datasets/FIRE_IFU_cubes/4x: 8 galaxies x 7 snapshots x 3 lines of sight, 20 kpc box,
256 px of 78 pc, -1000..1000 km/s) -> ALMA-like sky cubes on a common angular grid, for training / testing the 2D-1D
deconvolution.

Per galaxy, snapshot and line of sight:
  1. flux: the cubes are 4 pi I_nu, intrinsic rest frame (erg s^-1 cm^-2 Hz^-1); observed flux per pixel
     S = (4 pi I_nu / 4 pi) / (1 + z)^3 x Omega_pix -> Jy/pixel  (cosmology of scripts/build_fire_sim_obs_notebook.py)
  2. pixels: physical -> angular at the snapshot's redshift (angular diameter distance), then exact area-weighted
     (flux-conserving) resampling onto CELL arcsec pixels -- the same angular grid at every redshift, so a galaxy
     shrinks on the sky with z
  3. crop: NPIX x NPIX centred on the flux-weighted centroid of the [CII] moment 0 (--center box: on the box centre)
     (zero outside the simulated box); channels: the NCHAN native 20 km/s channels from -940 km/s, or with --dv, NCHAN
     channels of width DV (flux-conserving overlap resampling, flux density averaged) centred on the cube's
     flux-weighted mean velocity (--v0 <km/s>: on that velocity instead)
  4. beam: an elliptical Gaussian clean beam drawn per cube (reproducible seed), FWHM log-uniform in BEAM_RANGE,
     axis ratio uniform in [0.6, 1], PA uniform -> the convolved cube in Jy/beam

Snapshot 76 (z = 4.53) is skipped: it is 3 Myr-ish from snapshot 75 (z = 4.58), not a separate epoch. C1 is skipped:
its IFU cubes are a byte-for-byte copy of C2's, already in the raw import
(/nas/homes/alahiry/fire_gals_new_import/C1-snapnum*-20kpc-compressed.hdf5) (their mass maps and headers differ; the cubes correlate with C2's gas
map, 0.64-0.78, not C1's, 0.47-0.60) -- regenerate C1's from C1-snapnum*-20kpc-compressed.hdf5.

    python scripts/fire_preprocess.py [--src DIR --tag 1x --out data/fire_cii_1x] [--cell 0.02] [--npix 128] [--nchan 96] [--beam-min 0.1] [--beam-max 0.8] [--workers 8]
        [--dv 25 --v0 centroid|<km/s>] [--center centroid|box]
    e.g. the real sample's spectral grid (32 x 25 km/s) on 400 px:
      --src /nas/datasets/FIRE_IFU_cubes/1x --tag 1x --npix 400 --nchan 32 --dv 25 --cell 0.02   --out data/fire_cii_1x_n400_c0.02
      --src /nas/datasets/FIRE_IFU_cubes/1x --tag 1x --npix 400 --nchan 32 --dv 25 --cell 0.0081 --center box --out data/fire_cii_1x_n400_c0.0081
      (0.0081" x 400 = 3.24": the 20 kpc box at z = 5.43, the largest on the sky, just fits)
    or physical pixels (--cell-pc: the same pc per pixel at every redshift, so the angular cell varies; at the native
    pixel, 19.5 pc for the 1x cubes, the crop snaps to whole native pixels and copies them exactly, no resampling):
      --src /nas/datasets/FIRE_IFU_cubes/1x --tag 1x --npix 400 --nchan 32 --dv 25 --cell-pc 19.53125 --center peak --out data/fire_cii_1x_n400_19.5pc
    -> data/fire_cii/<gal>_s<snap>_los<l>.npz (sky: Jy/pixel, conv: Jy/beam, (nchan, npix, npix); v, z, beam, ...)
       data/fire_cii/index.json
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
from scipy.ndimage import gaussian_filter
import numpy as np
import astropy.units as u
from astropy.cosmology import LambdaCDM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

GALS = ["A1", "A2", "A4", "A8", "B1", "B2", "C2"]        # C1: broken copy of C2 (see above)
SNAPS = [60, 65, 70, 75, 80, 85]                          # 76: near-duplicate of 75
COSMO = LambdaCDM(H0=69.7, Om0=0.2821, Ode0=0.7179)
CM_PER_PC = 3.0857e18

ap = argparse.ArgumentParser()
ap.add_argument("--cell", type=float, default=0.02, help="arcsec")
ap.add_argument("--npix", type=int, default=128)
ap.add_argument("--nchan", type=int, default=96)
ap.add_argument("--beam-min", type=float, default=0.1); ap.add_argument("--beam-max", type=float, default=0.8)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--dv", type=float, default=None, help="output channel width, km/s (default: native, no resampling)")
ap.add_argument("--v0", default="centroid", help="with --dv: window centre, km/s, or 'centroid' (flux-weighted mean velocity)")
ap.add_argument("--cell-pc", type=float, default=None, help="pixel in pc (overrides --cell, which then varies with z)")
ap.add_argument("--center", default="centroid", choices=["centroid", "box", "peak"],
                help="spatial crop centre: flux-weighted centroid, box centre, or the peak of moment 0 smoothed by 0.5 kpc "
                     "(the brightest component: for crops smaller than the box, where the centroid of two separated "
                     "components can fall between them)")
ap.add_argument("--src", default="/nas/datasets/FIRE_IFU_cubes/4x", help="tree of <gal>/<gal>_snap_<n>_<tag>_20s.hdf5")
ap.add_argument("--tag", default="4x", help="spatial rebinning tag in the file names (4x, 1x, ...)")
ap.add_argument("--out", default=os.path.join(REPO, "data/fire_cii"))
A = ap.parse_args()
SRC, OUT = A.src, A.out


def overlap_matrix(n_new, cell, c0, n_old, a):
    """W[j, i]: fraction of old pixel i (size a, centred at (i - (n_old-1)/2) a) inside new pixel j
    (size cell, centred at c0 + (j - (n_new-1)/2) cell); out = W @ in conserves flux"""
    lo_n = c0 + (np.arange(n_new) - n_new / 2) * cell
    lo_o = (np.arange(n_old) - n_old / 2) * a
    ov = np.minimum(lo_n[:, None] + cell, lo_o[None] + a) - np.maximum(lo_n[:, None], lo_o[None])
    return np.clip(ov, 0, None) / a


def beam_for(gal, snap, los):
    rng = np.random.default_rng(int(hashlib.sha1(f"{gal}-{snap}-{los}".encode()).hexdigest()[:8], 16))
    bmaj = float(np.exp(rng.uniform(np.log(A.beam_min), np.log(A.beam_max))))
    return bmaj, bmaj * float(rng.uniform(0.6, 1.0)), float(rng.uniform(0.0, 180.0))


def process(job):
    gal, snap = job
    with h5py.File(f"{SRC}/{gal}/{gal}_snap_{snap}_{A.tag}_20s.hdf5", "r") as h:
        z = float(h["header/redshift"][()]); pc = float(h["metadata/pc_per_px"][()])
        v = h["metadata/velocity_array"][()]; los_vec = h["metadata/LoS {unit}"][()]
        cube_all = h["IFU/CII_158mu"][()]                                         # (3, ny, nx, nv), 4 pi I_nu
    DA = COSMO.angular_diameter_distance(z).to(u.cm).value
    a = pc * CM_PER_PC / DA * 206264.806                                          # native pixel, arcsec
    jy = (pc * CM_PER_PC / DA) ** 2 / (4 * np.pi * (1 + z) ** 3 * 1e-23)          # -> Jy per native pixel
    dv = float(v[1] - v[0])
    if A.dv is None:
        i0 = int(np.argmin(np.abs(v - (-940.0)))); vk = v[i0:i0 + A.nchan]
    out = []
    for los in range(cube_all.shape[0]):
        c = np.moveaxis(cube_all[los], -1, 0) * jy                                 # (nv, ny, nx) Jy/pixel
        n_old = c.shape[1]
        m0 = c.sum(0)
        if A.center == "box":
            cy = cx = 0.0
        elif A.center == "peak":
            iy, ix = np.unravel_index(np.argmax(gaussian_filter(m0, 500 / pc / 2.3548)), m0.shape)
            cy, cx = (iy - (n_old - 1) / 2) * a, (ix - (n_old - 1) / 2) * a
        else:
            yy, xx = np.mgrid[:n_old, :n_old] - (n_old - 1) / 2
            cy, cx = float((m0 * yy).sum() / m0.sum() * a), float((m0 * xx).sum() / m0.sum() * a)   # centroid, arcsec
        cell = A.cell if A.cell_pc is None else A.cell_pc / pc * a                # arcsec
        if A.cell_pc is not None and abs(A.cell_pc - pc) < 1e-6 * pc:             # native pixels: whole-pixel shift only
            cy, cx = round(cy / a) * a + (A.npix - n_old) % 2 * a / 2, round(cx / a) * a + (A.npix - n_old) % 2 * a / 2
        Wy = overlap_matrix(A.npix, cell, cy, n_old, a); Wx = overlap_matrix(A.npix, cell, cx, n_old, a)
        if A.dv is None:
            cs = c[i0:i0 + A.nchan]
        else:                                                                     # resample the velocity axis
            f = c.sum((1, 2))
            vc = float((f * v).sum() / f.sum()) if A.v0 == "centroid" else float(A.v0)
            vk = vc + (np.arange(A.nchan) - (A.nchan - 1) / 2) * A.dv
            Wv = overlap_matrix(A.nchan, A.dv, vc - (v[0] + v[-1]) / 2, len(v), dv) * dv / A.dv   # averages flux density
            cs = np.einsum("kv,vyx->kyx", Wv, c)
        sky = (Wy @ cs @ Wx.T).astype(np.float32)        # Jy/pixel
        bmaj, bmin, bpa = beam_for(gal, snap, los)
        conv = convolve_with_beam(sky, cell, bmaj, bmin, bpa).astype(np.float32)               # Jy/beam
        tot = float(c.sum() * dv)
        meta = dict(gal=gal, snap=snap, los=los, z=z, los_vector=los_vec[los].tolist(),
                    inclination_deg=float(np.degrees(np.arccos(abs(los_vec[los][2])))),
                    cell_arcsec=cell, cell_pc=cell * pc / a, native_pix_arcsec=a, kpc_per_arcsec=pc / 1e3 / a, dv_kms=A.dv or dv,
                    v_center_kms=float(vk.mean()), box_arcsec=n_old * a,
                    centroid_offset_arcsec=[cy, cx], beam=[bmaj, bmin, bpa],
                    flux_total_jykms=tot, flux_kept_frac=float(sky.sum() * (A.dv or dv) / tot))
        np.savez_compressed(os.path.join(OUT, f"{gal}_s{snap}_los{los}.npz"), sky=sky, conv=conv, v=vk.astype(np.float32),
                            meta=json.dumps(meta))
        out.append(meta)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(g, s) for g in GALS for s in SNAPS]
    index = []
    with ProcessPoolExecutor(A.workers) as ex:
        for res in ex.map(process, jobs):
            for m in res:
                index.append(m)
                print(f"{m['gal']} s{m['snap']} los{m['los']} z={m['z']:.2f} i={m['inclination_deg']:.0f} "
                      f"beam {m['beam'][0]:.2f}x{m['beam'][1]:.2f}\" flux {m['flux_total_jykms']:.4f} Jy km/s, "
                      f"kept {100 * m['flux_kept_frac']:.1f}%", flush=True)
    json.dump(dict(settings=vars(A), source=SRC, line="CII_158mu", cubes=index), open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print("wrote", len(index), "cubes to", OUT)
