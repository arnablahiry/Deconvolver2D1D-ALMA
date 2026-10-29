#!/usr/bin/env python
"""
Is a major cycle needed? For a converged model x, compare the fast image-plane model prediction the 2D-1D solver uses,
N x = PSF (*) x (gpu_pd.Op, full PSF imaged at 2x the field), with the exact one from the visibilities: CASA degrids x,
grids A^H W A x with the real (u, v) and weights -- one tclean major cycle.

  tclean niter=0, no model         -> D  = A^H W V           (CASA's dirty cube on this grid)
  tclean niter=0, startmodel = x   -> Rt = A^H W (V - A x)   (the TRUE residual)
  exact prediction  Nt x = D - Rt;   fast prediction  Nf x = Op.apply(x);   fast residual Rf = d - Nf x (d: the solver's
  dirty cube)

Reported, over the line channels (whole field and inside the finished extended-CLEAN mask), in Jy/beam and in units of
the line-free noise sigma: rms(Nt x - Nf x) (operator error), rms(D - d) (dirty-cube difference), rms(Rt - Rf).

    /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/major_cycle_check.py [GAL_DIR]
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results/NGC3110")
OUT = os.path.join(R, "compact", "major_cycle_check"); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
F = [i for i in range(GC["nchan"]) if i not in L]


def casa_images(x):
    """D and Rt from two tclean niter=0 calls on the compact visibilities, on the solver's grid"""
    import casatools
    from casatasks import tclean
    from casa_logs import redirect_casa_logs
    redirect_casa_logs(f"major_cycle_check_{GC['name']}.log")
    cc = GC["configs"]["compact"]
    field = cc["field"] if len(set(cc["field"])) > 1 else cc["field"][0]
    kw = dict(vis=cc["ms"], field=field, spw=cc["spw"], datacolumn="data", imsize=GC["imsize"],
              cell=f"{GC['cell_arcsec']}arcsec", phasecenter=GC["phasecenter"], specmode="cube",
              start=f"{GC['start_kms']}km/s", width=f"{GC['width_kms']}km/s", nchan=GC["nchan"],
              restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard",
              weighting="natural", niter=0, restoration=False, pbcor=False)
    ia = casatools.image()
    get = lambda p: np.asarray((ia.open(p), ia.getchunk(), ia.close())[1], dtype=np.float64)[:, :, 0, :].transpose(2, 1, 0)
    base = os.path.join(OUT, "_dirty")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    t0 = time.time(); tclean(imagename=base, **kw)
    D = get(base + ".residual"); print(f"dirty: {time.time() - t0:.0f} s", flush=True)
    # the model as a CASA image on the same grid (coordinates of the dirty run), Jy/pixel
    ia.open(base + ".residual"); cs = ia.coordsys(); ia.close()
    mpath = os.path.join(OUT, "model.image")
    if os.path.exists(mpath):
        shutil.rmtree(mpath)
    ia.fromarray(outfile=mpath, pixels=x.transpose(2, 1, 0)[:, :, None, :].astype(np.float32), csys=cs.torecord(),
                 overwrite=True)
    ia.setbrightnessunit("Jy/pixel"); ia.close()
    base2 = os.path.join(OUT, "_resid")
    for p in glob.glob(base2 + ".*"):
        shutil.rmtree(p)
    t0 = time.time(); tclean(imagename=base2, startmodel=mpath, **kw)
    Rt = get(base2 + ".residual"); print(f"residual with the model: {time.time() - t0:.0f} s", flush=True)
    return D, Rt


x = np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64)   # Jy/pixel on the solver's 0.02" grid
cache = os.path.join(OUT, "casa_DRt.npz")
if os.path.exists(cache):
    z = np.load(cache); D, Rt = z["D"], z["Rt"]
else:
    D, Rt = casa_images(x)
    np.savez_compressed(cache, D=D.astype(np.float32), Rt=Rt.astype(np.float32))

from astropy.io import fits
import gpu_pd as G
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
d = rd(os.path.join(R, "compact/cube/dirty_cube.fits")); psf = rd(os.path.join(R, "compact/cube/dirty_beam_cube.fits"))
op = G.Op(psf, d)
Nf = op.apply(G.to_t(x)).cpu().numpy().astype(np.float64)
Nt = D - Rt
Rf = d - Nf
M = np.load(os.path.join(R, "extended/msclean_finished/mask_final.npy")).astype(bool)
sig = float(np.std(Rt[F]))                                                 # line-free noise of the true residual
rms = lambda a, sel=None: float(np.sqrt(np.mean((a[L] if sel is None else a[L][sel]) ** 2)))
MV = M[L]
out = dict(sigma_line_free_mJy=1e3 * sig, peak_Nt_mJy=1e3 * float(Nt[L].max()))
for name, a in (("operator: Nt x - Nf x", Nt - Nf), ("dirty: D - d", D - d), ("residual: Rt - Rf", Rt - Rf)):
    out[name] = dict(rms_field_mJy=1e3 * rms(a), rms_field_sigma=rms(a) / sig, rms_mask_mJy=1e3 * rms(a, MV),
                     rms_mask_sigma=rms(a, MV) / sig, max_abs_mJy=1e3 * float(np.abs(a[L]).max()),
                     rel_to_Nx_pct=100 * rms(a) / rms(Nt))
out["true residual rms (line, field) mJy"] = 1e3 * rms(Rt)
out["fast residual rms (line, field) mJy"] = 1e3 * rms(Rf)
json.dump(out, open(os.path.join(OUT, "major_cycle_check.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()})
