#!/usr/bin/env python
"""
End-to-end 2D-1D sparse deconvolution of a real ALMA line cube.

Target: CO(2-1) in IRAS F23007+0836 (NGC 7469), from
`data/calibrated_final.ms.contsub` -- a 3-pointing 12 m-array mosaic,
1,248,030 rows, baselines 14.3-3171 m, synthesized beam ~0.09".

Run `scripts/split_line_ms.py` first to cut the line window out of the parent
MS; everything here works on that small MS.

Stages (`--stage`, repeatable, or `all`)
----------------------------------------
  image      grid the PSF, dirty cube and mosaic primary beam with CASA
  validate   check the operator: PSF normalization, the zero-spacing null
             space, and the fast FFT operator against a real CASA
             degrid/grid major cycle
  deconvolve run FISTA 2D-1D against the validated operator
  export     write model / restored / residual / PB-corrected FITS cubes

Each stage caches to `--outdir`, so `validate` and `deconvolve` can be rerun
without re-gridding.

Why the operator is set up the way it is: see `src/alma_fourier.py`.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alma_fourier import (ImagingConfig, CASAImager, PSFNormalOperator,
                          compare_operators, export_fits, read_casa_image,
                          write_casa_image)
from fista_2d1d import FISTA2D1D, restore, gaussian_beam, debias_on_support


def _npz(outdir, name):
    return os.path.join(outdir, name)


def restoring_beam(psf_image_path):
    """(bmaj, bmin, bpa) in arcsec/deg from the PSF image's fitted beam."""
    from casatools import image as _image

    ia = _image()
    ia.open(psf_image_path)
    try:
        rb = ia.restoringbeam()
    finally:
        ia.close()
    if "beams" in rb:                       # per-plane beams -> use the median
        beams = [b["*0"] for b in rb["beams"].values()]
        bmaj = np.median([b["major"]["value"] for b in beams])
        bmin = np.median([b["minor"]["value"] for b in beams])
        bpa = np.median([b["positionangle"]["value"] for b in beams])
    else:
        bmaj = rb["major"]["value"]
        bmin = rb["minor"]["value"]
        bpa = rb["positionangle"]["value"]
    return float(bmaj), float(bmin), float(bpa)


# --------------------------------------------------------------------------
def stage_image(cfg, outdir):
    t0 = time.time()
    im = CASAImager(cfg)
    psf = im.make_psf()
    dirty = im.dirty()
    try:
        pb = im.make_pb()
    except Exception as exc:
        print(f"[image] primary beam unavailable ({exc}); continuing without it")
        pb = np.ones_like(psf)
    bmaj, bmin, bpa = restoring_beam(im.psf_path)
    im.done()

    np.savez_compressed(_npz(outdir, "grids.npz"), psf=psf, dirty=dirty, pb=pb,
                        beam=np.array([bmaj, bmin, bpa]))
    print(f"[image] psf/dirty/pb cubes {psf.shape} saved "
          f"({time.time() - t0:.1f}s)")
    print(f"[image] restoring beam {bmaj:.4f}\" x {bmin:.4f}\" @ {bpa:.1f} deg "
          f"= {bmaj / cfg.cell_arcsec:.1f} x {bmin / cfg.cell_arcsec:.1f} pixels")
    print(f"[image] dirty rms {dirty.std():.4e} Jy/beam, "
          f"peak {dirty.max():.4e} Jy/beam")
    return psf, dirty, pb


def load_grids(outdir):
    z = np.load(_npz(outdir, "grids.npz"))
    return z["psf"], z["dirty"], z["pb"], z["beam"]


def stage_validate(cfg, outdir, skip_casa=False):
    psf, dirty, pb, beam = load_grids(outdir)
    op = PSFNormalOperator(psf, dirty, pad=cfg.fft_pad)

    report = {"self_test": op.self_test(),
              "zero_spacing": op.zero_spacing_response()}
    print("\n[validate] fast operator self-consistency")
    for k, v in report["self_test"].items():
        print(f"[validate]   {k}: {v}")
    print("[validate] zero-spacing null space")
    for k, v in report["zero_spacing"].items():
        print(f"[validate]   {k}: {v}")

    if not skip_casa:
        print("\n[validate] fast FFT operator vs. a real CASA degrid/grid pass")
        im = CASAImager(cfg)
        im._psf, im._dirty = psf, dirty       # reuse the cached grids
        report["vs_casa"] = compare_operators(im, op)
        im.done()

    with open(_npz(outdir, "validation.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    ok = report["self_test"]["delta_roundtrip_max_rel_err"] < 1e-8
    print(f"\n[validate] delta round trip {'PASS' if ok else 'FAIL'}")
    return report


def stage_deconvolve(cfg, outdir, args):
    psf, dirty, pb, beam = load_grids(outdir)
    op = PSFNormalOperator(psf, dirty, pad=cfg.fft_pad)

    # Whether the coarsest wavelet band is in the operator's null space is a
    # property of this image grid, not an assumption -- ask the operator.
    zc = op.recommend_zero_coarse() if args.zero_coarse is None else args.zero_coarse
    dc = op.zero_spacing_response()
    print(f"[fista] DC response |sum(psf)|/L = {dc['dc_over_lipschitz']:.4f} "
          f"-> coarsest sub-band {'zeroed (unmeasured)' if zc else 'kept (measured)'}")

    solver = FISTA2D1D(num_scales_2d=args.scales_2d,
                       num_scales_1d=args.scales_1d,
                       k_sigma=args.k_sigma,
                       k_extra_finescale=args.k_fine,
                       positivity=not args.allow_negative,
                       zero_coarse=zc,
                       noise_mode=args.noise_mode,
                       reweight=not args.no_reweight,
                       burn_in_iters=args.burn_in,
                       eps=args.reweight_eps)
    t0 = time.time()
    model = solver.deconvolve(op, n_iter=args.n_iter, step_safety=args.step_safety)
    print(f"[fista] {args.n_iter} iterations in {time.time() - t0:.1f}s")

    residual = solver.residual
    print(f"[fista] dirty rms {dirty.std():.4e} -> residual rms "
          f"{residual.std():.4e} Jy/beam "
          f"({dirty.std() / max(residual.std(), 1e-30):.2f}x)")
    print(f"[fista] dirty peak {np.abs(dirty).max():.4e} -> residual peak "
          f"{np.abs(residual).max():.4e} Jy/beam")

    np.savez_compressed(_npz(outdir, "model.npz"), model=model,
                        residual=residual,
                        history=np.array([list(h.values())
                                          for h in solver.history]),
                        history_keys=np.array(list(solver.history[0].keys())))
    return model, residual


def stage_debias(cfg, outdir, args):
    """Refit amplitudes on the FISTA support so the model can be used directly.

    Reports the peak of `model (*) beam` against the dirty peak before and
    after: that ratio is the shrinkage bias, and it should go to ~1.
    """
    from scipy.signal import fftconvolve

    psf, dirty, pb, beam = load_grids(outdir)
    z = np.load(_npz(outdir, "model.npz"))
    model = z["model"]
    op = PSFNormalOperator(psf, dirty, pad=cfg.fft_pad)
    bmaj, bmin, bpa = beam
    cb = gaussian_beam(model.shape[1:], bmaj / cfg.cell_arcsec,
                       bmin / cfg.cell_arcsec, bpa)

    def bias(mod):
        k = int(np.argmax(mod.sum(axis=(1, 2))))
        return fftconvolve(mod[k], cb, mode="same").max() / dirty[k].max()

    before = bias(model)
    debiased = debias_on_support(op, model, n_iter=args.debias_iter,
                                 positivity=not args.allow_negative)
    after = bias(debiased)
    resid = op.residual(debiased)

    print(f"[debias] model(*)beam / dirty peak: {before:.3f} -> {after:.3f}")
    print(f"[debias] flux {model.sum():.2f} -> {debiased.sum():.2f} Jy")
    print(f"[debias] residual rms {op.residual(model).std():.4e} -> "
          f"{resid.std():.4e} Jy/beam")
    print(f"[debias] support unchanged: "
          f"{np.count_nonzero(debiased) <= np.count_nonzero(model)}")

    np.savez_compressed(_npz(outdir, "model.npz"), model=debiased,
                        residual=resid, model_biased=model,
                        history=z["history"], history_keys=z["history_keys"])
    return debiased, resid


def stage_export(cfg, outdir):
    psf, dirty, pb, beam = load_grids(outdir)
    z = np.load(_npz(outdir, "model.npz"))
    model, residual = z["model"], z["residual"]
    bmaj, bmin, bpa = beam

    cb = gaussian_beam(model.shape[1:], bmaj / cfg.cell_arcsec,
                       bmin / cfg.cell_arcsec, bpa)
    # Sanity-check the clean beam against the PSF's main lobe. A wrong
    # position-angle convention shows up here immediately as a large error;
    # the true difference is only the PSF's non-Gaussian shoulders.
    core = psf[psf.shape[0] // 2] > 0.5
    beam_err = np.abs(cb - psf[psf.shape[0] // 2])[core].max()
    print(f"[export] clean beam vs PSF main lobe: max abs diff {beam_err:.4f} "
          f"(over {core.sum()} pixels with psf>0.5)")

    restored = restore(model, residual, cb)

    # PB correction: the flat-noise image divided by the mosaic response.
    pbc = np.where(pb > 0.2, restored / np.maximum(pb, 1e-6), np.nan)

    template = cfg.imagename + ".psf"
    # The model is a sum of delta functions in Jy/pixel and has no beam; the
    # rest are Jy/beam convolved with the restoring beam.
    outputs = {
        "model":    (model, "Jy/pixel", None),
        "restored": (restored, "Jy/beam", (bmaj, bmin, bpa)),
        "residual": (residual, "Jy/beam", (bmaj, bmin, bpa)),
        "dirty":    (dirty, "Jy/beam", (bmaj, bmin, bpa)),
        "pbcor":    (np.nan_to_num(pbc), "Jy/beam", (bmaj, bmin, bpa)),
    }
    for name, (cube, bunit, beam) in outputs.items():
        img = os.path.join(outdir, f"{name}.image")
        write_casa_image(img, cube, template=template, bunit=bunit, beam=beam)
        fits = os.path.join(outdir, f"ngc7469_co21_{name}.fits")
        export_fits(img, fits)
        print(f"[export] {fits}")

    # Integrated line flux over the PB-corrected restored cube, as a sanity
    # number to compare against the literature.
    pix_per_beam = (np.pi * bmaj * bmin / (4 * np.log(2))
                    / cfg.cell_arcsec ** 2)
    # Two numbers, because they mean different things. The model sum is the
    # deconvolved line flux; the restored-cube sum also integrates residual
    # noise over every pixel of the field and is only an upper bound.
    print(f"[export] integrated line flux (model)          "
          f"{model.sum() * cfg.width_kms:8.1f} Jy km/s")
    print(f"[export] integrated over restored cube (upper) "
          f"{np.nansum(pbc) / pix_per_beam * cfg.width_kms:8.1f} Jy km/s "
          f"({pix_per_beam:.1f} pixels/beam)")


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vis", default="data/ngc7469_co21.ms")
    p.add_argument("--outdir", default="results")
    p.add_argument("--stage", action="append", default=None,
                   choices=["image", "validate", "deconvolve", "debias", "export", "all"])

    # Imaging defaults come from ImagingConfig itself so the two cannot drift
    # apart -- they did once, and asking for velocities outside the data
    # silently produced empty channels with a zero PSF.
    D = ImagingConfig(msname="", imagename="")

    g = p.add_argument_group("imaging")
    g.add_argument("--imsize", type=int, default=D.imsize)
    g.add_argument("--cell", type=float, default=D.cell_arcsec, help="arcsec")
    g.add_argument("--nchan", type=int, default=D.nchan)
    g.add_argument("--start-kms", type=float, default=D.start_kms)
    g.add_argument("--width-kms", type=float, default=D.width_kms)
    g.add_argument("--spw", default=D.spw,
                   help="default 0,2 -- the tuning that covers the whole line "
                        "with uniform per-channel sensitivity")
    g.add_argument("--robust", type=float, default=D.robust)
    g.add_argument("--ftmachine", default=D.ftmachine)
    g.add_argument("--phasecenter", default=D.phasecenter,
                   help="pass '' to let CASA pick (which centres on field 0, "
                        "6.3\" off the source for this mosaic)")

    g = p.add_argument_group("deconvolution")
    g.add_argument("--n-iter", type=int, default=60)
    g.add_argument("--scales-2d", type=int, default=4)
    g.add_argument("--scales-1d", type=int, default=2)
    # 4.0, not the textbook 3.0: this is what the validated run in this
    # repo's results/ actually used. A 3.0 threshold measurably overfits on
    # this dataset (see FISTA2D1D/reweight docs) -- keep this in sync with
    # whatever k-sigma the current results/ was produced with.
    g.add_argument("--k-sigma", type=float, default=4.0)
    g.add_argument("--k-fine", type=float, default=1.0)
    g.add_argument("--step-safety", type=float, default=1.0)
    g.add_argument("--no-reweight", action="store_true",
                   help="disable iteratively-reweighted L1 (leaves soft "
                        "thresholding's amplitude bias in place)")
    g.add_argument("--burn-in", type=int, default=None,
                   help="plain-soft iterations before reweighting starts "
                        "(default n_iter//3)")
    g.add_argument("--reweight-eps", type=float, default=1e-2,
                   help="reweighting floor; smaller = sparser")
    g.add_argument("--noise-mode", default="fixed", choices=["fixed", "dynamic"],
                   help="fixed: estimate sub-band noise once from the dirty "
                        "image. dynamic: re-estimate from the shrinking "
                        "residual each iteration (degenerate -- overfits)")
    g.add_argument("--debias-iter", type=int, default=40,
                   help="support-restricted refit iterations removing "
                        "soft-threshold amplitude bias (0 to skip)")
    g.add_argument("--zero-coarse", type=lambda s: s.lower() == "true",
                   default=None,
                   help="force zeroing of the coarsest wavelet band; default "
                        "is decided from the measured DC response of the PSF")
    g.add_argument("--allow-negative", action="store_true",
                   help="drop the x>=0 projection (needed only for absorption)")
    g.add_argument("--skip-casa-check", action="store_true",
                   help="validate without the (slow) CASA major-cycle comparison")

    a = p.parse_args(argv)
    stages = a.stage or ["all"]
    if "all" in stages:
        # NOTE: "debias" (stage_debias, unconstrained least-squares refit of
        # the support) is deliberately NOT in the default chain. It is a
        # different bias-correction technique from the reweighted-L1
        # (`--no-reweight` off by default) already applied inside FISTA, and
        # measured to conflict with it: run on top of an IRL1-converged
        # model on this dataset, it left the amplitude bias unchanged
        # (0.733 -> 0.733) while pushing the residual to 0.88x the
        # source-free noise floor -- i.e. it started fitting noise, not
        # signal. Use `--stage debias` explicitly only when NOT using
        # reweighted L1 (`--no-reweight`), on a support sparse enough that
        # debias_on_support's own docstring warning doesn't apply.
        stages = ["image", "validate", "deconvolve", "export"]

    os.makedirs(a.outdir, exist_ok=True)
    cfg = ImagingConfig(
        msname=a.vis,
        imagename=os.path.join(a.outdir, "casa"),
        imsize=a.imsize, cell_arcsec=a.cell,
        nchan=a.nchan, start_kms=a.start_kms, width_kms=a.width_kms,
        spw=a.spw, robust=a.robust, ftmachine=a.ftmachine,
        phasecenter=a.phasecenter,
    )
    print(f"[cfg] {a.vis} -> {a.outdir}")
    print(f"[cfg] {a.imsize}^2 @ {a.cell}\" x {a.nchan} chan @ {a.width_kms} km/s "
          f"from {a.start_kms} km/s, spw {a.spw}, {a.ftmachine} gridder, "
          f"briggs robust={a.robust}")

    if "image" in stages:
        stage_image(cfg, a.outdir)
    if "validate" in stages:
        stage_validate(cfg, a.outdir, skip_casa=a.skip_casa_check)
    if "deconvolve" in stages:
        stage_deconvolve(cfg, a.outdir, a)
    if "debias" in stages:
        stage_debias(cfg, a.outdir, a)
    if "export" in stages:
        stage_export(cfg, a.outdir)


if __name__ == "__main__":
    main()
