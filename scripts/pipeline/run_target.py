#!/usr/bin/env python
"""
The NGC 3110 compact-vs-extended pipeline (figures/compact_vs_msclean_complete) for one target of the sample
(scripts/sample/image_co21_briggs.py TARGETS, data/sample/<target>/ metadata), into results/<target>/:

  config            config.json: grid, channels, phase centre, fields, spws, lambda/B_max of the extended array
  image             {compact,extended}/cube/{dirty_cube,dirty_beam_cube}.fits: natural weighting, NCHAN x WIDTH km/s
                    centred on v_sys (radio, LSRK), PSF 2x the field (simple/grid_with_casa.tclean_dirty_cube)
  deconvolve        compact/deconvolved/model.npy: 2D-1D primal-dual, tau-lambda reweighting, k = 5, 20000 iterations,
                    0.04"-binned grid as for NGC 3110 (scripts/experiments/common.py "pdtau"), saved on the native grid;
                    line channels detected from the compact dirty cube (line_channels.json)
  msclean_compact   compact/msclean/: tclean multiscale, scales 0, 1, 3, 6 beams, nsigma 3, auto-multithresh with the
                    ALMA pipeline's short-baseline values (as the NGC 3110 compact run)
  msclean_extended  extended/msclean/: the same with the long-baseline values
  finish_extended   extended/msclean_finished/: the extended run continued inside the growing astrodendro mask
                    (scripts/msclean_complete.finish; scales 0, 1, 3, 6, 12 beams, nsigma 1.5), mask_final.npy
  figures           figures/40-50 (scripts/compact_vs_msclean_dendro/make_figures.py, fig50_msclean_complete.py)

Grid: ~10 pixels across the extended beam's minor axis (0.15 x the sample's Briggs minor axis, >= 0.02"), the field
16" as for NGC 3110 unless the emission is larger (FIELD_ARCSEC). Mosaics are imaged from their central pointing
only (the 2D-1D operator is a single-pointing convolution). Each finished step leaves results/<target>/.done/<step>;
steps already done are skipped unless --force.

    /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/pipeline/run_target.py NGC2369 [--steps ...] [--force]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts/sample", "scripts", "simple", "scripts/experiments"):
    sys.path.insert(0, os.path.join(REPO, p))
RESULTS = os.path.join(REPO, "results")
STEPS = ["config", "image", "deconvolve", "msclean_compact", "msclean_extended", "finish_extended", "figures"]
C_KMS, RESTFREQ_GHZ, NCHAN, WIDTH_KMS = 299792.458, 230.538, 32, 25.0
# field of view: NGC 3110's 16" holds its emission (90% of its compact CLEAN flux within 9.9"); the others scaled by
# the same radius of their compact sample cubes, rounded up to 2" (16" where that is smaller)
FIELD_ARCSEC = {"IC4518E": 18, "NGC7130": 18, "IC5179": 20, "MCG-02-33-098": 20, "NGC3256": 24}
AUTOMASK = dict(compact=dict(sidelobethreshold=2.0, noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0,
                             minbeamfrac=0.3, growiterations=75),
                extended=dict(sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=7.0,
                              minbeamfrac=0.3, growiterations=75))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(T):
    return json.load(open(os.path.join(RESULTS, T, "config.json")))


def save(T, cfg):
    json.dump(cfg, open(os.path.join(RESULTS, T, "config.json"), "w"), indent=1)


# ======================================================================== steps
def step_config(T):
    import casatools
    from image_co21_briggs import TARGETS, ms_paths, science_fields, pointings, max_uv_m
    g = TARGETS[T]; data = g.get("data", T)
    meta = {c: json.load(open(os.path.join(REPO, "data/sample", T, c, "co21_cube/meta.json"))) for c in ("compact", "extended")}
    vsys = float(meta["extended"]["vsys_kms"])
    f_line = RESTFREQ_GHZ / (1 + vsys / C_KMS)
    mosaic = bool(meta["extended"]["mosaic"] or meta["compact"]["mosaic"])
    md, qa = casatools.msmetadata(), casatools.quanta()
    md.open(ms_paths(data, "extended")[0])
    ids = science_fields(md, g["fields"]); pts = pointings(md, ids)
    md.close()
    if mosaic:                                    # the pointing nearest the mosaic's centre
        ra_c, dec_c = np.angle(np.exp(1j * pts[:, 0]).mean()), pts[:, 1].mean()
        k = int(np.argmin(np.hypot((pts[:, 0] - ra_c) * np.cos(dec_c), pts[:, 1] - dec_c)))
        ra0, dec0 = pts[k]
        phasecenter = f"J2000 {qa.time(qa.quantity(ra0, 'rad'), prec=11)[0]} {qa.angle(qa.quantity(dec0, 'rad'), prec=10)[0]}"
    else:
        phasecenter = meta["extended"]["phasecenter"]
    configs, bmax = {}, 0.0
    for c in ("compact", "extended"):
        ms_list = list(meta[c]["ms"]); spw = list(meta[c]["spw"])
        fields = []
        for ms in ms_list:
            md.open(ms)
            fid = science_fields(md, g["fields"])
            if mosaic:
                p = pointings(md, fid)
                d = np.hypot((p[:, 0] - ra0) * np.cos(dec0), p[:, 1] - dec0) * 206264.806
                fid = [f for f, dd in zip(fid, d) if dd < 2.0]
                if not fid:
                    raise RuntimeError(f"{T} {c}: no pointing of {ms} within 2\" of the central one")
                fields.append(",".join(map(str, fid)))
            else:
                fields.append(",".join(sorted(set(md.namesforfields(fid)))))
            md.close()
            if c == "extended":
                bmax = max(bmax, max_uv_m(ms, fid))
        configs[c] = dict(ms=ms_list, spw=spw, field=fields)
    bmin_briggs = float(meta["extended"]["beam_arcsec"][1])
    cell = max(0.02, round(0.15 * bmin_briggs / 0.005) * 0.005)
    fov = FIELD_ARCSEC.get(T, 16)
    imsize = int(round(fov / cell / 2)) * 2
    lbmax = (C_KMS * 1e3 / (f_line * 1e9)) / bmax * 206264.806
    cfg = dict(name=T, label=g["label"], vsys_kms=vsys, f_line_ghz=f_line, start_kms=vsys - (NCHAN - 1) / 2 * WIDTH_KMS,
               width_kms=WIDTH_KMS, nchan=NCHAN, restfreq_ghz=RESTFREQ_GHZ, cell_arcsec=cell, imsize=imsize,
               field_arcsec=imsize * cell, phasecenter=phasecenter, mosaic=mosaic,
               note="mosaic: central pointing only" if mosaic else "", bmax_m=bmax, lbmax_arcsec=lbmax, configs=configs)
    save(T, cfg)
    log(f"config: v_sys {vsys:.1f} km/s, cell {cell}\", {imsize} px ({imsize * cell:.1f}\"), lambda/B_max {lbmax:.4f}\" "
        f"(B_max {bmax:.1f} m){'; MOSAIC: central pointing only' if mosaic else ''}")


def step_image(T):
    from grid_with_casa import tclean_dirty_cube
    cfg = load(T)
    beams = {}
    for c in ("compact", "extended"):
        cc = cfg["configs"][c]
        field = cc["field"] if len(set(cc["field"])) > 1 else cc["field"][0]
        out = tclean_dirty_cube(ms_list=cc["ms"], spw_list=cc["spw"], field=field, phasecenter=cfg["phasecenter"],
                                out_dir=os.path.join(RESULTS, T, c, "cube"), imsize=cfg["imsize"], cell_arcsec=cfg["cell_arcsec"],
                                start_kms=cfg["start_kms"], width_kms=cfg["width_kms"], nchan=cfg["nchan"],
                                restfreq_ghz=cfg["restfreq_ghz"], weighting="natural", log_name=f"pipeline_{T}_{c}_dirty.log")
        beams[c] = [float(v) for v in out["beam"]]
        log(f"image {c}: beam {beams[c][0]:.3f}\" x {beams[c][1]:.3f}\" @ {beams[c][2]:.1f} deg")
    cfg["beams_natural"] = beams
    save(T, cfg)


def step_deconvolve(T):
    import common
    common.DATA = RESULTS                         # common.read: RESULTS/<T>/<cfg>/cube/<name>
    for c, b in load(T)["beams_natural"].items():  # common.problem attaches the target's beams from its table
        common.BEAMS[(T, c)] = tuple(b)
    P = common.problem(T, "compact", k=5.0)
    log(f"deconvolve: line channels {P.line} (auto, results/{T}/line_channels.json), {len(P.noise)} noise channels; "
        f"grid {P.op.d.shape}")
    t0 = time.time()
    x = common.solve(P, "pdtau", n_iter=20000, log=lambda m: log("  " + m))
    out = os.path.join(RESULTS, T, "compact", "deconvolved"); os.makedirs(out, exist_ok=True)
    xs = common.to_saved(P, x)
    np.save(os.path.join(out, "model.npy"), xs)
    r = P.op.residual_sky(x)
    json.dump(dict(method="pdtau", k=5.0, n_iter=20000, grid="0.5 x native cell (2x2 binned)", line=P.line, noise=P.noise,
                   flux_line_Jy_kms=float(xs[P.line].sum() * WIDTH_KMS), resid_rms_line=float(r[P.line].std()),
                   seconds=round(time.time() - t0)), open(os.path.join(out, "run.json"), "w"), indent=1)
    log(f"deconvolve: line flux {xs[P.line].sum() * WIDTH_KMS:.1f} Jy km/s, {time.time() - t0:.0f} s")


def tclean_common(T, c, scales_beams):
    cfg = load(T); cc = cfg["configs"][c]
    b = cfg["beams_natural"][c]
    bpx = max(5, int(round(np.sqrt(b[0] * b[1]) / cfg["cell_arcsec"] / 5) * 5))
    field = cc["field"] if len(set(cc["field"])) > 1 else cc["field"][0]
    return dict(vis=cc["ms"], field=field, spw=cc["spw"], datacolumn="data", imsize=cfg["imsize"],
                cell=f"{cfg['cell_arcsec']}arcsec", phasecenter=cfg["phasecenter"], specmode="cube",
                start=f"{cfg['start_kms']}km/s", width=f"{cfg['width_kms']}km/s", nchan=cfg["nchan"],
                restfreq=f"{cfg['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard",
                weighting="natural", deconvolver="multiscale", scales=[int(s * bpx) for s in scales_beams],
                restoringbeam="common", pbcor=False, fullsummary=True)


def step_msclean(T, c):
    import glob
    import shutil
    from casa_logs import redirect_casa_logs
    from casatasks import tclean, exportfits
    redirect_casa_logs(f"pipeline_{T}_{c}_msclean.log")
    out = os.path.join(RESULTS, T, c, "msclean"); os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "_tclean")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    kw = tclean_common(T, c, (0, 1, 3, 6))
    log(f"msclean {c}: scales {kw['scales']} px, nsigma 3, auto-multithresh {AUTOMASK[c]}")
    t0 = time.time()
    rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", **AUTOMASK[c], **kw)
    for ext in ("image", "model", "residual", "psf", "mask"):
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(out, f"{ext}.fits"), velocity=True, overwrite=True)
    json.dump(dict(stopcode=rec.get("stopcode"), iterations=rec.get("iterdone"), major_cycles=rec.get("nmajordone"),
                   scales_px=kw["scales"], automask=AUTOMASK[c], seconds=round(time.time() - t0)),
              open(os.path.join(out, "run.json"), "w"), indent=1)
    log(f"msclean {c}: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {time.time() - t0:.0f} s")


def step_finish(T):
    from casa_logs import redirect_casa_logs
    from msclean_complete import finish
    redirect_casa_logs(f"pipeline_{T}_extended_finish.log")
    cfg = load(T)
    line = json.load(open(os.path.join(RESULTS, T, "line_channels.json")))["line"]
    kw = tclean_common(T, "extended", (0, 1, 3, 6, 12))
    finish(os.path.join(RESULTS, T, "extended", "msclean"), os.path.join(RESULTS, T, "extended", "msclean_finished"),
           kw, line, cfg["cell_arcsec"], label=f"{T} extended")


def step_figures(T):
    env = dict(os.environ, GAL_DIR=os.path.join(RESULTS, T))
    for mode in ("extended", "lbmax"):                # figures/{extended_clean_beam,largest_compact_baseline}_comparison/
        subprocess.run([sys.executable, os.path.join(REPO, "scripts/compact_vs_msclean_dendro/make_figures.py")],
                       env=dict(env, COMPARE=mode), check=True)


RUN = dict(config=step_config, image=step_image, deconvolve=step_deconvolve,
           msclean_compact=lambda T: step_msclean(T, "compact"), msclean_extended=lambda T: step_msclean(T, "extended"),
           finish_extended=step_finish, figures=step_figures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target"); ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    done = os.path.join(RESULTS, a.target, ".done"); os.makedirs(done, exist_ok=True)
    for s in [s for s in STEPS if s in a.steps]:
        if os.path.exists(os.path.join(done, s)) and not a.force:
            log(f"{a.target}: {s} already done"); continue
        log(f"{a.target}: {s} ...")
        t0 = time.time()
        RUN[s](a.target)
        open(os.path.join(done, s), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"{a.target}: {s} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
