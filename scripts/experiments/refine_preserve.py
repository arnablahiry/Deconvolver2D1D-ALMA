#!/usr/bin/env python
"""
Experiment 21: does 2D1D refine the data (resolve it more finely) while preserving its flux and spectra?

Every test can fail. All on the continuum-subtracted NGC 3110 compact cube (experiment 19, DS), with the GUI's
r002 recipe for 2D1D (flux_audit.py run: pdtau, k = 5, 2x2 rebin, 20 000 iterations) and, as the reference, a
converged multiscale CLEAN on EXACTLY the same inputs (CASA's image-plane `deconvolve`: the same dirty cube and
PSF as 2D1D, no major cycles, so it cannot stall the way tclean did (stop code 4); cleaned to 2 sigma).

  prepare      the input cubes (OUT/ds/<name>/dirty.npy, psf.npy, meta.json):
                 real          the continuum-subtracted dirty cube
                 sim_clumpy    a KNOWN sky with 0.2" structure: the extended configuration's dirty cube of NGC 3110
                               (continuum removed, > 4 sigma, connected), observed with the compact PSF + noise of the
                               measured power spectrum
                 sim_diffuse   the same + a smooth 4" component holding 30 % of the flux
                 null          that noise alone: no sky
                 inj_bright    real + 8 sources (points, Gaussians 0.5-2") and 4 point PAIRS 0.3-1.0" apart (north-south,
                 inj_faint     across the beam's 0.98" minor axis), peak S/N 30 / 8 per clean beam
                 jack_eb1/2    each execution block imaged alone (flux_audit.py jackknife)
  run NAME --device cuda:N      2D1D (flux_audit.py run)
  clean NAME [--nomask]         image-plane multiscale CLEAN -> OUT/clean/<NAME>[_nomask]/{model,residual,restored,mask}.npy
  analyze      -> OUT/report.md, OUT/results.json, FIG/*.png

    DS=/scratch/alahiry/contsub/ds_contsub python scripts/experiments/refine_preserve.py prepare
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gui"))
sys.path.insert(0, os.path.join(REPO, "simple"))
DS = os.environ.get("DS") or "/scratch/alahiry/contsub/ds_contsub"
OUT = os.environ.get("OUT") or os.path.join(REPO, "data/experiments/21_refine_preserve")
FIG = os.environ.get("FIG") or os.path.join(REPO, "figures/experiments/21_refine_preserve")
EXT = os.path.join(REPO, "data/ngc3110/extended/cube")
FA = os.path.join(REPO, "scripts/experiments/flux_audit.py")
PY = sys.executable

META = json.load(open(os.path.join(DS, "meta.json")))
CELL, DV = float(META["cell_arcsec"]), abs(float(META["width_kms"]))
NZ, NY, NX = int(META["nchan"]), int(META["imsize"]), int(META["imsize"])
VEL = float(META["start_kms"]) + float(META["width_kms"]) * np.arange(NZ)
BEAM = [float(v) for v in META["beam"]]
BPX = 1.1331 * BEAM[0] * BEAM[1] / CELL ** 2                     # clean beam area, px
FREE = json.load(open(os.path.join(REPO, "data/experiments/19_contsub/line_free.json")))["channels"]
LINE = [c for c in range(NZ) if c not in FREE]
SIMS = ["sim_clumpy", "sim_diffuse"]
INJ = {"inj_bright": 30.0, "inj_faint": 8.0}
CUBES = ["real", *SIMS, "null", *INJ, "jack_eb1", "jack_eb2"]
# injected sources: (name, dx", dy" (east, north), FWHM", line centre km/s); pairs: two points sep" apart north-south
SOURCES = [("off_pt", -6.0, 5.5, 0.12, 4900.0), ("off_0.5", 6.0, 5.5, 0.5, 5000.0), ("off_1.5", -6.0, -5.5, 1.5, 5100.0),
           ("off_2", 6.2, -6.2, 2.0, 4950.0), ("off_pt_b", 0.0, 6.8, 0.12, 5050.0), ("off_1.5_b", 6.8, 1.5, 1.5, 4850.0),
           ("on_pt", 1.5, 2.0, 0.12, 5000.0), ("on_pt_b", -2.0, -1.5, 0.12, 5100.0)]
PAIRS = [("pair_0.3", -6.9, 0.0, 0.3, 4950.0), ("pair_0.5", 3.0, -6.9, 0.5, 5050.0),
         ("pair_0.7", -3.2, -6.9, 0.7, 4900.0), ("pair_1.0", -3.0, 6.6, 1.0, 5000.0)]
LINE_SIG = 40.0
CLEAN_SCALES = [0, 24, 71, 213]                                 # px: point, 1/3, 1 and 3 clean beams (CASA deconvolve)
CLEAN_SCALES_GPU = [0, 24, 71]                                  # px, FWHM of the Gaussian components (gclean)


def p(*a):
    print(*a, flush=True)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def pix(dx, dy):
    return NX // 2 - dx / CELL, NY // 2 + dy / CELL


def robust_rms(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    med = np.median(a, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(a - med), axis=axis)


def noise_sigma(dirty):
    """noise per channel of a dirty cube, Jy/beam: the line-free channels, their mean removed"""
    lf = np.asarray(dirty[FREE], dtype=np.float64)
    return float((lf - lf.mean(0)).std() * np.sqrt(len(FREE) / (len(FREE) - 1)))


# ============================================================== prepare
def image_contsub(cube):
    """continuum removed in the image plane: a straight line in velocity fitted to each pixel's line-free channels"""
    v = VEL[FREE] - VEL[FREE].mean()
    A = np.vstack([np.ones_like(v), v]).T
    coef = np.linalg.lstsq(A, cube[FREE].reshape(len(FREE), -1), rcond=None)[0]
    return cube - (np.vstack([np.ones(NZ), VEL - VEL[FREE].mean()]).T @ coef).reshape(cube.shape)


def ext_beam():
    from astropy.io import fits
    h = fits.getheader(os.path.join(EXT, "restored_own_beam.fits"))
    return [h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]]


def truth_skies():
    """(clumpy, diffuse) skies in Jy/pixel. clumpy: the multiscale CLEAN model of the extended configuration
    (exttruth: only what CLEAN found significant; no sidelobes, no plateau), smoothed to the extended beam
    (0.22" x 0.17") and kept in the line channels: a realistic KNOWN sky with 0.2" structure, not the true galaxy
    (the extended array misses the large scales). diffuse: + a 4" smooth version of it holding 30 % of the total flux."""
    from scipy import ndimage
    from restore import convolve_with_beam
    eb = ext_beam()
    ebpx = 1.1331 * eb[0] * eb[1] / CELL ** 2
    model = np.load(os.path.join(OUT, "clean", "ext", "model.npy")).astype(np.float64)
    clumpy = np.clip(convolve_with_beam(model, CELL, *eb), 0, None) / ebpx      # Jy/beam -> Jy/pixel, flux kept
    clumpy[FREE] = 0.0
    s = 4.0 / 2.3548 / CELL
    smooth = ndimage.gaussian_filter(clumpy, (0, s, s))
    diffuse = clumpy + smooth * (0.3 / 0.7) * clumpy.sum() / smooth.sum()
    return clumpy, diffuse, dict(ext_beam=eb, ext_beam_px=ebpx, source="multiscale CLEAN model of the extended cube * its beam",
                                 ext_clean_flux=float(model.sum() * DV))


def source_cube(snr, sigma):
    """sky (Jy/pixel) of the injected sources and pairs, and their truth (fluxes in Jy km/s)"""
    sky, truth = np.zeros((NZ, NY, NX)), []
    yy, xx = np.mgrid[0:NY, 0:NX]

    def add(name, dx, dy, fwhm, v0, kind, **extra):
        x0, y0 = pix(dx, dy)
        s = max(fwhm, 0.5 * CELL) / CELL / 2.3548
        img = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / s ** 2)
        img /= img.sum()
        g = np.exp(-0.5 * ((VEL - v0) / LINE_SIG) ** 2)
        dil = np.sqrt((BEAM[0] ** 2 + fwhm ** 2) * (BEAM[1] ** 2 + fwhm ** 2)) / (BEAM[0] * BEAM[1])
        spec = snr * sigma * dil * g / g.max()
        sky[:] += spec[:, None, None] * img[None]
        truth.append(dict(name=name, kind=kind, dx=dx, dy=dy, fwhm=fwhm, v0=v0, flux=float(spec.sum() * DV), **extra))

    for name, dx, dy, fwhm, v0 in SOURCES:
        add(name, dx, dy, fwhm, v0, "source")
    for name, dx, dy, sep, v0 in PAIRS:
        for k, sgn in (("a", 1), ("b", -1)):
            add(f"{name}{k}", dx, dy + sgn * sep / 2, 0.02, v0, "pair_member", pair=name, sep=sep, cx=dx, cy=dy)
    return sky, truth


def write(name, cube, extra=None, psf=None):
    d = os.path.join(OUT, "ds", name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "dirty.npy"), cube.astype(np.float32))
    if not os.path.exists(os.path.join(d, "psf.npy")):
        os.symlink(psf or os.path.join(DS, "psf.npy"), os.path.join(d, "psf.npy"))
    jdump(dict(META, experiment=name, **(extra or {})), os.path.join(d, "meta.json"))
    p(f"  {name}: line-free rms {cube[FREE].std():.3e} Jy/beam, line-channel sum {cube[LINE].sum() / BPX * DV:.1f} Jy km/s")


def prepare(a):
    import torch
    import gpu_pd as G
    from uv_to_image import ShiftInvariantOperator
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(REPO, "data/experiments/19_contsub/line_free.json"), os.path.join(OUT, "line_free.json"))
    dirty = np.load(os.path.join(DS, "dirty.npy")).astype(np.float64)
    psf = np.load(os.path.join(DS, "psf.npy")).astype(np.float64)
    sigma = noise_sigma(dirty)
    p(f"noise {sigma * 1e3:.3f} mJy/beam per channel; clean beam {BEAM[0]:.3f}\" x {BEAM[1]:.3f}\" ({BPX:.0f} px)")
    op = ShiftInvariantOperator(psf, dirty)
    d_t = torch.tensor(dirty, device=G.DEV, dtype=G.DT)
    noise = G.noise_with_ps(G.measured_ps(G.noise_planes(d_t, FREE)), tuple(dirty.shape), seed=2121).cpu().numpy().astype(np.float64)
    if not a.sims_only:
        write("real", dirty)
        write("null", noise)
    if not os.path.exists(os.path.join(OUT, "clean", "ext", "model.npy")):
        p("no extended CLEAN model yet (exttruth): simulations skipped")
        return
    clumpy, diffuse, info = truth_skies()
    p(f"truth: extended beam {info['ext_beam'][0]:.3f}\" x {info['ext_beam'][1]:.3f}\"; clumpy {clumpy.sum() * DV:.1f}, diffuse {diffuse.sum() * DV:.1f} Jy km/s")
    for name, sky in (("sim_clumpy", clumpy), ("sim_diffuse", diffuse)):
        write(name, op.apply(sky) + noise, dict(truth_info=info, truth_flux=float(sky.sum() * DV)))
        np.save(os.path.join(OUT, "ds", name, "truth.npy"), sky.astype(np.float32))
    if a.sims_only:
        return
    for name, snr in INJ.items():
        sky, truth = source_cube(snr, sigma)
        write(name, dirty + op.apply(sky), dict(injected=truth))
        np.save(os.path.join(OUT, "ds", name, "injected_sky.npy"), sky.astype(np.float32))
    if not a.skip_jack:
        env = dict(os.environ, DS=DS, OUT=OUT, FIG=FIG)
        subprocess.run([PY, FA, "jackknife"], env=env, check=True, cwd=REPO)


# ================================================================ CLEAN
def _to_casa(arr, shape):
    """(nz, ny, nx) -> CASA's axis order (x, y, [stokes,] freq) as in `shape`"""
    a = np.transpose(arr, (2, 1, 0))
    return a.reshape(shape) if a.shape != tuple(shape) else a


def _from_casa(a):
    a = np.squeeze(a)
    return np.transpose(a, (2, 1, 0)) if a.ndim == 3 else a.T[None]


def image_clean(dirty, psf, beam, w, scales, nomask=False, nsig=2.0):
    """image-plane multiscale CLEAN (CASA deconvolve) of `dirty` (nz, ny, nx; Jy/beam) with `psf` (its centre cut to
    the image size), to nsig x the line-free noise, in folder w; auto-multithresh mask unless nomask.
    Saves and returns model (Jy/pixel), residual, restored = model * beam (no residual) and the mask."""
    from casatasks import deconvolve
    from casatools import image
    from restore import convolve_with_beam
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    here = os.getcwd()
    os.chdir(w)                                        # CASA logs land here
    py, px = np.unravel_index(np.argmax(psf[psf.shape[0] // 2]), psf.shape[1:])
    psf = psf[:, py - NY // 2: py + NY // 2, px - NX // 2: px + NX // 2]
    sigma = noise_sigma(dirty)
    base = os.path.join(w, "img")
    ia = image()
    ia.fromfits(outfile=f"{base}.tmpl", infile=os.path.join(DS, "casa", "dirty_cube.fits"), overwrite=True)
    order = "".join(str(i) for i in np.argsort([{"Right Ascension": 0, "Declination": 1, "Stokes": 2, "Frequency": 3}[n]
                                                 for n in ia.coordsys().names()]))
    ia.close()
    for ext, arr in (("residual", dirty), ("psf", psf)):   # CASA's synthesis images: RA, Dec, Stokes, Frequency
        ia.open(f"{base}.tmpl")
        im2 = ia.transpose(outfile=f"{base}.{ext}", order=order)
        ia.close()
        im2.putchunk(_to_casa(arr, im2.shape()))
        im2.setbrightnessunit("Jy/beam")
        im2.close()
    shutil.rmtree(f"{base}.tmpl")
    t0 = time.time()
    kw = dict(usemask="auto-multithresh", sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5,
              negativethreshold=7.0, minbeamfrac=0.3, growiterations=75) if not nomask else dict(usemask="user", mask="")
    rec = deconvolve(imagename=base, deconvolver="multiscale", scales=scales, niter=10_000_000, gain=0.1,
                     threshold=f"{nsig * sigma}Jy", restoration=False, **kw)
    out = {}
    for ext in ("model", "residual", "mask"):
        if os.path.isdir(f"{base}.{ext}"):
            ia.open(f"{base}.{ext}")
            out[ext] = _from_casa(ia.getchunk()).astype(np.float64)
            ia.close()
    out["restored"] = convolve_with_beam(out["model"], CELL, *beam)
    for k, v in out.items():
        np.save(os.path.join(w, f"{k}.npy"), v.astype(np.uint8 if k == "mask" else np.float32))
    for f in glob.glob(f"{base}.*"):
        shutil.rmtree(f, ignore_errors=True)
    rl = out["residual"][LINE]
    summ = dict(seconds=time.time() - t0, sigma=sigma, threshold=nsig * sigma, scales=scales, mask=not nomask, beam=beam,
                flux=float(out["model"].sum() * DV), residual_rms_line=float(rl.std()), residual_rms_free=float(out["residual"][FREE].std()),
                casa=rec if isinstance(rec, (dict, list, str, int, float, type(None))) else str(rec))
    jdump(summ, os.path.join(w, "summary.json"))
    os.chdir(here)
    return out, summ


def clean(a):
    """image-plane multiscale CLEAN of cube a.name (its own PSF and beam) -> OUT/clean/<name>[_nomask]"""
    tag = a.name + ("_nomask" if a.nomask else "")
    d = os.path.join(OUT, "ds", a.name)
    beam = [float(v) for v in json.load(open(os.path.join(d, "meta.json")))["beam"]]
    out, summ = image_clean(np.load(os.path.join(d, "dirty.npy")).astype(np.float64), np.load(os.path.join(d, "psf.npy")).astype(np.float64),
                            beam, os.path.join(OUT, "clean", tag), CLEAN_SCALES, nomask=a.nomask)
    p(f"[clean {tag}] {summ['seconds']:.0f} s; model {summ['flux']:.1f} Jy km/s; residual rms / noise: line "
      f"{summ['residual_rms_line'] / summ['sigma']:.2f}, free {summ['residual_rms_free'] / summ['sigma']:.2f}")


def exttruth(a):
    """CLEAN the extended configuration's dirty cube (continuum removed in the image plane) with its own PSF:
    scales 0, 1/2, 1, 3 extended beams -> OUT/clean/ext; its model is the simulations' truth (truth_skies)"""
    from deconvgui.worker import _read_fits_cube
    ed = image_contsub(_read_fits_cube(os.path.join(EXT, "dirty_cube.fits")).astype(np.float64))
    ep = _read_fits_cube(os.path.join(EXT, "dirty_beam_cube.fits")).astype(np.float64)
    eb = ext_beam()
    b = round(eb[0] / CELL)
    out, summ = image_clean(ed, ep, eb, os.path.join(OUT, "clean", "ext"), [0, b // 2, b, 3 * b], nsig=2.0)
    p(f"[ext] {summ['seconds']:.0f} s; model {summ['flux']:.1f} Jy km/s; residual rms / noise: line "
      f"{summ['residual_rms_line'] / summ['sigma']:.2f}, free {summ['residual_rms_free'] / summ['sigma']:.2f}")


def ms_clean_gpu(dirty, psf, scales, device="cuda:0", gain=0.1, nsig=2.0, ssbias=0.6, max_iter=400_000, positive=False, log=print):
    """Multiscale CLEAN (Cornwell 2008) per channel on the GPU, with the EXACT full PSF (2x the image, so every
    shift is covered: no PSF patch, no major cycles needed, no divergence) and no mask, to nsig x the line-free
    noise. Components: Gaussians of FWHM `scales` px (0 = a point), unit total flux. Each iteration picks, over
    scales and positions, the largest |residual smoothed by that scale| x the scale bias (1 - ssbias s / s_max),
    and subtracts gain x the least-squares amplitude of that component there. positive: positive components only
    (peaks, not |peaks|; stops when no positive peak is above the threshold), like 2D1D's positivity. Returns the model (Jy/pixel) and residual (Jy/beam), (nz, ny, nx)."""
    import torch
    dev = torch.device(device)
    nz, ny, nx = dirty.shape
    sigma = noise_sigma(dirty)
    thr = nsig * sigma
    py, px = np.unravel_index(np.argmax(psf[nz // 2]), psf.shape[1:])
    P = torch.tensor(psf[:, py - ny: py + ny, px - nx: px + nx], dtype=torch.float64, device=dev)      # (nz, 2ny, 2nx), peak at (ny, nx)
    fy = torch.fft.fftfreq(2 * ny, device=dev, dtype=torch.float64)[:, None]
    fx = torch.fft.rfftfreq(2 * nx, device=dev, dtype=torch.float64)[None, :]
    gft = [torch.ones_like(fy * fx) if sc == 0 else torch.exp(-2 * np.pi ** 2 * (sc / 2.3548) ** 2 * (fx ** 2 + fy ** 2)) for sc in scales]
    smax = max(scales) or 1
    bias = [1 - ssbias * sc / smax for sc in scales]
    S = len(scales)
    model = np.zeros((nz, ny, nx))
    resid = np.zeros((nz, ny, nx))
    t0 = time.time()
    for c in range(nz):
        Pf = torch.fft.rfft2(torch.fft.ifftshift(P[c]))                                  # PSF centred at (0, 0)
        # K[t][s] = B_t * P * B_s, centred: the change of residual_t per unit amplitude of a scale-s component
        K = [[torch.fft.fftshift(torch.fft.irfft2(Pf * gft[t] * gft[s], s=(2 * ny, 2 * nx))) for s in range(S)] for t in range(S)]
        norm = [float(K[s][s][ny, nx]) for s in range(S)]
        R0 = torch.tensor(dirty[c], dtype=torch.float64, device=dev)
        rf = torch.fft.rfft2(R0, s=(2 * ny, 2 * nx))
        Rs = [torch.fft.irfft2(rf * gft[s], s=(2 * ny, 2 * nx))[:ny, :nx] for s in range(S)]
        comps = torch.zeros((S, ny, nx), dtype=torch.float64, device=dev)
        # a scale-s component only where its Gaussian lies inside the field (1.5 FWHM from the edges): outside it the
        # PSF on its periodic grid is indefinite (its sum is negative) and the descent would run away there
        allow = []
        for sc in scales:
            m_ = int(np.ceil(1.5 * sc))
            w_ = torch.zeros((ny, nx), dtype=torch.float64, device=dev)
            w_[m_: ny - m_, m_: nx - m_] = 1.0
            allow.append(w_)
        Pfull = torch.fft.rfft2(torch.fft.ifftshift(P[c]))

        def exact_residual():   # d - P * (sum_s B_s * comps_s), linear convolution on the doubled grid
            x = torch.zeros((ny, nx), dtype=torch.float64, device=dev)
            for s in range(S):
                x += torch.fft.irfft2(torch.fft.rfft2(comps[s], s=(2 * ny, 2 * nx)) * gft[s], s=(2 * ny, 2 * nx))[:ny, :nx]
            px_ = torch.fft.irfft2(torch.fft.rfft2(x, s=(2 * ny, 2 * nx)) * Pfull, s=(2 * ny, 2 * nx))
            return R0 - px_[:ny, :nx]

        it = 0
        while it < max_iter:
            if it and it % 5000 == 0:   # a "major cycle": the exact residual, so rounding never accumulates
                rf = torch.fft.rfft2(exact_residual(), s=(2 * ny, 2 * nx))
                Rs = [torch.fft.irfft2(rf * gft[s], s=(2 * ny, 2 * nx))[:ny, :nx] for s in range(S)]
            best, bs, by, bx = 0.0, 0, 0, 0
            for s in range(S):   # the scale: the largest |smoothed residual| x its bias
                a = (Rs[s] * bias[s] * allow[s]).contiguous()
                i = int(torch.argmax(a if positive else a.abs()))
                v = float(a.view(-1)[i])
                if abs(v) > abs(best):
                    best, bs, by, bx = v, s, i // nx, i % nx
            if float(Rs[0].max() if positive else Rs[0].abs().max()) < thr or (positive and best <= 0):
                break
            amp = gain * float(Rs[bs][by, bx]) / norm[bs]       # its amplitude: the least-squares one for that component, x gain
            comps[bs, by, bx] += amp
            for t in range(S):
                Rs[t] -= amp * K[t][bs][ny - by: 2 * ny - by, nx - bx: 2 * nx - bx]
            it += 1
        m = torch.zeros((ny, nx), dtype=torch.float64, device=dev)
        for s in range(S):
            if float(comps[s].abs().sum()) == 0:
                continue
            cf = torch.fft.rfft2(comps[s], s=(2 * ny, 2 * nx))
            m += torch.fft.irfft2(cf * gft[s], s=(2 * ny, 2 * nx))[:ny, :nx]
        model[c] = m.cpu().numpy()
        resid[c] = Rs[0].cpu().numpy()
        log(f"  chan {c:2d}: {it:6d} iterations, model {model[c].sum():.4f} Jy, peak |residual| {np.abs(resid[c]).max() / sigma:.2f} sigma "
            f"({time.time() - t0:.0f} s)")
    return model, resid, dict(sigma=sigma, threshold=thr, scales=scales, gain=gain, smallscalebias=ssbias, positive=positive)


def gclean(a):
    """GPU multiscale CLEAN of cube a.name (its own PSF and beam), no mask, to 2 sigma -> OUT/gclean/<name>
    (--positive: positive components only -> OUT/gclean_pos/<name>)"""
    from restore import convolve_with_beam
    d, w = os.path.join(OUT, "ds", a.name), os.path.join(OUT, "gclean_pos" if a.positive else "gclean", a.name)
    os.makedirs(w, exist_ok=True)
    meta = json.load(open(os.path.join(d, "meta.json")))
    beam = [float(v) for v in meta["beam"]]
    dirty = np.load(os.path.join(d, "dirty.npy")).astype(np.float64)
    t0 = time.time()
    model, resid, info = ms_clean_gpu(dirty, np.load(os.path.join(d, "psf.npy")).astype(np.float64), CLEAN_SCALES_GPU, device=a.device,
                                      positive=a.positive, log=lambda m: p(f"[gclean{'+' if a.positive else ''} {a.name}]{m}"))
    out = dict(model=model, residual=resid, restored=convolve_with_beam(model, CELL, *beam))
    for k, v in out.items():
        np.save(os.path.join(w, f"{k}.npy"), v.astype(np.float32))
    summ = dict(info, seconds=time.time() - t0, beam=beam, flux=float(model.sum() * DV),
                residual_rms_line=float(resid[LINE].std() / info["sigma"]), residual_rms_free=float(resid[FREE].std() / info["sigma"]))
    jdump(summ, os.path.join(w, "summary.json"))
    p(f"[gclean {a.name}] {summ['seconds']:.0f} s; model {summ['flux']:.1f} Jy km/s; residual rms / noise: line {summ['residual_rms_line']:.2f}, "
      f"free {summ['residual_rms_free']:.2f}")


# ================================================================== run
def run(a):
    env = dict(os.environ, DS=DS, OUT=OUT, FIG=FIG)
    subprocess.run([PY, FA, "run", a.name, "--device", a.device] + (["--tag", a.tag] if a.tag else []) +
                   (["--extra", a.extra] if a.extra else []), env=env, check=True, cwd=REPO)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare"); s.add_argument("--skip-jack", action="store_true")
    s.add_argument("--sims-only", action="store_true", help="only the simulations (after exttruth)")
    s = sub.add_parser("run"); s.add_argument("name"); s.add_argument("--device", default="cuda:0")
    s.add_argument("--tag"); s.add_argument("--extra")
    s = sub.add_parser("clean"); s.add_argument("name"); s.add_argument("--nomask", action="store_true")
    sub.add_parser("exttruth")
    s = sub.add_parser("gclean"); s.add_argument("name"); s.add_argument("--device", default="cuda:0")
    s.add_argument("--positive", action="store_true")
    sub.add_parser("analyze")
    a = ap.parse_args()
    if a.cmd == "analyze":
        from refine_preserve_analyze import analyze
        analyze()
    else:
        {"prepare": prepare, "run": run, "clean": clean, "exttruth": exttruth, "gclean": gclean}[a.cmd](a)
