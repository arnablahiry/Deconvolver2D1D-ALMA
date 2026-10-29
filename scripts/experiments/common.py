"""
Shared set-up for the experiments in scripts/experiments/: the deconvolution
problem for a (galaxy, configuration, grid, real/simulated) combination, and the
solvers (primal-dual, primal-dual with FISTA's reweighting reference, FISTA, debiasing).

Recipe common to every solve (same as figures/pd and figures/fista): 2D starlet x
1D CDF 9/7 (5 levels), lambda_b = k x max(real, white) noise MAD per sub-band,
step 1/beta, reweighting (eps = 1) every 2000 iterations from 6000.
"""
import json
import os
import sys
import types

import numpy as np
import torch
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import gpu_pd as G  # noqa: E402

DATA = os.path.join(REPO, "data")
EXP_DATA = os.path.join(DATA, "experiments")
EXP_FIG = os.path.join(REPO, "figures", "experiments")
SIM = os.path.join(EXP_DATA, "01_simulations", "inputs")
RW = dict(start=6000, every=2000, eps=1.0)
MRS_K = 3.0          # significance of the multiresolution support (pdtau-mrs), in sigma
VSYS = {"ngc3110": 5073.9, "ngc2369": 3230.8}
BEAMS = {("ngc3110", "compact"): (1.418, 0.976, 83.5), ("ngc3110", "extended"): (0.220, 0.172, -74.7),
         ("ngc2369", "compact"): (1.169, 1.091, 53.1), ("ngc2369", "extended"): (0.284, 0.236, -87.6)}


def vel(galaxy):
    return VSYS[galaxy] + 25.0 * (np.arange(32) - 15.5)


def read(galaxy, cfg, name, cube_dir="cube"):
    return np.nan_to_num(fits.getdata(os.path.join(DATA, galaxy, cfg, cube_dir, name)).astype(np.float64)).squeeze()


def auto_line_channels(galaxy):
    """Line channels found from the COMPACT dirty cube alone: per-channel std of its large-scale
    (starlet planes >= 4) structure, continuum (median of the outer 3 + 3 channels) removed;
    a channel is 'line' if > 3x the median of the outer channels' values; padded by 1 channel.
    Cached in data/<galaxy>/line_channels.json."""
    path = os.path.join(DATA, galaxy, "line_channels.json")
    if os.path.exists(path):
        return json.load(open(path))["line"]
    d = G.to_t(read(galaxy, "compact", "dirty_cube.fits"))
    outer = [0, 1, 2, 29, 30, 31]
    d = d - d[outer].median(0).values
    pl = G.starlet(d, 7)
    s = np.array([float(pl[4:, c].std()) for c in range(32)])
    ref = np.median(s[outer])
    line = np.where(s > 3 * ref)[0]
    line = sorted(set(int(c) for i in line for c in (i - 1, i, i + 1) if 0 <= c < 32))
    json.dump(dict(line=line, rule="std of starlet planes >= 4 > 3x median of outer 6 channels, padded +-1",
                   per_channel_std_over_outer=[round(float(v / ref), 2) for v in s]), open(path, "w"), indent=1)
    return line


def line_channels(galaxy):
    if galaxy == "ngc3110":
        v = vel(galaxy)
        return [i for i, x in enumerate(v) if 4750 <= x <= 5280]
    return auto_line_channels(galaxy)


def noise_channels(galaxy, cfg, sim=False):
    """ngc3110 extended (real) keeps the window of its reference model (v < 4780 or v > 5180);
    everything else uses all channels outside the line."""
    if galaxy == "ngc3110" and cfg == "extended" and not sim:
        return [i for i, x in enumerate(vel(galaxy)) if x < 4780 or x > 5180]
    line = line_channels(galaxy)
    return [i for i in range(32) if i not in line]


def problem(galaxy="ngc3110", cfg="compact", grid=None, sim=False, k=2.0, mask=None, data=None, target_res=None,
            lam_mode="noise"):
    """grid: 'coarse' (0.04", 2x2 rebinned; compact default), 'fine' (0.02"; extended default),
    'large' (compact 32" field at 0.04", data/<galaxy>/compact/cube_32arcsec).
    mask: support mask path (extended default: the reference model's mask; ngc2369: its grown mask);
    "none" = no support mask, also for the extended configuration.
    lam_mode: "noise" = lambda_b = k x max(real, white) noise MAD of a noise realisation (gpu_pd.noise_lambdas);
    "data_mad" = lambda_b = k x MAD of the dirty cube's own 2D-1D coefficients in sub-band b (no noise model).
    data: optional dirty cube (native grid) replacing the real/simulated one (bootstrap).
    target_res: optional target resolution, arcsec (FWHM of a circular Gaussian G): the sky is y = G x,
    solved for x (gpu_pd.Op target_fwhm_px); to_saved returns y.
    Returns a namespace with op, D, lam, tau, beta, support, cell, bin, crop, line, noise, ..."""
    grid = grid or ("coarse" if cfg == "compact" else "fine")
    P = types.SimpleNamespace(galaxy=galaxy, cfg=cfg, grid=grid, sim=sim, k=k)
    if grid == "large":
        dirty_r, psf = read(galaxy, "compact", "dirty_cube.fits", "cube_32arcsec"), read(galaxy, "compact", "dirty_beam_cube.fits", "cube_32arcsec")
        P.cell, P.bin, P.J, P.crop, P.inner = 0.04, 1, 7, np.s_[200:600, 200:600], np.s_[200:600, 200:600]
    else:
        dirty_r, psf = read(galaxy, cfg, "dirty_cube.fits"), read(galaxy, cfg, "dirty_beam_cube.fits")
        if grid == "coarse":
            dirty_r, psf = G.rebin(dirty_r, psf, 2)
            P.cell, P.bin, P.J, P.crop = 0.04, 2, 7, np.s_[75:325, 75:325]
        else:
            P.cell, P.bin, P.J, P.crop = 0.02, 1, 8, np.s_[150:650, 150:650]
        P.inner = np.s_[:, :]
    dirty = dirty_r
    if sim:
        assert galaxy == "ngc3110" and grid in ("coarse", "fine")
        if cfg == "compact":
            assert grid == "coarse"
            dirty = np.load(os.path.join(SIM, "compact_sim_dirty_0.04arcsec.npy")).astype(np.float64)
            P.truth = os.path.join(SIM, "compact_sim_truth_0.04arcsec.npy")
        else:
            dirty = np.load(os.path.join(SIM, "extended_sim_dirty.npy")).astype(np.float64)
            P.truth = os.path.join(SIM, "extended_sim_truth.npy")
    if data is not None:
        dirty = data
    P.line = line_channels(galaxy)
    P.noise = noise_channels(galaxy, cfg, sim)
    P.target_res = target_res
    P.op = G.Op(psf, dirty, target_fwhm_px=target_res / P.cell if target_res else None)
    P.D = G.Dict2D1D(P.op.d.shape, P.J, 5)
    P.lam, _, _, P.s_pix = G.noise_lambdas(P.D, P.op.smooth(G.to_t(dirty_r)), P.noise, k=k)   # noise always from the real cube (as the solver sees it: G d with a target)
    if lam_mode == "data_mad":
        P.lam_noise = P.lam
        P.lam = k * P.D.band_mad(P.D.W(P.op.d))
    P.beta = G.field_lipschitz(P.op)
    P.tau = 1.0 / P.beta
    P.support = None
    if cfg == "extended" and mask is None:
        mask = (os.path.join(SIM, "extended_sim_support_mask.npy") if sim else
                os.path.join(DATA, galaxy, "extended", "pd", "support_mask.npy"))
    if mask == "none":
        mask = None
    elif mask is not None:                   # any configuration: x = 0 outside the mask, every iteration
        P.support = torch.tensor(np.load(mask), device=G.DEV).to(G.DT)
    P.beam = BEAMS[(galaxy, cfg)]
    return P


def learnt_threshold_lam(P, seed=5, scale=1.0):
    """lambda_b with the learnlet's own thresholds (experiment 22) instead of k x noise on the scales its filter bank
    was trained for. The learnlet thresholds scale j's coefficients (the starlet plane and every filter plane) at
    t_j x sigma, sigma the pixel noise of a WHITE-noise image (threshold_multipliers in dictionaries/<name>.npz); here
    plane b of scale j gets lambda_b = (t_j / s_b) x noise_b, s_b the MAD of plane b for unit white noise, noise_b
    = P.lam / k (the max(real, white) noise of the sub-band). Other scales and the coarse plane keep k x noise_b.
    scale: multiplies the learnt thresholds (the profile's level; they were learnt as one-step hard thresholds).
    Returns (lam, factor per plane)."""
    name = P.D.dict_2d.partition(":")[0]
    thr = np.load(name if name.endswith(".npz") else os.path.join(G.DICT_DIR, name + ".npz"))["threshold_multipliers"]
    w = torch.randn((1, 1024, 1024), device=G.DEV, dtype=G.DT, generator=torch.Generator(G.DEV).manual_seed(seed))
    p = P.D.t2d.forward(w)[:, 0].flatten(1)
    s = ((p - p.median(1, keepdim=True).values).abs().median(1).values / 0.6745).cpu().numpy()
    nt = min(len(thr), P.D.J)                     # never the coarse plane (scale J), even if the file extends further
    fac = np.array([scale * thr[j] / s[b] if j < nt else P.k for b, j in enumerate(P.D.plane_scale)])
    return P.lam / P.k * torch.tensor(fac, device=G.DEV, dtype=G.DT)[:, None, None, None], fac


def learnt_weight_lam(P, w):
    """lambda_b x w on the learnt filter planes A_jk (*) w_j of a filter-bank dictionary, the starlet planes w_j they
    are built on (and every other plane) unchanged. The filter bank contains the starlet planes, so w = 0 is exactly
    the starlet prior (the filter planes then carry no penalty: same minimiser, only a slower step) and w = 1 the full
    learnlet prior, whose 1 + F penalties per learnt scale regularise far harder at the same k (experiment 22's null
    test). Returns (lam, mask of the filter planes)."""
    nA = len(getattr(P.D.t2d, "A", []))
    seen, is_f = set(), []
    for j in P.D.plane_scale:
        is_f.append(j < nA and j in seen)
        seen.add(j)
    f = torch.tensor([w if x else 1.0 for x in is_f], device=G.DEV, dtype=G.DT)
    return P.lam * f.view(-1, *([1] * (P.lam.dim() - 1))), np.array(is_f)


def fista(P, n_iter, lam=None, reweight=RW, x0=None, callback=None, log=print, every=4000):
    lam = P.lam if lam is None else lam
    T0 = P.tau * lam; T = T0
    soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)
    x = torch.zeros_like(P.op.d) if x0 is None else x0.clone(); z = x.clone(); t = 1.0
    for it in range(1, n_iter + 1):
        if reweight and it > reweight["start"] and (it - 1 - reweight["start"]) % reweight["every"] == 0:
            T = T0 / (P.D.W(x).abs() / T0 + reweight["eps"])
        xn = P.D.Winv(soft(P.D.W(z - P.tau * P.op.grad(z)), T)).clamp_(min=0)
        if P.support is not None:
            xn.mul_(P.support)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); z = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn
        if callback is not None:
            callback(it, x, P.op.grad(x) if it in getattr(callback, "marks", ()) else None)
        if it % every == 0:
            log(f"fista iter {it:6d}  flux {float(x.sum()):8.3f}")
    return x


def solve(P, method, n_iter=20000, x0=None, callback=None, log=print):
    """method: pd | pdtau (primal-dual, reweighting reference tau*lambda as FISTA) | fista |
    pdtau-syn (pdtau with the dual variable brought back to image space by the dictionary's synthesis W^-1 instead of
    the adjoint W^T: the starlet's sum of planes, a learnlet's learnt synthesis filters; no known objective, experimental) |
    pdtau-mrs (pdtau with the residual denoised every iteration on the multiresolution support of the
    data: wavelet coefficients with |W d| > MRS_K sigma_b, sigma_b = lambda_b / k; experimental) |
    debias (needs x0: primal-dual from x0 with lambda = 0 on the coefficients detected in x0 and
    the normal lambda elsewhere; no reweighting. "Detected" is judged where lambda is defined, in
    the data: the model's predicted dirty cube N x0 has |W N x0| > lambda_b, i.e. above k x the
    noise of that sub-band -- so scales the array does not measure stay penalized.)"""
    if method == "fista":
        return fista(P, n_iter, x0=x0, callback=callback, log=log)
    if method == "debias":
        free = P.D.W(P.op.apply(x0)).abs() > P.lam
        lam = torch.where(free, torch.zeros_like(P.lam.expand_as(free)), P.lam.expand_as(free))
        log(f"debias: {float(free.float().mean()):.2%} of coefficients detected (lambda -> 0)")
        del free
        x, _, _, _ = G.solve_pd(P.op, P.D, lam, n_iter=n_iter, beta=P.beta, x0=x0, support=P.support,
                                print_every=max(n_iter // 5, 1), log=log, callback=callback)
        return x
    rw = dict(RW) if method == "pd" else dict(RW, ref=P.tau * P.lam)
    gmask = None
    if method == "pdtau-mrs":
        gmask = (P.D.W(P.op.d).abs() > MRS_K * P.lam / P.k).to(G.DT)
        log(f"multiresolution support: {float(gmask.mean()):.2%} of coefficients significant at {MRS_K:g} sigma")
    x, _, _, _ = G.solve_pd(P.op, P.D, P.lam, n_iter=n_iter, beta=P.beta, x0=x0, support=P.support, reweight=rw,
                            print_every=4000, log=log, callback=callback, grad_mask=gmask,
                            back=P.D.Winv if method == "pdtau-syn" else None)
    return x


def to_native(P, x_fine):
    """a 0.02" model (as saved) -> the problem's grid (block sum)"""
    x = np.asarray(x_fine, dtype=np.float64)
    if P.bin > 1:
        b = P.bin
        x = x.reshape(x.shape[0], x.shape[1] // b, b, x.shape[2] // b, b).sum((2, 4))
    return x


def to_saved(P, x):
    """problem grid -> saved form: real-data models at 0.02" (flux-conserving upsample), sims on their grid"""
    if P.target_res:
        x = P.op.to_sky(x if torch.is_tensor(x) else G.to_t(x))
    xm = x.cpu().numpy().astype(np.float64) if torch.is_tensor(x) else x
    if P.bin > 1 and not P.sim:
        xm = G.upsample(xm, P.bin)
    return xm.astype(np.float32)
