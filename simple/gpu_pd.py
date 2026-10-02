"""
GPU (PyTorch) implementation of the 2D-1D primal-dual deconvolution: the same
problem as deconvolve_pd.py, fast enough to run to convergence (tens of
thousands of iterations) on a full 800 x 800 x 32 cube.

======================================================================
THE PROBLEM
======================================================================
    minimize_x   1/2 x^T N x - d^T x  +  sum_b lambda_b ||(W x)_b||_1  +  lambda_D sum(x)
    subject to   x >= 0                  [ + optional support: x = 0 outside a mask ]

  x        sky model, Jy/pixel, (nz, ny, nx)
  d        dirty cube (natural weighting), Jy/beam
  N        PSF convolution, i.e. A^H W A for a single pointing (uv_to_image.py);
           the PSF must be imaged at >= 2x the field, see Op
  W        2D starlet (J scales, per channel) x 1D CDF 9/7 (L levels, along
           velocity), analysis transform; b indexes (spatial scale, spectral band)
  lambda_b k x max(real correlated-noise MAD, white-noise MAD) of sub-band b
           (noise_lambdas)
  lambda_D optional pixel-domain (Dirac) L1; with x >= 0 it is lambda_D sum(x)

======================================================================
THE ITERATION (Condat 2013; Vu 2013)
======================================================================
    x+ = P( x - tau (N x - d + lambda_D + W^T u) )          P: x >= 0 [, x = 0 off-mask]
    u+ = clip( u + sigma W (2 x+ - x), -lambda, +lambda )
tau = 1 / beta (beta = largest eigenvalue of N on the field), sigma = beta / (2 ||W||^2).
Every piece is exact (clip = prox of the L1 conjugate, P = projection), so the
iteration converges to the minimizer above -- unlike analyze / soft-threshold /
synthesize / clip, which is not the proximal operator of anything for the
redundant starlet frame (see deconvolve_pd.py).

Optional reweighted L1 (Candes, Wakin & Boyd 2008): at the scheduled
iterations lambda -> lambda / (|W x| / lambda + eps), which removes most of the
L1 amplitude bias on coefficients that are clearly signal.

Array conventions match the rest of simple/: cubes (nz, ny, nx), row = Dec,
col = RA pixel, PSF peak-normalized per channel and centred on its peak pixel.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32
_B3 = torch.tensor([1., 4., 6., 4., 1.]) / 16.
_ALPHA, _BETA, _GAMMA, _DELTA, _ZETA = (-1.586134342059924, -0.052980118572961,
                                        0.882911075530934, 0.443506852043971, 1.230174104914001)


def to_t(a):
    return torch.as_tensor(np.asarray(a), device=DEV, dtype=DT)


# ======================================================================
# measurement operator
# ======================================================================
class Op:
    """N x = PSF (*) x per channel, as an FFT on a grid of max(2 x field, PSF)
    so the convolution restricted to the field is linear (not circular) and
    uses every PSF lag up to +-(field size): with a PSF only as large as the
    field, lags beyond half the field are missing and the truncation rings the
    transfer function negative, making N indefinite."""

    def __init__(self, psf, dirty):
        nz, self.ny, self.nx = dirty.shape
        psf = np.asarray(psf, dtype=np.float64)
        psf = psf / psf.max(axis=(1, 2), keepdims=True)
        py, px = np.unravel_index(np.argmax(psf[0]), psf[0].shape)
        self.gy, self.gx = max(psf.shape[1], 2 * self.ny), max(psf.shape[2], 2 * self.nx)
        big = np.zeros((nz, self.gy, self.gx))
        big[:, :psf.shape[1], :psf.shape[2]] = psf
        big = np.roll(big, (-py, -px), axis=(1, 2))
        self.otf = torch.fft.rfft2(torch.tensor(big, device=DEV, dtype=torch.float64)).to(torch.complex64)
        self.d = to_t(dirty)
        self.bound = float(self.otf.abs().max())          # rigorous (loose) Lipschitz bound

    def apply(self, x):
        big = torch.zeros((x.shape[0], self.gy, self.gx), device=DEV, dtype=DT)
        big[:, :self.ny, :self.nx] = x
        return torch.fft.irfft2(torch.fft.rfft2(big) * self.otf, s=(self.gy, self.gx))[:, :self.ny, :self.nx]

    def grad(self, x):
        """gradient of the data term: N x - d"""
        return self.apply(x) - self.d


def field_lipschitz(op, n=40, margin=1.15, seed=0):
    """largest eigenvalue of N restricted to the field (power iteration x margin),
    capped at the rigorous bound max|OTF|"""
    v = torch.randn(op.d.shape, device=DEV, dtype=DT, generator=torch.Generator(DEV).manual_seed(seed))
    lam = 0.0
    for _ in range(n):
        w = op.apply(v)
        lam = float(w.norm() / v.norm())
        v = w / w.norm()
    return min(lam * margin, op.bound)


# ======================================================================
# 2D starlet x 1D CDF 9/7
# ======================================================================
def _smooth(x, step):
    k = _B3.to(DEV, DT)
    y = x[:, None]
    y = F.conv2d(F.pad(y, (2 * step, 2 * step, 0, 0), mode="reflect"), k.view(1, 1, 1, 5), dilation=(1, step))
    y = F.conv2d(F.pad(y, (0, 0, 2 * step, 2 * step), mode="reflect"), k.view(1, 1, 5, 1), dilation=(step, 1))
    return y[:, 0]


def starlet(x, J):
    """(nz, ny, nx) -> (J + 1, nz, ny, nx): J detail planes + coarse; planes sum to x"""
    planes, c = [], x
    for j in range(J):
        s = _smooth(c, 2 ** j)
        planes.append(c - s)
        c = s
    planes.append(c)
    return torch.stack(planes)


def _p1(e):
    return e + torch.cat([e[:, 1:], e[:, -1:]], 1)


def _p0(o):
    return o + torch.cat([o[:, :1], o[:, :-1]], 1)


def _cdf_fwd1(x):
    e, o = x[:, 0::2], x[:, 1::2]
    o = o + _ALPHA * _p1(e); e = e + _BETA * _p0(o)
    o = o + _GAMMA * _p1(e); e = e + _DELTA * _p0(o)
    return e / _ZETA, o * _ZETA


def _cdf_inv1(a, d):
    e, o = a * _ZETA, d / _ZETA
    e = e - _DELTA * _p0(o); o = o - _GAMMA * _p1(e)
    e = e - _BETA * _p0(o); o = o - _ALPHA * _p1(e)
    x = torch.stack([e, o], 2)
    return x.reshape(x.shape[0], -1, *x.shape[3:])


def cdf_fwd(x, L):
    """CDF 9/7 along dim 1 (length a multiple of 2**L), packed as
    [approx_L, detail_L, detail_{L-1}, ..., detail_1]"""
    dets, a = [], x
    for _ in range(L):
        a, d = _cdf_fwd1(a)
        dets.append(d)
    return torch.cat([a] + dets[::-1], 1)


def cdf_inv(c, L):
    n = c.shape[1]
    a, pos = c[:, : n >> L], n >> L
    for lev in range(L, 0, -1):
        m = n >> lev
        a = _cdf_inv1(a, c[:, pos:pos + m])
        pos += m
    return a


def band_slices(nz, L):
    out, pos = [("a", slice(0, nz >> L))], nz >> L
    for lev in range(L, 0, -1):
        out.append((f"d{lev}", slice(pos, pos + (nz >> lev))))
        pos += nz >> lev
    return out


class Dict2D1D:
    """W x: (J + 1, nz, ny, nx) tensor, axis 1 = packed CDF 9/7 bands of each starlet plane"""

    def __init__(self, shape, J, L):
        self.J, self.L, self.nz = J, L, shape[0]
        self.bands = band_slices(self.nz, L)
        self._vjp = None

    def W(self, x):
        return cdf_fwd(starlet(x, self.J), self.L)

    def Winv(self, c):
        return cdf_inv(c, self.L).sum(0)

    def WT(self, u):
        # W is linear, so its vjp at any point is W^T: build it once, reuse it
        if self._vjp is None:
            self._vjp = torch.func.vjp(self.W, torch.zeros(u.shape[1:], device=DEV, dtype=DT))[1]
        return self._vjp(u)[0]

    def band_mad(self, c):
        """robust sigma per (spatial scale, spectral band), shaped (J + 1, nz, 1, 1) like W x"""
        sig = torch.zeros((c.shape[0], self.nz, 1, 1), device=DEV, dtype=DT)
        for j in range(c.shape[0]):
            for _, sl in self.bands:
                b = c[j, sl].flatten()
                if b.numel() > 2 ** 24:
                    b = b[torch.randperm(b.numel(), device=DEV)[: 2 ** 24]]
                sig[j, sl] = (b - b.median()).abs().median() / 0.6745
        return sig

    def norm_sq(self, shape, n=60):
        """||W||^2 by power iteration (x 1.1): converges slowly from below"""
        side = min(shape[1], shape[2], 4 * 2 ** (self.J - 1) + 9)
        sub = Dict2D1D((shape[0], side, side), self.J, self.L)
        v = torch.randn((shape[0], side, side), device=DEV, dtype=DT)
        v /= v.norm()
        lam = 0.0
        for _ in range(n):
            w = sub.WT(sub.W(v))
            lam = float(w.norm())
            v = w / lam
        return lam * 1.1


# ======================================================================
# noise model and lambda_b
# ======================================================================
def noise_planes(dirty_t, noise_ch):
    """line-free channels with their per-pixel mean (continuum) removed, rescaled
    to undo the variance lost to that mean removal"""
    n = dirty_t[noise_ch] - dirty_t[noise_ch].mean(0, keepdim=True)
    return n * np.sqrt(len(noise_ch) / (len(noise_ch) - 1))


def measured_ps(planes):
    """mean 2D power spectrum (rfft2 layout) of (k, ny, nx) noise planes"""
    return (torch.fft.rfft2(planes).abs() ** 2).mean(0) / (planes.shape[1] * planes.shape[2])


def noise_with_ps(ps, shape, seed=0):
    """Gaussian noise cube, channels independent, each with the 2D power spectrum ps"""
    w = torch.randn(shape, device=DEV, dtype=DT, generator=torch.Generator(DEV).manual_seed(seed))
    return torch.fft.irfft2(torch.fft.rfft2(w) * ps.sqrt(), s=shape[1:])


def noise_lambdas(D, dirty_t, noise_ch, k, seed=1):
    """lambda_b = k x max(sigma_real, sigma_white) per sub-band. Returns
    (lam, sig_real, sig_white, pixel_sigma)."""
    npl = noise_planes(dirty_t, noise_ch)
    s_pix = float(npl.std())
    sig_real = D.band_mad(D.W(noise_with_ps(measured_ps(npl), dirty_t.shape, seed=seed)))
    white = torch.randn(dirty_t.shape, device=DEV, dtype=DT, generator=torch.Generator(DEV).manual_seed(seed + 100)) * s_pix
    sig_white = D.band_mad(D.W(white))
    return k * torch.maximum(sig_real, sig_white), sig_real, sig_white, s_pix


# ======================================================================
# solver
# ======================================================================
def solve_pd(op, D, lam, n_iter=1000, beta=None, x0=None, u0=None, dirac=0.0, support=None,
             reweight=None, snaps=(), print_every=500, log=print, callback=None):
    """Condat-Vu iteration of the module docstring.
    dirac: lambda_D (Jy/beam units of the gradient), 0 = off
    support: optional 0/1 tensor like x (x = 0 outside)
    reweight: None or dict(start, every, eps[, ref]): lambda -> lambda / (|W x| / ref + eps), ref defaults to
              lambda itself; ref = tau * lambda (the threshold in model units) is FISTA's convention
    snaps: iterations at which to keep a copy of x (numpy)
    callback: optional f(iteration, x, gradient) called after every iteration (e.g. to grab frames)
    Returns (x, u, history, snapshots)."""
    beta = beta or field_lipschitz(op)
    tau, sig = 1.0 / beta, beta / (2.0 * D.norm_sq(op.d.shape))
    x = torch.zeros_like(op.d) if x0 is None else x0.clone()
    u = torch.zeros_like(D.W(x)) if u0 is None else u0.clone()
    bound = lam
    hist, kept = [], {}
    for it in range(n_iter):
        g = op.grad(x)
        xn = (x - tau * (g + dirac + D.WT(u))).clamp_(min=0)
        if support is not None:
            xn.mul_(support)
        if reweight and it >= reweight["start"] and (it - reweight["start"]) % reweight["every"] == 0:
            bound = lam / (D.W(xn).abs() / reweight.get("ref", lam) + reweight["eps"])
        u = torch.maximum(torch.minimum(u + sig * D.W(2 * xn - x), bound), -bound)
        if it % print_every == 0 or it == n_iter - 1:
            rel = float((xn - x).norm() / max(float(xn.norm()), 1e-30))
            hist.append(dict(iter=it, flux=float(xn.sum()), rel_change=rel, resid_rms=float(g.std()),
                             peak_ratio=float((op.d + g).max() / op.d.max())))
            log(f"iter {it:6d}  flux {hist[-1]['flux']:8.3f}  change {rel:.2e}  resid rms {hist[-1]['resid_rms']:.3e}")
        x = xn
        if it + 1 in snaps:
            kept[it + 1] = x.cpu().numpy()
        if callback is not None:
            callback(it + 1, x, g)
    return x, u, hist, kept


# ======================================================================
# helpers
# ======================================================================
def rebin(dirty, psf, b):
    """b x b binning: dirty -> block mean; PSF -> box-smoothed twice and sampled every
    b pixels from its peak (so that d_b = psf_b (*) x_b for x_b the block-summed sky),
    renormalized to peak 1."""
    ny, nx = dirty.shape[1] // b * b, dirty.shape[2] // b * b
    d = dirty[:, :ny, :nx].reshape(dirty.shape[0], ny // b, b, nx // b, b).mean((2, 4))
    py, px = np.unravel_index(np.argmax(psf[0]), psf[0].shape)
    sm = uniform_filter(uniform_filter(psf, size=(1, b, b), mode="constant"), size=(1, b, b), mode="constant")
    pb = sm[:, py % b::b, px % b::b]
    n = min(pb.shape[1], pb.shape[2]) // 2 * 2
    pb = pb[:, :n, :n]
    return d, pb / pb.max(axis=(1, 2), keepdims=True)


def upsample(x, b):
    """flux-conserving inverse of rebin's block sum: each pixel split into b x b"""
    return np.kron(x, np.ones((1, b, b))) / b ** 2


def grow_support(mask, resid, sig_j, side_j, scales, nsig=5.0, slfac=1.5, grow_px=10, vchan=3):
    """One step of the iteratively grown support used for the extended cube:
    starlet planes `scales` of the velocity-smoothed residual, significant where
    > max(nsig x the plane's noise sigma, slfac x PSF-sidelobe ratio x the plane's peak);
    dilated by grow_px spatially and +-1 channel; OR-ed into `mask`."""
    nz, ny, nx = resid.shape
    vs = F.avg_pool1d(resid.permute(1, 2, 0).reshape(-1, 1, nz), vchan, 1, vchan // 2,
                      count_include_pad=False).reshape(ny, nx, nz).permute(2, 0, 1)
    pl = starlet(vs, max(scales) + 1)
    new = torch.zeros_like(mask)
    for j in scales:
        new |= pl[j] > max(nsig * sig_j[j], slfac * side_j[j] * float(pl[j].max()))
    yy, xx = torch.meshgrid(torch.arange(-grow_px, grow_px + 1), torch.arange(-grow_px, grow_px + 1), indexing="ij")
    disk = ((yy ** 2 + xx ** 2) <= grow_px ** 2).to(DEV, DT)[None, None]
    new = F.conv2d(new.to(DT)[:, None], disk, padding=grow_px)[:, 0] > 0
    new = F.max_pool1d(new.to(DT).permute(1, 2, 0).reshape(-1, 1, nz), 3, 1, 1).reshape(ny, nx, nz).permute(2, 0, 1) > 0
    return mask | new
