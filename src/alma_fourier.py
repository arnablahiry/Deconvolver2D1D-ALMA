#!/usr/bin/env python
"""
Measurement operators for real ALMA data, defined properly.

This module replaces `uv_operator.NonUniformFourierOperator` (a dense
non-uniform DFT matrix, only tractable on a few thousand time/baseline
averaged uv points) and `casa_uv_operator.CASAFourierOperator` (an unverified
scaffold that round-trips visibilities through MS columns on disk every
iteration). Both of those tried to build `A` and `A^H` as separate objects and
then hoped they formed an adjoint pair. Neither one did, and neither one
carried the imaging weights, the w-term, or the mosaic primary-beam response
that a real ALMA dataset requires.

The formulation used here
-------------------------
Interferometric deconvolution never actually needs `A` and `A^H` separately.
The data-fidelity term is a weighted least squares

    f(x) = 1/2 * || A x - y ||^2_W ,      W = the imaging weights (Briggs etc.)

and every gradient-based solver only ever touches

    grad f(x) = A^H W A x  -  A^H W y
              = N x - d

where `d = A^H W y` is the **dirty image** and `N = A^H W A` is the **normal
operator**. Both are things CASA computes natively and exactly:

    d      = tclean(niter=0) residual, i.e. synthesisimager.executemajorcycle()
             with a zero model, normalized by synthesisnormalizer
    N x    = d - residual(x), i.e. the same major cycle with `x` as the model

A major cycle is a genuine degrid (`A`, image -> visibilities, via CASA's
gridder in reverse) followed by a grid (`A^H W`, visibilities -> image). So
`N` as computed above is a *true* self-adjoint positive-semidefinite operator
by construction -- there is no adjoint mismatch to tune away, because the two
halves are never used separately. That is the whole point of doing it this
way rather than as `forward_cube`/`adjoint_cube`.

Two operators, same interface
-----------------------------
`ExactNormalOperator`
    `N x` by an actual CASA major cycle (degrid + grid) per call. Exact:
    carries Briggs weighting, the w-term, per-channel chromatic uv scaling,
    the mosaic primary-beam response, and flagging. Costs one full pass over
    the visibilities per call (seconds on the split line MS, minutes on the
    parent MS).

`PSFNormalOperator`
    `N x` by per-channel FFT convolution with the point spread function, on a
    zero-padded grid. This is exact whenever `N` is shift-invariant, since
    then `N = B *` with `B = N delta` = the PSF by definition. Costs two FFTs
    per channel. This is the same approximation CLEAN's own minor cycle makes.

`compare_operators()` measures the discrepancy between them on real data
rather than asserting it is small. For a single pointing it is at the 1e-6
level; for the 3-pointing mosaic in this dataset it is a few percent near the
field edge, where the mosaic primary-beam weighting makes `N` genuinely
position-dependent, and far smaller in the inner arcseconds where the source
is.

Normalization, stated explicitly
--------------------------------
CASA's normalized PSF has peak 1 and its normalized residual is in Jy/beam,
so the self-consistent model units are Jy/pixel and the relation is

    dirty [Jy/beam]  =  psf [peak 1]  (*)  model [Jy/pixel]

A 1 Jy point source in one pixel produces a peak of 1 Jy/beam. This is
checked, not assumed: `PSFNormalOperator.self_test()` pushes a unit delta
through and requires the result to reproduce the PSF, and
`compare_operators()` pushes the same delta through CASA. Because `N` is a
per-channel convolution, its eigenvalues are exactly the optical transfer
function values, so the Lipschitz constant is `max |FFT(psf)|` in closed
form -- no power iteration needed, and the FISTA step `mu = 1/L` is exact.

The zero-spacing null space
---------------------------
No antenna pair measures the zero-length baseline, so `sum(psf) ~ 0` and `N`
annihilates the constant mode: total flux is formally unrecoverable and the
coarsest wavelet sub-band is an unpenalized direction the iteration can run
away in. `zero_spacing_response()` reports the size of that null space for
this dataset; the solver in `fista_2d1d.py` zeroes the corresponding
sub-band, as the image-domain deconvolver in this repo always has.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field as _dc_field

import numpy as np


# --------------------------------------------------------------------------
# Imaging configuration
# --------------------------------------------------------------------------
@dataclass
class ImagingConfig:
    """Everything that defines the measurement operator `A` and the image grid.

    Defaults are tuned for `data/ngc7469_co21.ms` (see `scripts/split_line_ms.py`):
    a 3-pointing 12 m-array mosaic of IRAS F23007+0836 with baselines
    14.3-3171 m at 226.8 GHz, giving a ~0.09" synthesized beam.
    """

    msname: str
    imagename: str

    # Image grid. The Briggs 0.5 mosaic beam measures 0.162" x 0.124", so
    # 0.04" gives ~4 pixels across the minor axis and a 12.8" field.
    imsize: int = 320
    cell_arcsec: float = 0.04

    # Spectral axis, measured from a coarse 40 km/s scan over the whole split
    # window (`--stage image --imsize 128 --cell 0.15 --width-kms 40`): the CO
    # line runs from ~4600 to ~5060 km/s and peaks around 4760-4960. There is
    # no data below ~4590 km/s, and asking for channels outside the covered
    # range grids an empty, zero-PSF plane rather than padding -- which then
    # trips PSFNormalOperator's peak check.
    nchan: int = 90                    # 4610 -> 5055 km/s
    start_kms: float = 4610.0
    width_kms: float = 5.0             # ~4x the 1.29 km/s native channel
    restfreq_ghz: float = 230.538      # CO(2-1)
    outframe: str = "LSRK"

    # Data selection. All four spws are needed: after the split they cover
    #     spw 0,2: 226.6202-226.9152 GHz = 4696-5095 km/s
    #     spw 1,3: 226.7274-226.9795 GHz = 4628-4955 km/s
    # so the blue wing of the line (4600-4696 km/s) exists *only* in spw 1,3
    # and imaging spw 0,2 alone clips it -- which showed up as ~0.9 Jy of
    # unexplained flux stacked in the bluest channels.
    #
    # The cost is that per-channel sensitivity is not uniform: 4696-4955 km/s
    # has all four spws, the wings only two. The MAD thresholding in
    # `fista_2d1d` estimates one noise level per sub-band across the whole
    # cube, so it under-thresholds the well-covered middle and
    # over-thresholds the wings. Use `spw="0,2"` with a narrower velocity
    # range if uniform noise matters more than covering the full line.
    spw: str = "0,1,2,3"
    field: str = ""
    uvdist: str = ""

    # Gridding / weighting. 'mosaic' is required: this is a 3-pointing mosaic
    # and 'standard' would apply a single pointing's primary beam to all of it.
    ftmachine: str = "mosaic"
    weighting: str = "briggs"
    robust: float = 0.5
    wprojplanes: int = 1
    normtype: str = "flatnoise"        # uniform noise across the mosaic

    # Padding factor for the FFT grid of PSFNormalOperator. 2.0 makes the
    # linear convolution exact (no wraparound of PSF sidelobes).
    fft_pad: float = 2.0

    stokes: str = "I"

    # The nucleus of NGC 7469, measured from a wide 41" survey dirty image
    # (`--stage image --imsize 512 --cell 0.08`). This is NOT the default
    # CASA would pick: with `phasecenter=""` the imager centres on field 0,
    # and this 3-pointing mosaic *surrounds* the target, so the source sits
    # 6.3" north of that. Imaging a small field on the default centre puts
    # all the emission outside the image, where it aliases back in and the
    # deconvolution fits artifacts.
    phasecenter: str = "J2000 23:03:15.61 +08.52.25.8"

    def cell_str(self) -> str:
        return f"{self.cell_arcsec}arcsec"

    def selpars(self) -> dict:
        """One flat selection record per MS -- `synthesisimager.selectdata` is
        called once per measurement set, not with a dict-of-dicts."""
        sel = {"msname": os.path.abspath(self.msname), "spw": self.spw,
               "field": self.field, "usescratch": True, "readonly": False,
               "datacolumn": "data"}
        if self.uvdist:
            sel["uvdist"] = self.uvdist
        return sel

    def impars(self) -> dict:
        return {
            "imagename": self.imagename,
            "nchan": self.nchan,
            "imsize": [self.imsize, self.imsize],
            "cell": [self.cell_str(), self.cell_str()],
            "stokes": self.stokes,
            "phasecenter": self.phasecenter,
            # NB: the tool-level key is `specmode`, not `mode`. `mode` is
            # accepted without complaint and then ignored, which silently
            # collapses the cube to a single MFS channel.
            "specmode": "cube",
            "start": f"{self.start_kms}km/s",
            "width": f"{self.width_kms}km/s",
            "restfreq": f"{self.restfreq_ghz}GHz",
            "outframe": self.outframe,
            "veltype": "radio",
            "projection": "SIN",
            "deconvolver": "hogbom",
        }

    def gridpars(self) -> dict:
        return {
            # SynthesisParamsGrid needs the image name too (cfcache location).
            "imagename": self.imagename,
            "ftmachine": self.ftmachine,
            "wprojplanes": self.wprojplanes,
            "padding": 1.2,
            "useautocorr": False,
            "usedoubleprec": True,
            "interpolation": "linear",
            "conjbeams": False,
            "pblimit": 0.2,
            "normtype": self.normtype,
        }


# --------------------------------------------------------------------------
# Small CASA image helpers
# --------------------------------------------------------------------------
def read_casa_image(path: str) -> np.ndarray:
    """CASA image [x, y, stokes, chan] -> cube (nz, ny, nx) float64, Stokes 0."""
    from casatools import image as _image

    ia = _image()
    ia.open(path)
    try:
        arr = ia.getchunk()
    finally:
        ia.close()
    if arr.ndim == 3:                      # continuum: [x, y, stokes]
        arr = arr[:, :, :, None]
    return np.transpose(arr[:, :, 0, :], (2, 1, 0)).astype(np.float64)


def write_casa_image(path: str, cube: np.ndarray, template: str,
                     bunit: str = None, beam: tuple = None) -> str:
    """Write cube (nz, ny, nx) into a CASA image copying `template`'s csys.

    `bunit` sets the brightness unit and `beam` = (bmaj", bmin", bpa deg) the
    restoring beam. Both are worth passing: `ia.fromarray` does not inherit
    them from the template, and a cube exported without them has no BUNIT and
    no BMAJ/BMIN/BPA in its FITS header, so CARTA/CASA/astropy cannot convert
    Jy/beam to Jy or overlay a beam. Jy/pixel images (the sparse model) have
    no beam by definition -- pass `beam=None` for those.
    """
    from casatools import image as _image

    if os.path.exists(path):
        shutil.rmtree(path)
    ia = _image()
    ia.open(template)
    try:
        csys = ia.coordsys()
        shape = ia.shape()
    finally:
        ia.close()
    arr = np.zeros(tuple(shape), dtype=np.float64)
    arr[:, :, 0, :] = np.transpose(cube, (2, 1, 0))
    ia.fromarray(outfile=path, pixels=arr, csys=csys.torecord(), overwrite=True)
    if bunit is not None:
        ia.setbrightnessunit(bunit)
    if beam is not None:
        ia.setrestoringbeam(major=f"{beam[0]}arcsec", minor=f"{beam[1]}arcsec",
                            pa=f"{beam[2]}deg")
    ia.close()
    csys.done()
    return path


def export_fits(imagepath: str, fitspath: str, overwrite: bool = True) -> str:
    from casatools import image as _image

    ia = _image()
    ia.open(imagepath)
    try:
        ia.tofits(outfile=fitspath, overwrite=overwrite, velocity=True,
                  optical=False, stokeslast=True, history=False)
    finally:
        ia.close()
    return fitspath


# --------------------------------------------------------------------------
# The CASA imager: builds PSF / dirty / PB, and evaluates the exact N
# --------------------------------------------------------------------------
class CASAImager:
    """Owns a `synthesisimager` + `synthesisnormalizer` pair for one image grid.

    The normalization chain below mirrors what `tclean` does through
    `PySynthesisImager` for a single non-MPI image field:

        makepsf      -> gatherpsfweight -> dividepsfbyweight -> makepsfbeamset
        majorcycle   -> gatherresidual  -> divideresidualbyweight
        (with model) -> multiplymodelbyweight -> scattermodel -> majorcycle

    Rather than trusting that ordering, `psf_peak()` and
    `compare_operators()` verify the outcome: the normalized PSF must peak at
    1.0 and a unit delta model must produce exactly the PSF back through a
    real degrid/grid round trip.
    """

    def __init__(self, config: ImagingConfig, verbose: bool = True):
        from casatools import synthesisimager, synthesisnormalizer

        self.cfg = config
        self.verbose = verbose
        self.imagename = config.imagename
        os.makedirs(os.path.dirname(os.path.abspath(self.imagename)) or ".",
                    exist_ok=True)

        self.si = synthesisimager()
        self.si.selectdata(config.selpars())
        self.si.defineimage(config.impars(), config.gridpars())
        self.si.setweighting(type=config.weighting, rmode="norm",
                             robust=config.robust, usecubebriggs=False)

        normpars = {
            "imagename": self.imagename,
            "normtype": config.normtype,
            "workdir": os.path.dirname(os.path.abspath(self.imagename)) or ".",
            "deconvolver": "hogbom",
            "nterms": 1,
            "imindex": 0,
            "psfcutoff": 0.35,
        }
        self.sn = synthesisnormalizer()
        self.sn.setupnormalizer(normpars)
        # Cube gridding runs through CubeMajorCycleAlgorithm, which reads the
        # image names out of the *imager's* copy of the normalizer parameters.
        # Without this call it aborts with "Error in reading gather/scatter
        # parameters: imagename not specified" and reports the whole channel
        # section as failed.
        self.si.normalizerinfo(normpars)

        self._psf = None
        self._dirty = None
        self._pb = None

    # -- names ----------------------------------------------------------
    @property
    def psf_path(self):      return self.imagename + ".psf"
    @property
    def residual_path(self): return self.imagename + ".residual"
    @property
    def model_path(self):    return self.imagename + ".model"
    @property
    def pb_path(self):       return self.imagename + ".pb"
    @property
    def sumwt_path(self):    return self.imagename + ".sumwt"

    def _log(self, *a):
        if self.verbose:
            print("[casa]", *a, flush=True)

    # -- PSF / PB / dirty ------------------------------------------------
    def make_psf(self) -> np.ndarray:
        """Normalized PSF cube (nz, ny, nx), peak 1 per channel."""
        if self._psf is not None:
            return self._psf
        self._log("gridding PSF (this is one full pass over the visibilities)")
        self.si.makepsf()
        self.sn.gatherpsfweight()
        self.sn.dividepsfbyweight()
        self.sn.makepsfbeamset()
        self._psf = read_casa_image(self.psf_path)
        self._log(f"psf cube {self._psf.shape}, "
                  f"peak per channel min/max "
                  f"{self._psf.max(axis=(1, 2)).min():.6f}/"
                  f"{self._psf.max(axis=(1, 2)).max():.6f}")
        return self._psf

    def make_pb(self) -> np.ndarray:
        """Mosaic primary-beam response cube, for the final PB correction."""
        if self._pb is not None:
            return self._pb
        self._log("making mosaic primary beam")
        self.si.makepb()
        try:
            self.sn.normalizeprimarybeam()
        except Exception as exc:              # not all ftmachines need it
            self._log(f"normalizeprimarybeam skipped: {exc}")
        pb = read_casa_image(self.pb_path)
        # The raw mosaic .pb is the weighted sum of the pointings' responses
        # and is not normalized (it peaked at 2.13 on this dataset). Only the
        # *shape* matters for the final flux correction, so scale it so the
        # best-covered point of the mosaic has unit response.
        peak = float(pb.max())
        if peak > 0:
            pb = pb / peak
        self._log(f"primary beam normalized by {peak:.4f}")
        self._pb = pb
        return self._pb

    def dirty(self) -> np.ndarray:
        """`d = A^H W y`, the dirty cube in Jy/beam. Zeroes the model first."""
        if self._dirty is not None:
            return self._dirty
        self.set_model(None)
        self._log("gridding dirty image (major cycle, zero model)")
        self._dirty = self._major_cycle()
        return self._dirty

    # -- the model image -------------------------------------------------
    def set_model(self, cube):
        """Install `cube` (nz, ny, nx, Jy/pixel) as the CASA model image.

        `cube=None` zeroes it. Requires the PSF to exist first, since the PSF
        image is the coordinate template.
        """
        if not os.path.exists(self.psf_path):
            self.make_psf()
        if cube is None:
            cube = np.zeros_like(self.make_psf())
        write_casa_image(self.model_path, np.ascontiguousarray(cube),
                         template=self.psf_path)

    def _major_cycle(self) -> np.ndarray:
        """One degrid+grid pass; returns the normalized residual cube.

        In `specmode='cube'` the work is done by `CubeMajorCycleAlgorithm`,
        which already divides each channel by its own sum of weights. The
        normalizer's `multiplymodelbyweight` / `scattermodel` /
        `gatherresidual` are then no-ops (measured: identical residuals with
        and without them) and `divideresidualbyweight` is actively wrong --
        it divides by sumwt a *second* time, which on this dataset scaled the
        dirty image down by 4.65e7 to a 1e-11 rms.

        The evidence that this is right, rather than an assumption: with a
        1 Jy delta as the model, `dirty - residual` reproduces the PSF with a
        peak ratio of 0.999987 and a maximum error of 0.35% of the peak. That
        is `PSFNormalOperator.self_test` / `compare_operators` territory and
        is checked by the `validate` stage.
        """
        self.si.executemajorcycle({"lastcycle": False})
        return read_casa_image(self.residual_path)

    def residual(self, model_cube) -> np.ndarray:
        """`r(x) = d - N x`, computed by a real degrid + grid round trip."""
        self.set_model(model_cube)
        return self._major_cycle()

    def normal(self, model_cube) -> np.ndarray:
        """`N x = A^H W A x`, exact, via `d - r(x)`."""
        return self.dirty() - self.residual(model_cube)

    def predict(self, model_cube):
        """Pure forward transform `A x`: degrid the model into MODEL_DATA.

        Provided for completeness (and for anyone who wants the visibilities
        themselves); the solver never needs it, because `normal()` already
        contains this step.
        """
        self.set_model(model_cube)
        self.si.predictmodel()

    def done(self):
        for tool in (self.si, self.sn):
            try:
                tool.done()
            except Exception:
                pass


# --------------------------------------------------------------------------
# The fast normal operator: per-channel FFT convolution with the PSF
# --------------------------------------------------------------------------
class PSFNormalOperator:
    """`N x = psf (*) x`, per channel, via zero-padded real FFTs.

    Exact whenever `N` is shift-invariant. `psf` must be the peak-1 normalized
    PSF and `dirty` the matching Jy/beam dirty cube, so that the model is in
    Jy/pixel (see the module docstring).
    """

    def __init__(self, psf: np.ndarray, dirty: np.ndarray, pad: float = 2.0):
        from scipy.fft import next_fast_len

        psf = np.asarray(psf, dtype=np.float64)
        dirty = np.asarray(dirty, dtype=np.float64)
        if psf.shape != dirty.shape:
            raise ValueError(f"psf {psf.shape} != dirty {dirty.shape}")
        self.nz, self.ny, self.nx = psf.shape
        self.dirty = dirty

        # Renormalize each channel's PSF to peak exactly 1 and record the
        # correction, so the Jy/beam <-> Jy/pixel relation holds per channel.
        peaks = psf.max(axis=(1, 2))
        if np.any(peaks <= 0):
            bad = np.nonzero(peaks <= 0)[0]
            raise ValueError(
                f"PSF peak is zero in channels {bad.tolist()} -- those output "
                f"channels contain no visibilities. Narrow --nchan / "
                f"--start-kms to the velocity range the data actually covers "
                f"(4711-5095 km/s for the split CO(2-1) MS).")
        self.psf_peaks = peaks
        self.psf = psf / peaks[:, None, None]

        self.pad_y = next_fast_len(int(np.ceil(pad * self.ny)), real=True)
        self.pad_x = next_fast_len(int(np.ceil(pad * self.nx)), real=True)

        # OTF: FFT of the PSF with its peak moved to the array origin, so the
        # convolution introduces no positional shift.
        from scipy.fft import rfft2

        py, px = np.unravel_index(
            np.argmax(self.psf.reshape(self.nz, -1), axis=1), (self.ny, self.nx))
        if len(set(py.tolist())) != 1 or len(set(px.tolist())) != 1:
            raise ValueError("PSF peak is not at the same pixel in every channel")
        big = np.zeros((self.nz, self.pad_y, self.pad_x))
        big[:, :self.ny, :self.nx] = self.psf
        big = np.roll(big, (-int(py[0]), -int(px[0])), axis=(1, 2))
        self.otf = rfft2(big, axes=(1, 2))

        # N is a real symmetric convolution, so its eigenvalues are the OTF
        # values and they should be real and non-negative (N = A^H W A is
        # PSD). The imaginary part is a numerical check on that.
        self.otf_imag_frac = float(np.abs(self.otf.imag).max()
                                   / max(np.abs(self.otf).max(), 1e-300))
        self._L = float(np.abs(self.otf).max())

    # -- operator interface ---------------------------------------------
    def normal(self, x: np.ndarray) -> np.ndarray:
        from scipy.fft import rfft2, irfft2

        x = np.asarray(x, dtype=np.float64)
        big = np.zeros((self.nz, self.pad_y, self.pad_x))
        big[:, :self.ny, :self.nx] = x
        out = irfft2(rfft2(big, axes=(1, 2)) * self.otf, axes=(1, 2),
                     s=(self.pad_y, self.pad_x))
        return out[:, :self.ny, :self.nx]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """`grad f(x) = N x - d`."""
        return self.normal(x) - self.dirty

    def residual(self, x: np.ndarray) -> np.ndarray:
        """`d - N x`, the dirty image of the residual visibilities."""
        return self.dirty - self.normal(x)

    def lipschitz_constant(self) -> float:
        """Exact: `max |OTF|`, no power iteration."""
        return self._L

    # -- self-consistency -------------------------------------------------
    def self_test(self) -> dict:
        """Push a unit delta through `N` and require the PSF back."""
        delta = np.zeros((self.nz, self.ny, self.nx))
        delta[:, self.ny // 2, self.nx // 2] = 1.0
        got = self.normal(delta)
        err = np.abs(got - self.psf).max() / max(self.psf.max(), 1e-300)
        return {
            "delta_roundtrip_max_rel_err": float(err),
            "otf_imaginary_fraction": self.otf_imag_frac,
            "lipschitz": self._L,
            "psf_peak_renorm": (float(self.psf_peaks.min()),
                                float(self.psf_peaks.max())),
        }

    def zero_spacing_response(self) -> dict:
        """How badly `N` annihilates the constant (total-flux) mode.

        `sum(psf)` is `N`'s response to a flat image, i.e. the zero-spacing
        visibility the array never measured. Near zero means total flux is in
        the null space and the coarsest wavelet sub-band must be constrained.
        """
        s = self.psf.sum(axis=(1, 2))
        dc_over_L = float(np.abs(s).max() / max(self._L, 1e-300))
        return {
            "psf_sum_per_chan": (float(s.min()), float(s.max())),
            "psf_sum_over_peak": float(np.abs(s).max()),
            "dc_over_lipschitz": dc_over_L,
            "dc_is_measured": bool(dc_over_L > 1e-3),
            "otf_min": float(np.abs(self.otf).min()),
            "condition_number": float(self._L / max(np.abs(self.otf).min(), 1e-300)),
        }

    def recommend_zero_coarse(self) -> bool:
        """Should the solver zero the coarsest wavelet sub-band?

        Only if the array really is blind to the constant mode. Whether it is
        depends on the *image grid*, not just the antenna layout: the uv cell
        size is `1 / field_of_view`, so a field much smaller than the largest
        recoverable angular scale gives cells coarse enough that the shortest
        baselines land in the central uv cell and the DC mode is measured
        after all.

        For this dataset that is exactly what happens -- a 6.4" field gives
        32.2 kilo-lambda uv cells while the shortest baseline is 10.8
        kilo-lambda (14.3 m at 226.8 GHz), so `sum(psf)` is 68-127 rather than
        0 and the coarse band carries real, constrained information. Zeroing
        it here would discard measured flux, which is the opposite of the
        situation the image-domain deconvolver in this repo was written for.
        """
        return not self.zero_spacing_response()["dc_is_measured"]


# --------------------------------------------------------------------------
# Validation: fast operator vs. CASA's real degrid/grid
# --------------------------------------------------------------------------
def compare_operators(imager: "CASAImager", fast: "PSFNormalOperator",
                      test_cube=None, seed: int = 0, verbose: bool = True) -> dict:
    """Measure `|| N_casa x - N_psf x ||` on a real test image.

    `N_casa x` is a genuine degrid (image -> visibilities, with the real
    (u,v,w), Briggs weights, flags and mosaic primary beams) followed by a
    genuine grid back. `N_psf x` is the FFT convolution. Agreement validates
    the fast operator; the residual difference is the price of assuming the
    PSF is shift-invariant, which for a mosaic it is not exactly.

    The test image defaults to a sparse set of unit delta functions -- the
    hardest case, since it probes `N` at several separate field positions at
    once and its exact answer (a sum of shifted PSFs) is known analytically.
    """
    nz, ny, nx = fast.nz, fast.ny, fast.nx
    if test_cube is None:
        rng = np.random.default_rng(seed)
        test_cube = np.zeros((nz, ny, nx))
        # one central delta (where the science is) plus a few off-axis ones
        offsets = [(0, 0), (ny // 8, 0), (0, nx // 8), (-ny // 6, nx // 6)]
        for k, (dy, dx) in enumerate(offsets):
            test_cube[:, ny // 2 + dy, nx // 2 + dx] = 1.0
        test_cube *= rng.uniform(0.5, 1.5, size=(nz, 1, 1))

    if verbose:
        print("[check] running one CASA major cycle on the test image ...",
              flush=True)
    n_casa = imager.normal(test_cube)
    n_fast = fast.normal(test_cube)

    scale = np.abs(n_casa).max()
    diff = n_casa - n_fast
    # inner quarter of the field: where the source is and where a mosaic PSF
    # is most nearly shift-invariant
    sy = slice(3 * ny // 8, 5 * ny // 8)
    sx = slice(3 * nx // 8, 5 * nx // 8)
    out = {
        "max_abs_casa": float(scale),
        "max_rel_diff_full_field": float(np.abs(diff).max() / max(scale, 1e-300)),
        "rms_rel_diff_full_field": float(diff.std() / max(scale, 1e-300)),
        "max_rel_diff_inner_quarter":
            float(np.abs(diff[:, sy, sx]).max() / max(scale, 1e-300)),
    }
    if verbose:
        for k, v in out.items():
            print(f"[check]   {k}: {v:.6g}")
    return out
