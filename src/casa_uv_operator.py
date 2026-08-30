#!/usr/bin/env python
"""
CASAFourierOperator: the same forward/adjoint measurement-operator interface
as `uv_operator.NonUniformFourierOperator`, but with the visibility <-> image
("dirac space") transforms done by **CASA's own gridder/degridder**
(`casatools.imager`) instead of this repo's dense non-uniform-DFT matrix.

Why this exists
---------------
`NonUniformFourierOperator` is a dense `(n_uv, n_pixels)` matrix, which is
only tractable after `export_visibilities.py` time/baseline-averages the real
MS down to ~1000 (u, v) points per channel. That averaging is exactly what
made this repo's dirty beam/image differ from CASA's (which images the full
~68000 visibilities/channel with Briggs weighting). This operator removes
that gap by never averaging: every forward/adjoint application goes through
CASA on the *full* measurement set, so `UVDeconvolver2D1D` (which only ever
calls `operator.forward_cube`/`adjoint_cube`/`lipschitz_constant`) fits the
same visibilities, with the same gridding/weighting, that CASA `tclean`
itself uses.

    forward_cube(model_cube)  = CASA predict:  im.ft(model) -> MODEL_DATA      (image -> visibilities)
    adjoint_cube(vis_cube)    = CASA grid:     put vis in a column, im.makeimage -> dirty image  (visibilities -> image)

*** SUPERSEDED BY `alma_fourier.py`. ***
Use `alma_fourier.CASAImager` / `alma_fourier.PSFNormalOperator` instead.
This module's central assumption -- that `forward_cube` (im.ft) and
`adjoint_cube` (im.makeimage) form an adjoint pair -- is false: the gridder
applies imaging weights that the degridder does not, so they differ by `W`.
`alma_fourier` sidesteps the problem entirely by never using the two halves
separately, only the normal operator `N = A^H W A`, which a single CASA major
cycle computes exactly. It also drops this module's per-iteration write to
the MS `CORRECTED_DATA` column, which is both slow and destructive.

*** STATUS: UNTESTED IN THIS REPO'S SANDBOX. ***
`casatools` is not available in the environment the rest of this repo's
tooling was built/run in (same constraint as `export_visibilities.py` /
`run_ms_clean_uv_grid.py`), so this module has NOT been executed or verified
here. It is written to the documented casatools API and is meant to be run
and debugged in a CASA environment (`casa_env`). Treat the specific tool
calls below as a starting scaffold, not verified-correct code -- in
particular the forward/adjoint normalization (so that they form a true
adjoint pair and `mu = 1/L` is the correct step size) is the most likely
thing to need tuning against CASA's actual gridding normalization.

Interface contract (must match uv_operator.NonUniformFourierOperator so
UVDeconvolver2D1D is a drop-in):
    - attribute `n_uv`            : number of visibility rows per channel
    - forward_cube(cube)  -> (nz, n_uv) complex
    - adjoint_cube(vis)   -> (nz, ny, nx) real
    - lipschitz_constant() -> float
"""

import os
import numpy as np

try:
    from casatools import imager as _imager_tool
    from casatools import image as _image_tool
    from casatools import table as _table_tool
    from casatools import calibrater as _calibrater_tool
except ImportError as exc:  # pragma: no cover -- only runs inside CASA
    raise ImportError(
        "casa_uv_operator requires casatools (CASA). Run this in a CASA "
        "environment (e.g. casa_env); it is not importable in the sandbox "
        "the rest of this repo's pure-numpy tooling runs in."
    ) from exc


class CASAFourierOperator:
    """
    CASA-gridder-backed measurement operator. See module docstring.

    Parameters
    ----------
    msname : str
        Path to the calibrated measurement set (e.g. `twhya_calibrated.ms`).
    nx, ny : int
        Image size (pixels).
    cell_arcsec : float
        Pixel scale in arcsec (e.g. 0.2099 to match the uv-notebook grid, or
        0.08 to match CASA's benchmark grid).
    spw, field : str
        Spectral-window / field selection, matching `prep_wavelet_data.py`.
    nchan, start, width : int
        Channel selection (CASA `tclean`/`im` convention), matching the rest
        of the pipeline's 40-channel CO(3-2) window.
    weighting, robust : str, float
        Imaging weighting -- 'briggs' + robust=0.5 to match the CASA
        benchmark exactly.
    workdir : str
        Scratch directory for the temporary model/dirty CASA images this
        operator writes and reads each iteration.
    """

    def __init__(self, msname, nx, ny, cell_arcsec, spw="0", field="",
                 nchan=40, start=0, width=1, weighting="briggs", robust=0.5,
                 workdir="."):
        self.msname = msname
        self.nx, self.ny = nx, ny
        self.cell = f"{cell_arcsec}arcsec"
        self.spw, self.field = spw, field
        self.nchan, self.start, self.width = nchan, start, width
        self.weighting, self.robust = weighting, robust
        self.workdir = workdir

        self._model_path = os.path.join(workdir, "_casaop_model.image")
        self._dirty_path = os.path.join(workdir, "_casaop_dirty.image")

        # Ensure the MODEL_DATA / CORRECTED_DATA scratch columns exist -- the
        # forward (im.ft -> MODEL_DATA) and adjoint (write CORRECTED_DATA ->
        # grid) round-trips need them, and a fresh calibrated MS often only
        # has DATA. `calibrater.open(addmodel=..., addcorr=...)` creates them
        # (MODEL_DATA initialized to 1, CORRECTED_DATA to DATA) -- reversible
        # later with `delmod` / `clearcal`.
        cb = _calibrater_tool()
        cb.open(msname, addcorr=True, addmodel=True)
        cb.close()

        # One imager tool kept open for the operator's lifetime.
        self.im = _imager_tool()
        self.im.open(msname, usescratch=True)
        self.im.selectvis(spw=spw, field=field, nchan=nchan, start=start, step=width)
        self.im.defineimage(nx=nx, ny=ny, cellx=self.cell, celly=self.cell,
                            spw=[int(spw)], nchan=nchan, start=start, step=width,
                            mode="channel")
        self.im.weight(type=weighting, rmode="norm", robust=robust)

        # Number of visibility rows per channel (for the interface contract).
        tb = _table_tool()
        tb.open(msname)
        # NB: real selection is per spw/field; this is the whole-MS row count.
        # Adjust to the actual selected subset if spw/field narrow it down.
        self.n_uv = int(tb.nrows())
        tb.close()

    # -- helpers ---------------------------------------------------------
    def _cube_to_model_image(self, cube):
        """Write a (nz, ny, nx) numpy cube into `self._model_path` as a CASA
        image with this operator's coordinate system. Reuses the dirty image
        as a template on first call so the csys/beam/axes line up."""
        ia = _image_tool()
        # Expect an existing template (make one via a niter=0 makeimage first
        # run if absent). CASA images are [x, y, stokes, chan]; transpose.
        if not os.path.exists(self._dirty_path):
            self.im.makeimage(type="observed", image=self._dirty_path)
        ia.open(self._dirty_path)
        csys = ia.coordsys()
        shape = ia.shape()  # (nx, ny, nstokes, nchan)
        ia.close()
        arr = np.zeros(shape, dtype=np.float64)
        # cube (nz, ny, nx) -> CASA (nx, ny, 1, nz)
        arr[:, :, 0, :] = np.transpose(cube, (2, 1, 0))
        os.system(f"rm -rf {self._model_path}")
        ia.fromarray(outfile=self._model_path, pixels=arr, csys=csys.torecord(),
                     overwrite=True)
        ia.close()
        csys.done()

    def _read_column_as_cube_vis(self, column):
        """Read an MS visibility column as (n_row, n_chan) complex (Stokes-I,
        polarization-averaged). `forward_cube` transposes to (n_chan, n_uv)."""
        tb = _table_tool()
        tb.open(self.msname)
        data = tb.getcol(column)   # (n_corr, n_chan, n_row)
        tb.close()
        return data.mean(axis=0).T  # (n_row, n_chan)

    def observed_visibilities(self, column="DATA"):
        """The observed visibilities as (nz, n_uv) complex -- the `vis_obs`
        argument `UVDeconvolver2D1D.deconvolve` expects. Reads the calibrated
        `DATA` column (Stokes-I, polarization-averaged) by default."""
        return self._read_column_as_cube_vis(column).T  # (n_chan, n_uv)

    # -- operator interface ---------------------------------------------
    def forward_cube(self, cube):
        """(nz, ny, nx) real model image -> (nz, n_uv) complex visibilities,
        via CASA predict (im.ft fills MODEL_DATA), read back."""
        self._cube_to_model_image(cube)
        self.im.ft(model=self._model_path, incremental=False)
        vis = self._read_column_as_cube_vis("MODEL_DATA")  # (n_row, n_chan)
        return vis.T  # (n_chan, n_uv)

    def adjoint_cube(self, vis):
        """(nz, n_uv) complex visibilities -> (nz, ny, nx) real dirty image,
        via CASA gridding. Writes `vis` into CORRECTED_DATA, then grids it.

        NOTE: this round-trips visibilities through an MS column and a CASA
        image on disk every call -- correct but slow; a production version
        would keep everything in memory via the SynthesisImager tool. The
        normalization here must be matched to CASA's gridder so that
        adjoint_cube is the true adjoint of forward_cube (see module
        docstring's status note)."""
        tb = _table_tool()
        tb.open(self.msname, nomodify=False)
        col = tb.getcol("CORRECTED_DATA")  # (n_corr, n_chan, n_row)
        vis_t = vis.T  # (n_uv, n_chan) == (n_row, n_chan)
        for c in range(col.shape[0]):
            col[c, :, :] = vis_t.T  # broadcast to every correlation
        tb.putcol("CORRECTED_DATA", col)
        tb.close()

        os.system(f"rm -rf {self._dirty_path}")
        self.im.makeimage(type="corrected", image=self._dirty_path)

        ia = _image_tool()
        ia.open(self._dirty_path)
        arr = ia.getchunk()  # (nx, ny, nstokes, nchan)
        ia.close()
        # CASA (nx, ny, 1, nz) -> cube (nz, ny, nx)
        return np.transpose(arr[:, :, 0, :], (2, 1, 0))

    def lipschitz_constant(self, n_iter=20, seed=0):
        """Power iteration on adjoint_cube(forward_cube(.)) for the largest
        eigenvalue = Lipschitz constant of the data-fidelity gradient, exactly
        as `NonUniformFourierOperator.lipschitz_constant` does, but using
        these CASA-backed transforms."""
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(self.nchan, self.ny, self.nx))
        x /= np.linalg.norm(x)
        eig = 0.0
        for _ in range(n_iter):
            y = self.adjoint_cube(self.forward_cube(x))
            eig = float(np.linalg.norm(y))
            if eig < 1e-300:
                break
            x = y / eig
        return eig

    def close(self):
        self.im.close()
        for p in (self._model_path, self._dirty_path):
            os.system(f"rm -rf {p}")
