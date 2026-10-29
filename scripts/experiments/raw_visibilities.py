#!/usr/bin/env python
"""
NGC 3110's raw line visibilities, per baseline and integration, for both arrays (step 1 of the visibility comparison;
raw_visibility_compare.py makes the figure). From each calibrated MS (DATA column, CO(2-1) spw, target field,
cross-correlations, flags applied, the two polarisations averaged):

  line   sum over the channels in the line window (LSRK radio 4748.9-5273.9 km/s, the 21 line channels of the
         cubes) of (V - continuum) x the channel width in km/s -> Jy km/s
  null   the same over an equally wide window without line (5400-5925 km/s), continuum from its own sides: pure noise
  continuum  the mean of the line-free channels (4450-4700 and 5300-5400 km/s) -> Jy

The LSRK window edges are converted to TOPO frequency at each MS's mean time (casatools measures; the drift over a
track is < 1 km/s, far below the 525 km/s window). u, v in klambda at the line frequency.

    python scripts/experiments/raw_visibilities.py      -> data/ngc3110/raw_visibilities.npz (cached)
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
from msclean_cube import GALAXIES, RESTFREQ_GHZ  # noqa: E402
from casa_logs import redirect_casa_logs  # noqa: E402

redirect_casa_logs("raw_visibilities_ngc3110.log")
import casatools  # noqa: E402

C = 299792.458
F0 = RESTFREQ_GHZ * 1e9
LINE, NULL = (4748.9, 5273.9), (5400.0, 5925.0)
CONT = ((4450.0, 4700.0), (5300.0, 5400.0))
NULL_CONT = ((5320.0, 5390.0), (5935.0, 6050.0))
OUT = os.path.join(REPO, "data/ngc3110/raw_visibilities.npz")
g = GALAXIES["ngc3110"]
tb, md, me, qa = casatools.table(), casatools.msmetadata(), casatools.measures(), casatools.quanta()


def topo(ms, fid, vel_kms):
    """TOPO frequency (Hz) of LSRK radio velocity vel_kms at the MS's mean time, towards the target"""
    md.open(ms)
    t = np.mean(md.timesforfield(fid)); d = md.phasecenter(fid); md.close()
    me.doframe(me.epoch("UTC", qa.quantity(t, "s"))); me.doframe(me.observatory("ALMA")); me.doframe(d)
    f_lsrk = F0 * (1 - vel_kms / C)
    return me.measure(me.frequency("LSRK", qa.quantity(f_lsrk, "Hz")), "TOPO")["m0"]["value"]


def window(freqs, ms, fid, v1, v2):
    fa, fb = sorted((topo(ms, fid, v1), topo(ms, fid, v2)))
    return (freqs >= fa) & (freqs <= fb)


out = {}
for cfg in ("compact", "extended"):
    rows = []
    for ms in g["configs"][cfg]:
        md.open(ms); fid = int(md.fieldsforname(g["field"])[0]); freqs = md.chanfreqs(25); md.close()
        tb.open(ms + "/DATA_DESCRIPTION"); dd = int(np.nonzero(tb.getcol("SPECTRAL_WINDOW_ID") == 25)[0][0]); tb.close()
        wl = window(freqs, ms, fid, *LINE); wn = window(freqs, ms, fid, *NULL)
        wc = np.any([window(freqs, ms, fid, *w) for w in CONT], axis=0)
        wnc = np.any([window(freqs, ms, fid, *w) for w in NULL_CONT], axis=0)
        dv = C * np.abs(np.median(np.diff(freqs))) / F0                     # km/s per channel
        lam = C * 1e3 / np.mean(freqs[wl])                                  # m
        print(f"{cfg} {os.path.basename(ms)[:28]}: field {fid}, dd {dd}, line {wl.sum()} ch, null {wn.sum()}, cont {wc.sum()} "
              f"(+{wnc.sum()}) ch of {dv:.3f} km/s", flush=True)
        tb.open(ms); t = tb.query(f"DATA_DESC_ID=={dd} && FIELD_ID=={fid} && ANTENNA1!=ANTENNA2")
        n = t.nrows(); step = 20000
        for r0 in range(0, n, step):
            nr = min(step, n - r0)
            d = t.getcol("DATA", r0, nr); fl = t.getcol("FLAG", r0, nr); w = t.getcol("WEIGHT", r0, nr); uvw = t.getcol("UVW", r0, nr)
            d = np.where(fl, np.nan, d).mean(0)                              # (chan, row), pols averaged
            ww = w.sum(0)

            def band(m):
                return np.nanmean(d[m], axis=0)

            cont = band(wc); ncont = band(wnc)
            line = np.nansum(d[wl] - cont[None], axis=0) * dv
            null = np.nansum(d[wn] - ncont[None], axis=0) * dv
            good = np.isfinite(line) & np.isfinite(null) & (ww > 0) & ~fl[:, wl].all((0, 1))
            rows.append(np.stack([uvw[0][good] / lam / 1e3, uvw[1][good] / lam / 1e3, line[good].real, line[good].imag,
                                  null[good].real, null[good].imag, cont[good].real, cont[good].imag, ww[good]]))
        t.close(); tb.close()
    out[cfg] = np.concatenate(rows, axis=1).astype(np.float32)
    print(f"{cfg}: {out[cfg].shape[1]} visibilities", flush=True)
np.savez_compressed(OUT, **out, columns=np.array(["u_kl", "v_kl", "line_re", "line_im", "null_re", "null_im", "cont_re",
                                                    "cont_im", "weight"]))
print("saved", OUT, flush=True)
