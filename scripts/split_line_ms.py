#!/usr/bin/env python
"""
Stage 0 of the real-ALMA pipeline: cut the CO(2-1) line channels out of the
55 GB `calibrated_final.ms.contsub` into a small working MS.

Why this exists
---------------
Every later stage (PSF, dirty cube, and -- if you use the exact operator --
*every FISTA iteration*) has to make a full pass over the visibilities. On
the 55 GB parent MS (1,248,030 rows x 4 spw x 1920 chan) a single pass is
minutes; on the ~300 MHz line window it is seconds. Nothing is lost: the
continuum is already subtracted, so the other ~7.2 GHz of bandwidth in this
MS contains no signal for this line.

Line identification (measured, not assumed)
-------------------------------------------
A vector average of the short-baseline (<300 m) visibilities of field 0 puts
the only coherent signal in the MS at 226.74-226.90 GHz, consistently in all
four spectral windows. With CO(2-1) at 230.538 GHz rest that is z ~= 0.0165,
i.e. this is the CO(2-1) line of IRAS F23007+0836 (NGC 7469).

Spectral-window layout matters here
-----------------------------------
    spw 0:  226.9039 -> 225.0299 GHz   (1920 chan, 1.291 km/s)
    spw 2:  226.9152 -> 225.0411 GHz   (the same tuning, second execution)
    spw 1:  228.6014 -> 226.7274 GHz
    spw 3:  228.6124 -> 226.7383 GHz

The line sits in the *overlap* between the two tunings. spw 0 and 2 contain
it entirely; spw 1 and 3 cut off at 226.727 GHz and so clip the red wing.
Combining all four therefore gives a channel-dependent visibility count and
hence a channel-dependent noise level -- which is exactly what the MAD-based
per-sub-band thresholding in the deconvolver assumes away. All four are
split out here so the choice stays open, but the imaging default is spw 0,2.
"""

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Frequency window kept, generous enough to leave line-free channels at both
# edges for noise estimation (the line itself spans ~226.74-226.90 GHz).
FREQ_LO_GHZ = 226.62
FREQ_HI_GHZ = 226.98


def channel_range(chanfreqs_hz, lo_ghz, hi_ghz):
    """Inclusive channel index range of `chanfreqs` inside [lo, hi] GHz.

    Returns None if the window does not intersect the spw at all. Handles
    descending frequency axes (all four spws here are descending).
    """
    f = np.asarray(chanfreqs_hz) / 1e9
    inside = np.nonzero((f >= lo_ghz) & (f <= hi_ghz))[0]
    if inside.size == 0:
        return None
    return int(inside.min()), int(inside.max())


def build_spw_selection(msname, lo_ghz, hi_ghz, spws=None):
    """CASA `spw` selection string covering [lo, hi] GHz, e.g. '0:0~265,2:0~275'."""
    from casatools import msmetadata

    md = msmetadata()
    md.open(msname)
    try:
        all_spws = range(md.nspw()) if spws is None else spws
        parts = []
        for s in all_spws:
            rng = channel_range(md.chanfreqs(s), lo_ghz, hi_ghz)
            if rng is None:
                continue
            parts.append(f"{s}:{rng[0]}~{rng[1]}")
    finally:
        md.close()
    if not parts:
        raise RuntimeError(f"No spw covers {lo_ghz}-{hi_ghz} GHz in {msname}")
    return ",".join(parts)


def split_line(msname, outputms, lo_ghz=FREQ_LO_GHZ, hi_ghz=FREQ_HI_GHZ,
               datacolumn="DATA", overwrite=False):
    from casatools import mstransformer

    msname = os.path.abspath(msname)
    outputms = os.path.abspath(outputms)
    if os.path.exists(outputms):
        if not overwrite:
            print(f"[split] {outputms} already exists, skipping (pass "
                  f"--overwrite to regenerate it)")
            return outputms
        shutil.rmtree(outputms)

    spwsel = build_spw_selection(msname, lo_ghz, hi_ghz)
    print(f"[split] {msname}")
    print(f"[split] window {lo_ghz}-{hi_ghz} GHz -> spw selection {spwsel!r}")

    # `ms.split` only applies row selection -- its `spw` argument does not
    # honour the `spw:chan~chan` syntax, so it copies all 1920 channels back
    # out. `mstransformer` (the engine behind the `mstransform`/`split` tasks)
    # does apply the channel selection.
    mt = mstransformer()
    try:
        mt.config({
            "inputms": msname,
            "outputms": outputms,
            "datacolumn": datacolumn,
            "spw": spwsel,
            "reindex": True,
            "keepflags": True,
            # The two tunings (spw 0/2 vs 1/3) are on different frequency
            # grids; leave them as separate spws and let the imager regrid
            # them onto the common velocity axis.
            "combinespws": False,
        })
        mt.open()
        mt.run()
    finally:
        mt.done()

    report(outputms)
    return outputms


def report(msname):
    from casatools import msmetadata, table

    md = msmetadata()
    md.open(msname)
    print(f"[split] output {msname}")
    print(f"[split]   fields {md.fieldnames()}  nspw {md.nspw()}")
    for s in range(md.nspw()):
        f = md.chanfreqs(s) / 1e9
        print(f"[split]   spw {s}: {f[0]:.4f} -> {f[-1]:.4f} GHz, {len(f)} chan")
    md.close()
    tb = table()
    tb.open(msname)
    print(f"[split]   rows {tb.nrows()}")
    tb.close()
    du = sum(os.path.getsize(os.path.join(r, x))
             for r, _, fs in os.walk(msname) for x in fs)
    print(f"[split]   size {du / 1e9:.2f} GB")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vis", default="data/calibrated_final.ms.contsub")
    p.add_argument("--out", default="data/ngc7469_co21.ms")
    p.add_argument("--lo-ghz", type=float, default=FREQ_LO_GHZ)
    p.add_argument("--hi-ghz", type=float, default=FREQ_HI_GHZ)
    p.add_argument("--datacolumn", default="DATA")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    split_line(a.vis, a.out, a.lo_ghz, a.hi_ghz, a.datacolumn, a.overwrite)


if __name__ == "__main__":
    main()
