#!/usr/bin/env python
"""
Report what is actually inside one or more calibrated ALMA MSs.

Why this exists
---------------
The downloads under /nas/datasets/ALMA_visibilities are one *MOUS* each, and
a MOUS is one array configuration. Before assuming a galaxy has the compact +
extended pair you need for a combined image, you have to look: how many
science fields, which array (12m/7m/TP, from the dish diameter), what the
baseline range is, and what the spectral setup looks like. This prints all of
that per MS and, when given several, a side-by-side summary so two MSs of the
same target can be identified as two configurations -- or not.

Requires casatools (run in a CASA-enabled env).

Usage
-----
    python scripts/inspect_ms.py /path/to/a.ms /path/to/b.ms
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simple"))
try:
    from casa_logs import redirect_casa_logs
    redirect_casa_logs("inspect_ms.log")
except ImportError:
    pass  # the --help path, and any other casatools-free invocation

# Dish diameter -> ALMA array name. TP antennas are 12 m too, but they are
# single-dish: they show up with no baselines at all, which the caller sees
# from the baseline range rather than from the diameter.
_ARRAY_BY_DIAMETER = {12.0: "12m", 7.0: "7m (ACA)"}


def _array_name(diameters):
    """Label the array from the set of dish diameters present."""
    uniq = sorted({round(float(d), 1) for d in diameters})
    return "+".join(_ARRAY_BY_DIAMETER.get(d, f"{d:g}m") for d in uniq)


def baseline_extremes(ms_path):
    """(min, max) physical baseline length in metres, from ANTENNA positions.

    Read off the ANTENNA subtable rather than the UVW column: it is a few
    dozen rows instead of millions, and for a single configuration the
    antenna-pad geometry is what defines the resolution anyway.
    """
    from casatools import table

    tb = table()
    tb.open(os.path.join(ms_path, "ANTENNA"))
    try:
        pos = tb.getcol("POSITION").T          # (nant, 3), ITRF metres
        diam = tb.getcol("DISH_DIAMETER")
        names = tb.getcol("NAME")
        stations = tb.getcol("STATION")
    finally:
        tb.close()

    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    offdiag = d[~np.eye(len(pos), dtype=bool)]
    if offdiag.size == 0:                      # single dish
        return 0.0, 0.0, diam, names, stations
    return float(offdiag.min()), float(offdiag.max()), diam, names, stations


def describe(ms_path):
    """Collect the identifying facts about one MS into a dict."""
    from casatools import msmetadata

    md = msmetadata()
    md.open(ms_path)
    try:
        fields = list(md.fieldnames())
        try:
            targets = [fields[i] for i in md.fieldsforintent("OBSERVE_TARGET#ON_SOURCE")]
        except Exception:
            targets = []
        # Science spws: the wide correlator windows, excluding WVR and the
        # square-law/channel-average windows the pipeline leaves behind.
        try:
            sci_spws = list(md.spwsforintent("OBSERVE_TARGET#ON_SOURCE"))
        except Exception:
            sci_spws = list(range(md.nspw()))

        spws = []
        for s in sci_spws:
            freqs = md.chanfreqs(s) / 1e9       # GHz
            if freqs.size < 2:                  # WVR / single-channel window
                continue
            widths = md.chanwidths(s) / 1e3     # kHz
            spws.append(
                dict(
                    spw=int(s),
                    nchan=int(freqs.size),
                    f_lo=float(min(freqs)),
                    f_hi=float(max(freqs)),
                    chanwidth_khz=float(np.mean(np.abs(widths))),
                )
            )

        nant = int(md.nantennas())
        times = md.timesforfield(md.fieldsforname(targets[0])[0]) if targets else None
        obs_s = float(times.max() - times.min()) if times is not None and times.size else float("nan")
    finally:
        md.close()

    bl_min, bl_max, diam, ant_names, stations = baseline_extremes(ms_path)

    return dict(
        path=ms_path,
        fields=fields,
        targets=targets,
        spws=spws,
        nant=nant,
        array=_array_name(diam),
        bl_min=bl_min,
        bl_max=bl_max,
        elapsed_s=obs_s,
        stations=sorted(set(stations)),
    )


def resolution_arcsec(bl_max_m, freq_ghz):
    """Rough synthesised beam: lambda / B_max, in arcsec.

    This is the diffraction limit of the longest baseline, not the fitted
    beam of a weighted image -- good enough to tell a compact configuration
    from an extended one, which is all it is used for here.
    """
    lam = 2.99792458e8 / (freq_ghz * 1e9)
    return np.degrees(lam / bl_max_m) * 3600.0


def report(info):
    print(f"\n=== {info['path']} ===")
    print(f"  array            : {info['array']}, {info['nant']} antennas")
    print(f"  fields           : {', '.join(info['fields'])}")
    print(f"  science target(s): {', '.join(info['targets']) or '(none flagged as target)'}")
    print(f"  baselines        : {info['bl_min']:.1f} - {info['bl_max']:.1f} m")
    if info["spws"]:
        fref = 0.5 * (info["spws"][0]["f_lo"] + info["spws"][0]["f_hi"])
        if info["bl_max"] > 0:
            print(f"  approx lambda/B  : {resolution_arcsec(info['bl_max'], fref):.3f} arcsec "
                  f"at {fref:.2f} GHz")
    print(f"  elapsed on target: {info['elapsed_s'] / 60.0:.1f} min")
    print(f"  science spws     :")
    for s in info["spws"]:
        print(f"      spw {s['spw']:>3}  {s['nchan']:>5} chan  "
              f"{s['f_lo']:.4f} - {s['f_hi']:.4f} GHz  "
              f"({s['chanwidth_khz']:.1f} kHz/chan)")


def summarise(infos):
    """Group the MSs by science target so missing configurations are obvious."""
    print("\n\n=== summary: configurations per target ===")
    by_target = {}
    for info in infos:
        for t in info["targets"] or ["(unknown)"]:
            by_target.setdefault(t, []).append(info)

    for target, group in sorted(by_target.items()):
        print(f"\n  {target}: {len(group)} configuration(s)")
        for info in group:
            print(f"      {info['array']:<10} B = {info['bl_min']:6.1f} - {info['bl_max']:7.1f} m"
                  f"   {os.path.basename(info['path'])}")
        if len(group) < 2:
            print("      ^ only one configuration -- no compact/extended pair to combine")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ms", nargs="+", help="MS directories to inspect")
    args = p.parse_args(argv)

    missing = [m for m in args.ms if not os.path.isdir(m)]
    if missing:
        p.error("not a directory: " + ", ".join(missing))

    infos = [describe(m) for m in args.ms]
    for info in infos:
        report(info)
    summarise(infos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
