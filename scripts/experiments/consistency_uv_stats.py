#!/usr/bin/env python
"""
Which galaxies should the compact-vs-extended consistency test pass most easily? For each target (results/<T>/config.json):
the (u, v, weight) of the target field's line spw for both arrays (cached in results/<T>/consistency/uv.npz, also used for
the figure), and
  f_ext_short   fraction of the extended weight at baselines shorter than the compact array's median (weighted) baseline:
                how much of the extended data sits where the compact beam's flux lives
  q5_ext        the extended array's 5th-percentile baseline (klambda) and its largest recoverable scale ~ 0.6 / q5
  r90           the radius (arcsec) holding 90% of the compact CLEAN flux (data/sample, earlier measurement)
A small r90 against the extended LRS and a large f_ext_short make the extended data able to reproduce the compact image.

    /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/consistency_uv_stats.py
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from casa_logs import redirect_casa_logs  # noqa: E402
redirect_casa_logs("consistency_uv_stats.log")
import casatools  # noqa: E402

C = 299792.458
T_ALL = ["NGC3110", "NGC2369", "ESO297-G011", "NGC1614", "IC4518E", "IC4518W", "IRASF17138-1017", "NGC7130", "IC5179",
         "NGC5135", "IC4687"]
R90 = {"NGC3110": 9.9, "NGC2369": 9.2, "ESO297-G011": 9.4, "NGC1614": 4.9, "IC4518E": 11.0, "IC4518W": 4.3,
       "IRASF17138-1017": 5.1, "NGC7130": 10.7, "IC5179": 11.4, "NGC5135": 6.3, "IC4687": 6.6}
tb, md = casatools.table(), casatools.msmetadata()


def uvw(T, cfg):
    c = json.load(open(os.path.join(REPO, "results", T, "config.json")))
    lam_m = C * 1e3 / (c["f_line_ghz"] * 1e9)
    us, vs, ws = [], [], []
    for ms, spw, field in zip(c["configs"][cfg]["ms"], c["configs"][cfg]["spw"], c["configs"][cfg]["field"]):
        sid = int(spw.split(":")[0])
        md.open(ms)
        fids = [int(f) for f in field.split(",")] if field.replace(",", "").isdigit() else \
               sorted({int(i) for n in field.split(",") for i in md.fieldsforname(n)})
        md.close()
        tb.open(ms + "/DATA_DESCRIPTION"); dd = int(np.nonzero(tb.getcol("SPECTRAL_WINDOW_ID") == sid)[0][0]); tb.close()
        tb.open(ms)
        q = tb.query(f"FIELD_ID IN [{','.join(map(str, fids))}] && DATA_DESC_ID=={dd} && ANTENNA1!=ANTENNA2 && !FLAG_ROW",
                     columns="UVW,WEIGHT")
        u = q.getcol("UVW"); w = q.getcol("WEIGHT").mean(0); q.close(); tb.close()
        us.append(u[0] / lam_m / 1e3); vs.append(u[1] / lam_m / 1e3); ws.append(w)
    return np.concatenate(us), np.concatenate(vs), np.concatenate(ws)


rows = []
for T in T_ALL:
    out = os.path.join(REPO, "results", T, "consistency"); os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "uv.npz")
    if os.path.exists(cache):
        z = np.load(cache); d = {k: (z[k + "_u"], z[k + "_v"], z[k + "_w"]) for k in ("compact", "extended")}
    else:
        d = {k: uvw(T, k) for k in ("compact", "extended")}
        np.savez_compressed(cache, **{f"{k}_{a}": d[k][i].astype(np.float32) for k in d for i, a in enumerate("uvw")})
    q = {k: np.hypot(d[k][0], d[k][1]) for k in d}
    wq = lambda k, p: float(np.interp(p, np.cumsum(d[k][2][np.argsort(q[k])]) / d[k][2].sum(), np.sort(q[k])))
    qc50 = wq("compact", 0.5)
    f_short = float(d["extended"][2][q["extended"] < qc50].sum() / d["extended"][2].sum())
    q5 = wq("extended", 0.05); lrs = 0.6 / (q5 * 1e3) * 206264.806
    rows.append(dict(T=T, q_compact_median_kl=qc50, f_ext_short=f_short, q5_ext_kl=q5, lrs_ext_arcsec=lrs, r90_arcsec=R90[T],
                     r90_over_lrs=R90[T] / lrs))
    print(f"{T:16s} compact median {qc50:5.0f} kl | extended weight below it {100 * f_short:5.1f}% | extended q5 {q5:5.0f} kl "
          f"(LRS {lrs:4.1f}\") | r90 {R90[T]:4.1f}\" -> r90/LRS {R90[T] / lrs:4.1f}", flush=True)
json.dump(rows, open(os.path.join(REPO, "results", "consistency_uv_stats.json"), "w"), indent=1)
