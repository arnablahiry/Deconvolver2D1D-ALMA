#!/usr/bin/env python
"""
Experiment 22, equal false-detection calibration: the k at which each learnlet dictionary puts the same spurious flux
into a pure-noise cube (null test, run_model.py --null) as the starlet does at k = 5, the pipeline's setting.
Spurious flux = total model flux of the null solve; k between the null runs' values by linear interpolation of
log(flux) in k (linear extrapolation from the two nearest points if outside, flagged).

    learnlet_calibrate.py   (reads data/experiments/22_learnlet/null/<tag>_k<k>.json)
    -> data/experiments/22_learnlet/calibration.json  {"learnlet16": k, "learnlet": k, ...}
"""
import glob
import json
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(REPO, "data/experiments/22_learnlet")
REF_K = 5.0


def points(tag):
    out = {}
    for f in glob.glob(os.path.join(D, "null", f"{tag}_k*.json")):
        m = re.search(r"_k([0-9.]+)\.json$", f)
        if m:
            out[float(m.group(1))] = json.load(open(f))["flux"]
    return dict(sorted(out.items()))


ref = points("starlet")
F0 = ref[REF_K]
cal = dict(reference=f"starlet, k = {REF_K:g}", target_flux=F0, starlet=REF_K, points=dict(starlet=ref), extrapolated={})
tags = sorted({re.sub(r"_k[0-9.]+\.json$", "", os.path.basename(f)) for f in glob.glob(os.path.join(D, "null", "*_k*.json"))}
              - {"starlet"})
for tag in tags:                                              # every dictionary with null runs at >= 2 values of k
    pts = points(tag)
    if len(pts) < 2:
        continue
    cal["points"][tag] = pts
    k = np.array(list(pts)); y = np.log(np.maximum(np.array(list(pts.values())), 1e-6))
    t = np.log(max(F0, 1e-6))
    inside = [i for i in range(len(k) - 1) if (y[i] - t) * (y[i + 1] - t) <= 0]
    i = inside[0] if inside else (0 if abs(y[0] - t) < abs(y[-1] - t) else len(k) - 2)
    kc = float(k[i] + (t - y[i]) * (k[i + 1] - k[i]) / (y[i + 1] - y[i])) if y[i + 1] != y[i] else float(k[i])
    cal[tag] = round(float(np.clip(kc, 1.5, 10.0)), 2)
    cal["extrapolated"][tag] = not inside
    print(f"{tag}: null flux {pts} -> k = {cal[tag]} (target {F0:.3f} Jy{', EXTRAPOLATED' if not inside else ''})")
json.dump(cal, open(os.path.join(D, "calibration.json"), "w"), indent=1)
print("wrote", os.path.join(D, "calibration.json"))
