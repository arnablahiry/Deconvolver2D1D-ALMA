#!/usr/bin/env python
"""
Re-extend the thresholds of a shared-filter learnlet dictionary (train_learnlet_fire.py --shared --export-scales)
past its trained scales in units of each starlet plane's noise (t_j = t_last / ||psi_last|| x ||psi_j||) instead of
holding the last pixel-unit threshold, which made scale j's threshold grow 2x per scale relative to its noise (exports
before 2026-10-22). Filters untouched; the file is rewritten in place (meta.threshold_extension = "plane").

    python scripts/fix_learnlet_threshold_extension.py learnlet_fire_shared4 learnlet_fire_shared6
"""
import json
import os
import sys

import numpy as np

sys.argv, names = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_learnlet_fire as T  # noqa: E402

for n in names:
    path = os.path.join(T.REPO, "dictionaries", n + ".npz")
    z = dict(np.load(path)); meta = json.loads(str(z["meta"]))
    nt, t = meta["trained_detail_scales"], z["threshold_multipliers"].astype(np.float64)
    if meta.get("threshold_extension") == "plane" or len(t) <= nt:
        print(n, "nothing to do"); continue
    u = T.plane_noise(len(t))
    t[nt:] = t[nt - 1] / u[nt - 1] * u[nt:]
    meta["threshold_extension"] = "plane"
    z["threshold_multipliers"] = t.astype(np.float32); z["meta"] = json.dumps(meta)
    np.savez(path, **z)
    print(n, "thresholds (pixel sigma):", np.round(t, 4).tolist(), "per plane sigma:", np.round(t / u, 2).tolist())
