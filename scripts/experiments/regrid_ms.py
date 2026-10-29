#!/usr/bin/env python
"""
Pre-average a target's visibilities onto the imaging channel grid once (mstransform: target field, CO spw, LSRK radio
velocity channels of WIDTH_KMS starting where consistency_target.py's cube starts, NCHAN of them), so every later tclean
major cycle grids NCHAN channels instead of the spw's 960 native ones. Weights are averaged with the data, so natural
weighting is unchanged. Writes /scratch/alahiry/regrid/<T>_<NCHAN>x<WIDTH>kms/<config>_<i>.ms and ms.json (the MS lists
and spw "0", read by consistency_target.py via REGRID=<that folder>).

    NCHAN=64 WIDTH_KMS=20 /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/regrid_ms.py NGC3110
"""
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from casa_logs import redirect_casa_logs  # noqa: E402

T = sys.argv[1]
NCHAN, WIDTH = int(os.environ.get("NCHAN", 64)), float(os.environ.get("WIDTH_KMS", 20))
redirect_casa_logs(f"regrid_ms_{T}.log")
from casatasks import mstransform  # noqa: E402

GC = json.load(open(os.path.join(REPO, "results", T, "config.json")))
start = GC["vsys_kms"] - (NCHAN - 1) / 2 * WIDTH
OUT = f"/scratch/alahiry/regrid/{T}_{NCHAN}x{WIDTH:g}kms"; os.makedirs(OUT, exist_ok=True)
lists = {}
for cfg in ("compact", "extended"):
    cc = GC["configs"][cfg]; lists[cfg] = []
    for i, (ms, spw, field) in enumerate(zip(cc["ms"], cc["spw"], cc["field"])):
        out = os.path.join(OUT, f"{cfg}_{i}.ms")
        if not os.path.exists(out):
            t0 = time.time()
            ids = ",".join(p.split(":")[0] for p in spw.split(","))     # several windows (NGC 3256): combined
            kw = dict(vis=ms, outputvis=out + ".tmp", field=field, datacolumn="data", regridms=True, mode="velocity",
                      nchan=NCHAN, start=f"{start}km/s", width=f"{WIDTH}km/s", restfreq=f"{GC['restfreq_ghz']}GHz",
                      outframe="LSRK", veltype="radio", interpolation="linear", keepflags=True)
            try:
                mstransform(spw=ids, combinespws="," in ids, **kw)
            except RuntimeError as e:                    # windows of unequal channel counts cannot be combined (NGC
                if "," not in ids:                       # 5135: a 10 MHz sliver of a second window at the band edge)
                    raise
                print(f"{cfg} {os.path.basename(ms)}: cannot combine spws {ids} ({e}); using spw {ids.split(',')[0]}",
                      flush=True)
                shutil.rmtree(out + ".tmp", ignore_errors=True)
                mstransform(spw=ids.split(",")[0], combinespws=False, **kw)
            shutil.move(out + ".tmp", out)
            print(f"{cfg} {os.path.basename(ms)} -> {out}: {time.time() - start if False else time.time() - t0:.0f} s", flush=True)
        lists[cfg].append(out)
json.dump(dict(ms=lists, spw="0", nchan=NCHAN, width_kms=WIDTH, start_kms=start), open(os.path.join(OUT, "ms.json"), "w"), indent=1)
print("DONE", OUT, flush=True)
