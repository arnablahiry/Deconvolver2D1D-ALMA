#!/bin/bash
# Waits for the 1x FIRE cubes (GalPropML-IFU/shared/rebin_v2.py --products 1x20 -> /nas/datasets/FIRE_IFU_cubes/1x_new),
# then: ALMA-like preprocessing from 1x (data/fire_cii_1x) + its moment-0 grids (results/gallery/*_1x), the learnlet
# training set (data/fire_cii_learnlet) + its preview grid, and a 4x-vs-1x consistency check. Only the 42 cubes used
# are waited for (no C1: duplicate of C2; no snap 76: near-duplicate of 75). Log: logs/fire_1x_pipeline.log
#
#     nohup bash scripts/fire_1x_pipeline.sh &
REPO=/nas/homes/alahiry/repos/Deconvolver2D1D-ALMA
PY=/scratch/alahiry/conda/envs/deconvolver2d1d/bin/python
SRC=/nas/datasets/FIRE_IFU_cubes/1x_new
LOG=$REPO/logs/fire_1x_pipeline.log
cd $REPO || exit 1
mkdir -p logs
say() { echo "[$(TZ=Europe/Athens date '+%Y-%m-%d %H:%M')] $*" | tee -a $LOG; }
missing() { n=0; for g in A1 A2 A4 A8 B1 B2 C2; do for s in 60 65 70 75 80 85; do
    [ -f $SRC/$g/${g}_snap_${s}_1x_20s.hdf5 ] || n=$((n + 1)); done; done; echo $n; }

say "waiting for the 1x cubes ($(missing) of 42 missing)"
while [ "$(missing)" -gt 0 ]; do
    if ! pgrep -f "rebin_v2.py --products 1x20" > /dev/null; then
        sleep 60
        if [ "$(missing)" -gt 0 ]; then say "FAILED: rebin_v2 1x stopped with $(missing) of 42 cubes missing"; exit 2; fi
    fi
    sleep 300
    say "  $(missing) of 42 missing"
done
say "all 42 cubes present; starting"

run() { say "RUN $*"; "$@" >> $LOG 2>&1 || { say "FAILED ($?): $*"; exit 3; }; }
run $PY scripts/fire_preprocess.py --src $SRC --tag 1x --out data/fire_cii_1x --workers 6
for st in conv sky; do run env STAGE=$st D=data/fire_cii_1x SUFFIX=_1x $PY scripts/fire_moment0_grid.py; done
run $PY scripts/fire_learnlet_dataset.py --src $SRC --tag 1x --workers 4
run env STAGE=native $PY scripts/fire_moment0_grid.py
say "4x vs 1x consistency (same 0.02\" products):"
$PY - >> $LOG 2>&1 <<'EOF'
import json, numpy as np
a = {(c["gal"], c["snap"], c["los"]): c for c in json.load(open("data/fire_cii/index.json"))["cubes"]}
b = {(c["gal"], c["snap"], c["los"]): c for c in json.load(open("data/fire_cii_1x/index.json"))["cubes"]}
fr, md = [], []
for k in sorted(set(a) & set(b)):
    fr.append(b[k]["flux_total_jykms"] / a[k]["flux_total_jykms"] - 1)
    x, y = (np.load(f"data/{d}/{k[0]}_s{k[1]}_los{k[2]}.npz")["conv"].sum(0) for d in ("fire_cii", "fire_cii_1x"))
    md.append(np.linalg.norm(y - x) / np.linalg.norm(x))
print(f"  {len(fr)} cubes: total flux 1x/4x - 1: median {np.median(fr):+.2e}, max |.| {np.max(np.abs(fr)):.2e}; "
      f"convolved moment 0 ||1x - 4x|| / ||4x||: median {np.median(md):.2e}, max {np.max(md):.2e}")
EOF
say "DONE"
