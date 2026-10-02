#!/usr/bin/env bash
# Download + extract the EU ARC CalMS delivery for NGC 2369, 12m compact, MOUS uid://A001/X1284/X136d (1 EB, ~5.5 GB).
# Resumable: re-run after an interruption and wget -c picks up where it stopped.
set -euo pipefail

BASE="https://almascience.eso.org/arcdistribution/mydir/4eede91ed3d456f2d715de8804114df3"
DEST="/nas/datasets/ALMA_visibilities/gal2/NGC2369_12m_compact_X136d"
MS_TGZ=(uid___A002_Xcd07af_X20e.ms.split.cal.tgz)

mkdir -p "$DEST"
cd "$DEST"

for f in _README.txt makecalms-out.txt "${MS_TGZ[@]}" output_assess_ms3_public_uid___A001_X1284_X136d.tgz; do
    wget -c --no-verbose --show-progress --tries=20 --retry-connrefused --timeout=60 "$BASE/$f"
done

for f in "${MS_TGZ[@]}"; do
    gzip -t "$f"                 # fail loudly on a truncated download before extracting
    tar -xzf "$f"
done

ls -d *.ms.split.cal
