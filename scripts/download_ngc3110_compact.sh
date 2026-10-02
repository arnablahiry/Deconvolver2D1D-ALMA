#!/usr/bin/env bash
# Download + extract the EU ARC CalMS delivery for NGC 3110, 12m compact, MOUS uid://A001/X1284/X1373 (2 EBs, ~11.8 GB).
# Resumable: re-run after an interruption and wget -c picks up where it stopped.
set -euo pipefail

BASE="https://almascience.eso.org/arcdistribution/mydir/c180aca4bcab079378743c39151c1f7d"
DEST="/nas/datasets/ALMA_visibilities/gal1/NGC3110_12m_compact_X1373"
MS_TGZ=(uid___A002_Xcf92df_X4e8e.ms.split.cal.tgz uid___A002_Xd09670_X3f8d.ms.split.cal.tgz)

mkdir -p "$DEST"
cd "$DEST"

for f in _README.txt makecalms-out.txt "${MS_TGZ[@]}" output_assess_ms3_public_uid___A001_X1284_X1373.tgz; do
    wget -c --no-verbose --show-progress --tries=20 --retry-connrefused --timeout=60 "$BASE/$f"
done

for f in "${MS_TGZ[@]}"; do
    gzip -t "$f"                 # fail loudly on a truncated download before extracting
    tar -xzf "$f"
done

ls -d *.ms.split.cal
