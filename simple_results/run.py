"""
Runs the simplified pipeline (legacy/simple/) on NGC 7469, writing its
output here instead of into legacy/simple/data/.

Reuses legacy/simple/grid_with_casa.py and deconvolve.py unmodified -- this
script only points their output-path constants at simple_results/data/
before calling them, so legacy/simple/ stays the untouched reference copy.

Run from the repo root:
    python simple_results/run.py
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
LEGACY_SIMPLE = os.path.join(REPO_ROOT, "legacy", "simple")
OUT_DIR = os.path.join(THIS_DIR, "data")

sys.path.insert(0, LEGACY_SIMPLE)
os.chdir(REPO_ROOT)   # grid_with_casa.MS is relative to the repo root

import grid_with_casa
import deconvolve

# Point both scripts' output at simple_results/data/ instead of
# legacy/simple/data/ -- everything else (imaging config, algorithm
# parameters) is unchanged.
os.makedirs(OUT_DIR, exist_ok=True)
grid_with_casa.OUT_DIR = OUT_DIR
grid_with_casa.IMAGE_STEM = os.path.join(OUT_DIR, "casa")
deconvolve.PSF_PATH = os.path.join(OUT_DIR, "psf.npy")
deconvolve.DIRTY_PATH = os.path.join(OUT_DIR, "dirty.npy")
deconvolve.OUT_PATH = os.path.join(OUT_DIR, "model.npy")

if __name__ == "__main__":
    grid_with_casa.grid_psf_and_dirty()
    deconvolve.main()
