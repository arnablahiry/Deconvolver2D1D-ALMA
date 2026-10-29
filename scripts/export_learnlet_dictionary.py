"""
Export the 2D filters of a trained learnlet (github.com/vicbonj/learnlet; Ramzi et al. 2020,
Bonjean et al. 2025) into dictionaries/<name>.npz, the format gpu_pd.Dict2D1D(..., dict_2d=<name>) reads.

The learnlet works on the starlet: at detail scale j it splits c_j into the starlet plane
w_j = c_j - H_j c_j and c_{j+1} = H_j c_j (H_j the B3 spline dilated by 2**j, exactly gpu_pd.starlet), then
  analysis   a_jk = A_jk (*) w_j                      k = 1..F, A_jk L2-normalized, dilated by 2**j
  synthesis  w_j  = w_j + sum_k S_jk (*) (a_jk - A_jk (*) w_j)   (exact for any subset of the k)
Only the linear filters A, S are used by the deconvolution; the learnt per-scale thresholds (MiniNet) are
stored for reference (threshold_multipliers, in units of the noise sigma) but not used.

    python scripts/export_learnlet_dictionary.py                       # ~/repos/learnlet, default weights
    python scripts/export_learnlet_dictionary.py --repo PATH --weights FILE --name learnlet_xyz

The file also records where it came from (meta), and the export checks the filters against the
learnlet's own modules on a random image.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--repo", default=os.path.expanduser("~/repos/learnlet"))
ap.add_argument("--weights", default="weights/LEARNLET_FINAL_64_5_sc5_True_hard.pth",
                help="relative to --repo; LEARNLET_FINAL_<filters>_<kernel>_sc<n_scales>_<exact_rec>_<thresh>.pth")
ap.add_argument("--name", default="learnlet", help="written to dictionaries/<name>.npz")
args = ap.parse_args()

wpath = os.path.join(args.repo, args.weights)
filters, kernel, sc, exact, thresh = os.path.basename(wpath)[len("LEARNLET_FINAL_"):-len(".pth")].split("_")
filters, kernel, n_scales = int(filters), int(kernel), int(sc[2:])
if exact != "True":
    sys.exit("only exact-reconstruction learnlets (…_True_…) give a dictionary with an exact inverse")

sys.path.insert(0, args.repo)
from learnlet.learnlet import Learnlet  # noqa: E402

net = Learnlet(n_scales=n_scales, kernel_size=kernel, filters=filters, exact_rec=True, thresh=thresh, pretrained=False)
net.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
net.eval()

with torch.no_grad():
    A = np.stack([c.weight.numpy()[:, 0] for c in net.convs_A])          # parametrized: L2-normalized
    S = np.stack([c.weight.numpy()[:, 0] for c in net.convs_S])
    smooth = np.stack([c.weight.numpy()[0, 0] for c in net.convs_starlet])
    thr = (net.mininet(net.k_init) * 5).numpy()

b3 = np.array([1., 4., 6., 4., 1.]) / 16.
if not np.allclose(smooth, np.outer(b3, b3)[None], atol=1e-7):
    sys.exit("this learnlet's smoothing kernel is not the B3 starlet: gpu_pd's split would not match it")

# check: the analysis / synthesis written as in gpu_pd reproduce the learnlet's own convolutions
from torch.nn import functional as F  # noqa: E402
x = torch.randn(1, 1, 96, 96)
with torch.no_grad():
    c, err = x, 0.0
    for j in range(n_scales - 1):
        s = net.convs_starlet[j](c)
        w, c = c - s, s
        p = (kernel - 1) // 2 * 2 ** j
        a = F.conv2d(F.pad(w, (p,) * 4, mode="reflect"), torch.from_numpy(A[j])[:, None], dilation=2 ** j)
        err = max(err, float((a - net.convs_A[j](w)).abs().max()))
        sa = F.conv2d(F.pad(a, (p,) * 4, mode="reflect"), torch.from_numpy(S[j])[:, None], dilation=2 ** j, groups=filters)
        err = max(err, float((sa - net.convs_S[j](a)).abs().max()))
print(f"max |difference| from the learnlet's own modules: {err:.1e}")

try:
    commit = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
meta = dict(kind="starlet_filterbank", source="learnlet (github.com/vicbonj/learnlet), trained on 10,000 ImageNet images",
            references="Ramzi et al. 2020; Bonjean et al. 2025", repo_commit=commit, weights=os.path.basename(wpath),
            filters=filters, kernel_size=kernel, trained_detail_scales=n_scales - 1, thresh=thresh,
            note="scale j filters are applied with dilation 2**j to the starlet plane w_j (B3 spline, reflect)")
os.makedirs(os.path.join(ROOT, "dictionaries"), exist_ok=True)
out = os.path.join(ROOT, "dictionaries", f"{args.name}.npz")
np.savez(out, analysis=A.astype(np.float32), synthesis=S.astype(np.float32),
         threshold_multipliers=thr.astype(np.float32), meta=json.dumps(meta))
print(f"wrote {out}: {n_scales - 1} detail scales x {filters} filters of {kernel}x{kernel}; "
      f"learnt threshold multipliers {np.round(thr, 3).tolist()}")
