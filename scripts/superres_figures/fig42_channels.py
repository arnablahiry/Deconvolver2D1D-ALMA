import os
# ---- configuration (environment variables), see scripts/superres_figures/README.md
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPACT_MODEL = os.environ.get("COMPACT_MODEL", os.path.join(REPO, "data/ngc3110/compact/pd/model.npy"))
EXTENDED_MODEL = os.environ.get("EXTENDED_MODEL", os.path.join(REPO, "data/ngc3110/extended/pd/model.npy"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(REPO, "data/ngc3110/compact/debug"))
COMPACT_TITLE = "Compact (" + os.environ.get("COMPACT_LABEL", "deconvolved") + ")"
os.makedirs(OUT_DIR, exist_ok=True)
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, os.path.join(REPO, 'simple'))
from restore import convolve_with_beam
OUT = os.path.join(OUT_DIR, '42_superres_channels.png')
CELL = 0.02; BE = (0.220, 0.172, -74.7); BC = (1.418, 0.976, 83.5)
vel = 5073.9 + 25 * (np.arange(32) - 15.5)
CH = [6, 9, 12, 15, 18]
FS = 22
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'font.size': FS,
                     'axes.labelsize': FS, 'xtick.labelsize': FS - 2, 'ytick.labelsize': FS - 2,
                     'axes.linewidth': 1.4, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 6, 'ytick.major.size': 6,
                     'xtick.major.pad': 10, 'ytick.major.pad': 10})
UNIT = r'Jy arcsec$^{-2}$'
XLAB, YLAB = r'$\Delta$RA [$^{\prime\prime}$]', r'$\Delta$Dec [$^{\prime\prime}$]'

barea = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))                  # arcsec^2
xc = np.load(COMPACT_MODEL).astype(np.float64)[CH]; xe = np.load(EXTENDED_MODEL).astype(np.float64)[CH]
sb = lambda x, b: convolve_with_beam(x, CELL, *b) / barea(b)              # surface brightness, Jy/arcsec^2
Rc, Cc, Ec = sb(xc, BC), sb(xc, BE), sb(xe, BE)
h = 250; s = np.s_[:, 400 - h:400 + h, 400 - h:400 + h]; Rc, Cc, Ec = Rc[s], Cc[s], Ec[s]; ext = [5, -5, -5, 5]
vmax = max(Rc.max(), Cc.max(), Ec.max()); rmax = max(Cc.max(), Ec.max())

P, g, CBW = 4.6, 0.45, 0.28
ncol, nrow = len(CH), 4
LM, TM, BM, RM = 3.3, 1.3, 1.3, 2.2
W = LM + ncol * P + (ncol - 1) * g + g + CBW + RM
H = TM + nrow * P + (nrow - 1) * g + BM
fig = plt.figure(figsize=(W, H))
def ax_at(x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)
def frame(ax):
    for sp in ax.spines.values(): sp.set_visible(True)
def beam(ax, col, b):
    big = b[0] > 1
    x0, y0 = (2.9, -3.0) if big else (2.85, -3.6)
    ax.add_patch(Ellipse((x0, y0), width=b[0], height=b[1], angle=90 - b[2], fc='none', ec=col, lw=2))
    ax.text(x0, y0 - (1.0 if big else 0.45), f'{b[0]:.2f}$^{{\\prime\\prime}}$ $\\times$ {b[1]:.2f}$^{{\\prime\\prime}}$', color=col, ha='center', va='top', fontsize=FS - 6, fontweight='bold')

ys = [BM + (nrow - 1 - r) * (P + g) for r in range(nrow)]    # rows: compact*compact, compact*ext, extended, residual
first = None
for j, ch in enumerate(CH):
    x = LM + j * (P + g)
    rows = ((Rc[j], 'inferno', 0, vmax, 'w', BC), (Cc[j], 'inferno', 0, vmax, 'w', BE), (Ec[j], 'inferno', 0, vmax, 'w', BE),
            (Cc[j] - Ec[j], 'RdBu_r', -rmax, rmax, 'k', BE))
    for r, (img, cmap, lo, hi, col, b) in enumerate(rows):
        a = ax_at(x, ys[r], P, P, **({} if first is None else dict(sharex=first, sharey=first)))
        first = first or a
        im = a.imshow(img, origin='lower', extent=ext, cmap=cmap, vmin=lo, vmax=hi)
        if r == 0: im_map = im
        if r == nrow - 1:
            im_res = im
            a.text(0.05, 0.95, f'rms {img.std() / rmax:.1%} of peak', transform=a.transAxes, ha='left', va='top', fontsize=FS - 4, fontweight='bold')
        beam(a, col, b); frame(a); a.set_xlim(5, -5); a.set_ylim(-5, 5)
        a.set_xticks([4, 2, 0, -2, -4]); a.set_yticks([-4, -2, 0, 2, 4])
        if r < nrow - 1: plt.setp(a.get_xticklabels(), visible=False)
        else: a.set_xlabel(XLAB)
        if j > 0: plt.setp(a.get_yticklabels(), visible=False)
        else: a.set_ylabel(YLAB)
    fig.text((x + P / 2) / W, (H - TM + 0.55) / H, f'$v$ = {vel[ch]:.0f} km s$^{{-1}}$', ha='center', va='bottom', fontsize=FS + 1, fontweight='bold')
    fig.text((x + P / 2) / W, (H - TM + 0.12) / H, f'channel {ch}', ha='center', va='bottom', fontsize=FS - 3, fontweight='bold')
# row labels (left)
xl = (LM - 2.25) / W
LABELS = ((COMPACT_TITLE, r'$\ast$ Compact clean beam'), (COMPACT_TITLE, r'$\ast$ Extended clean beam'),
          ('Extended (deconvolved)', r'$\ast$ Extended clean beam'), ('Compact $-$ Extended', r'both $\ast$ extended beam'))
for r, (l1, l2) in enumerate(LABELS):
    yc = (ys[r] + P / 2) / H
    fig.text(xl - 0.28 / W, yc, l1, rotation=90, ha='center', va='center', fontsize=FS + 1, fontweight='bold')
    fig.text(xl + 0.2 / W, yc, l2, rotation=90, ha='center', va='center', fontsize=FS - 3, fontweight='bold')
# colorbars: maps (rows 1-3), residual (row 4)
xcb = LM + ncol * P + (ncol - 1) * g + g
for cax_y, cax_h, im in ((ys[2], 3 * P + 2 * g, im_map), (ys[3], P, im_res)):
    cax = ax_at(xcb, cax_y, CBW, cax_h); cb = fig.colorbar(im, cax=cax); cb.set_label(UNIT); frame(cax)
    cax.tick_params(which='both', direction='out', left=False, right=True, labelright=True)
fig.savefig(OUT, dpi=45, bbox_inches='tight', pad_inches=0.25)
print('saved; peaks', round(vmax, 2), round(rmax, 2), 'Jy/arcsec^2')
