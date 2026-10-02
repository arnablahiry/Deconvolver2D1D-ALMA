"""3 columns (dirty, raw deconvolved, deconvolved * extended beam) x 2 rows (compact, extended).
All panels in surface brightness Jy km/s arcsec^-2 so the configurations share each column's scale."""
import os
# ---- configuration (environment variables), see scripts/superres_figures/README.md
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPACT_MODEL = os.environ.get("COMPACT_MODEL", os.path.join(REPO, "data/ngc3110/compact/pd/model.npy"))
EXTENDED_MODEL = os.environ.get("EXTENDED_MODEL", os.path.join(REPO, "data/ngc3110/extended/pd/model.npy"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(REPO, "data/ngc3110/compact/debug"))
COMPACT_TITLE = "Compact (" + os.environ.get("COMPACT_LABEL", "deconvolved") + ")"
os.makedirs(OUT_DIR, exist_ok=True)
import sys, numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, os.path.join(REPO, 'simple'))
from restore import convolve_with_beam
OUT = os.path.join(OUT_DIR, '44_dirty_raw_restored.png')
CELL = 0.02; BE = (0.220, 0.172, -74.7); BC = (1.418, 0.976, 83.5)
vel = 5073.9 + 25 * (np.arange(32) - 15.5); L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
FS = 22
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'font.size': FS,
                     'axes.labelsize': FS, 'xtick.labelsize': FS - 2, 'ytick.labelsize': FS - 2,
                     'axes.linewidth': 1.4, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 6, 'ytick.major.size': 6,
                     'xtick.major.pad': 10, 'ytick.major.pad': 10})
UNIT = r'Jy km s$^{-1}$ arcsec$^{-2}$'
XLAB, YLAB = r'$\Delta$RA [$^{\prime\prime}$]', r'$\Delta$Dec [$^{\prime\prime}$]'
barea = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
h = 250; s = np.s_[400 - h:400 + h, 400 - h:400 + h]; ext = [5, -5, -5, 5]
m0 = lambda c: c[L].sum(0)[s] * 25

xc = np.load(COMPACT_MODEL).astype(np.float64); xe = np.load(EXTENDED_MODEL).astype(np.float64)
ROWS = [('Compact', m0(np.nan_to_num(fits.getdata(os.path.join(REPO, 'data/ngc3110/compact/cube/dirty_cube.fits')).astype(np.float64)).squeeze()) / barea(BC), m0(xc) / CELL ** 2, m0(convolve_with_beam(xc, CELL, *BE)) / barea(BE), BC),
        ('Extended', m0(np.nan_to_num(fits.getdata(os.path.join(REPO, 'data/ngc3110/extended/cube/dirty_cube.fits')).astype(np.float64)).squeeze()) / barea(BE), m0(xe) / CELL ** 2, m0(convolve_with_beam(xe, CELL, *BE)) / barea(BE), BE)]
vmax = [max(r[k].max() for r in ROWS) for k in (1, 2, 3)]

P, g, CBW = 7.0, 0.55, 0.32
f0 = plt.figure(); a = f0.add_axes([0.3, 0.3, 0.4, 0.4]); a.set_xlim(5, -5); a.set_xlabel(XLAB)
c = f0.add_axes([0.8, 0.3, 0.03, 0.4]); c.set_ylim(0, 115); c.yaxis.tick_right(); c.yaxis.set_label_position('right'); c.set_ylabel(UNIT)
c.tick_params(direction='out', left=False, right=True); f0.canvas.draw(); r_ = f0.canvas.get_renderer()
XL = 0.3 * f0.get_figheight() - a.get_tightbbox(r_).y0 / f0.dpi
CBL = c.get_tightbbox(r_).x1 / f0.dpi - 0.83 * f0.get_figwidth(); plt.close(f0)
LM, RM, TM, BM = 2.6, 0.2, 1.35, XL + 0.2
colw = P + g + CBW + CBL                          # one map + its colorbar
W = LM + 3 * colw + 2 * g + RM
H = TM + 2 * P + g + XL + BM
fig = plt.figure(figsize=(W, H))
ax_at = lambda x, y, w, hh, **kw: fig.add_axes([x / W, y / H, w / W, hh / H], **kw)
def frame(a):
    for sp in a.spines.values(): sp.set_visible(True)
def beam(a, b, col, label=None):
    big = b[0] > 1
    x0, y0 = (3.1, -3.4) if big else (3.15, -3.9)
    a.add_patch(Ellipse((x0, y0), width=b[0], height=b[1], angle=90 - b[2], fc='none', ec=col, lw=2))
    a.text(x0, y0 - (0.85 if big else 0.35), label or rf'{b[0]:.2f}$^{{\prime\prime}}$ $\times$ {b[1]:.2f}$^{{\prime\prime}}$',
           color=col, ha='center', va='top', fontsize=FS - 4, fontweight='bold')

y2 = BM; y1 = y2 + P + XL + g
xs = [LM + k * (colw + g) for k in range(3)]
first = None
for r, (nm, dirty, raw, rest, bown) in enumerate(ROWS):
    y = (y1, y2)[r]
    for k, img in enumerate((dirty, raw, rest)):
        a = ax_at(xs[k], y, P, P, **({} if first is None else dict(sharex=first, sharey=first))); first = first or a
        im = a.imshow(img, origin='lower', extent=ext, cmap='inferno', vmin=img.min(), vmax=img.max(), interpolation='nearest')
        if k == 0: beam(a, bown, 'w')
        elif k == 2: beam(a, BE, 'w')
        else: a.text(0.04, 0.05, r'no beam (0.02$^{\prime\prime}$ pixels)', transform=a.transAxes, color='w', fontsize=FS - 4, fontweight='bold')
        frame(a); a.set_xlim(5, -5); a.set_ylim(-5, 5); a.set_xticks([4, 2, 0, -2, -4]); a.set_yticks([-4, -2, 0, 2, 4])
        a.set_xlabel(XLAB)
        if k > 0: plt.setp(a.get_yticklabels(), visible=False)
        else: a.set_ylabel(YLAB)
        cax = ax_at(xs[k] + P + g, y, CBW, P); cb = fig.colorbar(im, cax=cax); cb.set_label(UNIT); frame(cax)
        cax.tick_params(which='both', direction='out', left=False, right=True, labelright=True)
    fig.text((xs[0] - 1.9) / W, (y + P / 2) / H, nm, rotation=90, ha='center', va='center', fontsize=FS + 3, fontweight='bold')
for k, (l1, l2) in enumerate((('Dirty', r'$\ast$ own dirty beam'), ('Deconvolved (raw)', 'model, no convolution'),
                              ('Deconvolved', r'$\ast$ Extended clean beam'))):
    fig.text((xs[k] + P / 2) / W, (H - TM + 0.62) / H, l1, ha='center', va='bottom', fontsize=FS + 1, fontweight='bold')
    fig.text((xs[k] + P / 2) / W, (H - TM + 0.14) / H, l2, ha='center', va='bottom', fontsize=FS - 3, fontweight='bold')
fig.savefig(OUT, dpi=55)
print('saved; panel (min, max)', [[(round(v.min(), 2), round(v.max(), 1)) for v in r[1:4]] for r in ROWS])
