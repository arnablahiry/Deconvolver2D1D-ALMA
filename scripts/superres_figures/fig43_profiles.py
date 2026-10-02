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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.join(REPO, 'simple'))
from restore import convolve_with_beam
OUT = os.path.join(OUT_DIR, '43_superres_profiles.png')
CELL = 0.02; BE = (0.220, 0.172, -74.7); BC = (1.418, 0.976, 83.5)
vel = 5073.9 + 25 * (np.arange(32) - 15.5); L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
COL_C, COL_E, COL_R = '#2a78d6', '#eb6834', '#4a3aa7'
FS = 22
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'font.size': FS,
                     'axes.labelsize': FS, 'xtick.labelsize': FS - 2, 'ytick.labelsize': FS - 2, 'legend.fontsize': FS - 4,
                     'axes.linewidth': 1.4, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 6, 'ytick.major.size': 6,
                     'xtick.major.pad': 10, 'ytick.major.pad': 10})
UNIT = r'Jy km s$^{-1}$ arcsec$^{-2}$'
XLAB, YLAB = r'$\Delta$RA [$^{\prime\prime}$]', r'$\Delta$Dec [$^{\prime\prime}$]'

xc = np.load(COMPACT_MODEL).astype(np.float64); xe = np.load(EXTENDED_MODEL).astype(np.float64)
barea = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))                      # beam solid angle [arcsec^2]
m0 = lambda x, b: convolve_with_beam(x[L], CELL, *b).sum(0) * 25 / barea(b)   # surface brightness, beam-independent
C, E, R = m0(xc, BE), m0(xe, BE), m0(xc, BC)
CUTS = [('1: central ridge', (-0.35, -1.6), (-0.45, 2.3)),
        ('2: northern bright knot', (1.0, 1.82), (-1.8, 1.82)),
        ('3: across northern arm', (2.2, 2.4), (3.0, 0.2)),
        ('4: southern arm', (0.6, -4.2), (-3.2, -3.1))]

def cut(img, p0, p1, n=400):
    (ra0, d0), (ra1, d1) = p0, p1
    t = np.linspace(0, 1, n); ra = ra0 + (ra1 - ra0) * t; de = d0 + (d1 - d0) * t
    return np.hypot(ra - ra0, de - d0), map_coordinates(img, [400 + de / CELL, 400 - ra / CELL], order=1)

def _measure():
    f = plt.figure(figsize=(10, 10))
    a = f.add_axes([0.3, 0.3, 0.4, 0.4]); a.set_xlim(1, -1.5); a.set_ylim(-1.5, 1.0); a.set_xlabel(XLAB); a.set_ylabel(YLAB)
    f.canvas.draw(); b = a.get_tightbbox(f.canvas.get_renderer())
    xl = 3 - b.y0 / f.dpi; yl = 3 - b.x0 / f.dpi; plt.close(f); return xl, yl
XL, YL = _measure()

def build(CBL, PYL):
    global fig, W, H
    P, g, CBW, SPW = 5.6, 0.55, 0.28, 9.0
    ROWLAB = 0.55
    LM, RM, TM, BM = ROWLAB + YL + 0.1, 0.3, 1.35, 0.1
    x_r = LM; x_c = x_r + P + g; x_e = x_c + P + g; x_cb = x_e + P + g; x_p = x_cb + CBW + CBL + g + PYL
    W = x_p + SPW + RM; rowh = P + XL + g; H = TM + len(CUTS) * rowh - g + BM
    fig = plt.figure(figsize=(W, H))
    ax_at = lambda x, y, w, h, **kw: fig.add_axes([x / W, y / H, w / W, h / H], **kw)
    def frame(a):
        for sp in a.spines.values(): sp.set_visible(True)
    def beam(a, e, b, col):
        wx, hy = e[0] - e[1], e[3] - e[2]
        big = b[0] > 1
        x0, y0 = (e[0] - 0.24 * wx, e[2] + 0.24 * hy) if big else (e[0] - 0.2 * wx, e[2] + 0.2 * hy)
        a.add_patch(Ellipse((x0, y0), width=b[0], height=b[1], angle=90 - b[2], fc='none', ec=col, lw=2))
        a.text(x0, y0 - (0.14 if big else 0.08) * hy, f'{b[0]:.2f}$^{{\\prime\\prime}}$ $\\times$ {b[1]:.2f}$^{{\\prime\\prime}}$',
               color=col, ha='center', va='top', fontsize=FS - 5, fontweight='bold')
    meas = []
    for i, (nm, p0, p1) in enumerate(CUTS):
        y = H - TM - i * rowh - P
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2); half = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) / 2 + 0.6
        r0, r1 = int(400 + mid[1] / CELL - half / CELL), int(400 + mid[1] / CELL + half / CELL)
        c0, c1 = int(400 - mid[0] / CELL - half / CELL), int(400 - mid[0] / CELL + half / CELL)
        z = np.s_[r0:r1, c0:c1]; e = [(400 - c0) * CELL, (400 - c1) * CELL, (r0 - 400) * CELL, (r1 - 400) * CELL]
        vz = max(C[z].max(), E[z].max(), R[z].max()); length = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        ticks = np.arange(0, length + 1e-6, 0.5)
        axs = [ax_at(x_r, y, P, P)]; axs += [ax_at(x, y, P, P, sharex=axs[0], sharey=axs[0]) for x in (x_c, x_e)]
        for a, img, b in zip(axs, (R, C, E), (BC, BE, BE)):
            im = a.imshow(img[z], origin='lower', extent=e, cmap='inferno', vmin=0, vmax=vz)
            a.plot([p0[0], p1[0]], [p0[1], p1[1]], color='c', lw=1.8)
            for tk in ticks:
                f_ = tk / length; a.plot(p0[0] + f_ * (p1[0] - p0[0]), p0[1] + f_ * (p1[1] - p0[1]), 'o', ms=7, mfc='w', mec='c', mew=1.5)
            a.plot(*p0, 's', ms=10, mfc='c', mec='w'); beam(a, e, b, 'w'); frame(a)
            a.set_xlim(e[0], e[1]); a.set_ylim(e[2], e[3]); a.set_xlabel(XLAB)
        axs[0].xaxis.set_major_locator(MaxNLocator(5, prune='both')); axs[0].yaxis.set_major_locator(MaxNLocator(6, prune='both'))
        for a in axs[1:]: plt.setp(a.get_yticklabels(), visible=False)
        axs[0].set_ylabel(YLAB)
        cax = ax_at(x_cb, y, CBW, P); cb = fig.colorbar(im, cax=cax); cb.set_label(UNIT); frame(cax)
        cax.tick_params(which='both', direction='out', left=False, right=True, labelright=True)
        fig.text((x_r - YL - 0.1 - ROWLAB / 2) / W, (y + P / 2) / H, f'Cut {nm}', rotation=90, ha='center', va='center', fontsize=FS + 2, fontweight='bold')
        ap = ax_at(x_p, y, SPW, P)
        x, yr = cut(R, p0, p1); _, ye = cut(E, p0, p1); _, yc = cut(C, p0, p1)
        ap.plot(x, yr, color=COL_R, lw=2.2, ls='--'); ap.plot(x, yc, color=COL_C, lw=2.4); ap.plot(x, ye, color=COL_E, lw=2.8)
        ap.set_xticks(ticks)
        ap.plot(0, 0, 's', ms=10, mfc='c', mec='w', clip_on=False, zorder=5)
        frame(ap); ap.set_xlim(0, length); ap.set_ylim(0, max(yr.max(), ye.max(), yc.max()) * 1.38)
        ap.set_xlabel('distance along cut [$^{\\prime\\prime}$]'); ap.set_ylabel(UNIT)
        ap.legend(handles=[Line2D([], [], color=COL_R, lw=2.2, ls='--', label=r'Compact $\ast$ compact beam'),
                           Line2D([], [], color=COL_C, lw=2.4, label=r'Compact $\ast$ extended beam'),
                           Line2D([], [], color=COL_E, lw=2.8, label=r'Extended $\ast$ extended beam')],
                  loc='upper right', ncol=1, frameon=True, fancybox=False, framealpha=1, edgecolor='k')
        meas.append((cax, ap))
    top = H - TM
    def colhead(xmid, l1, l2=None):
        fig.text(xmid / W, (top + 0.62) / H, l1, ha='center', va='bottom', fontsize=FS + 1, fontweight='bold')
        if l2: fig.text(xmid / W, (top + 0.14) / H, l2, ha='center', va='bottom', fontsize=FS - 3, fontweight='bold')
    colhead(x_r + P / 2, COMPACT_TITLE, r'$\ast$ Compact clean beam')
    colhead(x_e + P / 2, 'Extended (deconvolved)', r'$\ast$ Extended clean beam')
    colhead(x_c + P / 2, COMPACT_TITLE, r'$\ast$ Extended clean beam')
    colhead(x_p + SPW / 2, 'Brightness along the cut', 'x ticks = white dots (every 0.5$^{\\prime\\prime}$); square = start')
    fig.canvas.draw(); r = fig.canvas.get_renderer(); d = fig.dpi
    m1 = max(c.get_tightbbox(r).x1 / d - c.get_position().x1 * W for c, _ in meas)
    m2 = max(a.get_position().x0 * W - a.get_tightbbox(r).x0 / d for _, a in meas)
    return m1, m2

m1, m2 = build(1.2, 1.2); plt.close(fig); build(m1, m2)
fig.savefig(OUT, dpi=45); print('saved')
