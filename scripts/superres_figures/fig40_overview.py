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
from matplotlib.patches import Ellipse, Rectangle, Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.join(REPO, 'simple'))
from restore import convolve_with_beam
OUT = os.path.join(OUT_DIR, '40_superres_overview.png')
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
barea = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))              # arcsec^2
bpx = lambda b: barea(b) / CELL ** 2                                   # pixels
cubes = {'R': convolve_with_beam(xc, CELL, *BC), 'C': convolve_with_beam(xc, CELL, *BE), 'E': convolve_with_beam(xe, CELL, *BE)}   # Jy/beam
beams = {'R': BC, 'C': BE, 'E': BE}
h = 250; s = np.s_[400 - h:400 + h, 400 - h:400 + h]; ext = [5, -5, -5, 5]
SB = {k: v[L].sum(0)[s] * 25 / barea(beams[k]) for k, v in cubes.items()}          # surface brightness
SPEC = {k: v[:, s[0], s[1]].sum((1, 2)) / bpx(beams[k]) for k, v in cubes.items()}  # Jy per channel in the 10" box
vmax = max(SB[k].max() for k in SB); D = SB['C'] - SB['E']; rmax = max(SB['C'].max(), SB['E'].max())

def frame(a):
    for sp in a.spines.values(): sp.set_visible(True)
def heading(a, t, col, sub=None):
    a.text(0.04, 0.96, t, transform=a.transAxes, ha='left', va='top', color=col, fontsize=FS, fontweight='bold')
    if sub: a.text(0.04, 0.885, sub, transform=a.transAxes, ha='left', va='top', color=col, fontsize=FS - 5, fontweight='bold')
def beam(a, b, col):
    big = b[0] > 1
    x0, y0 = (3.1, -3.4) if big else (3.15, -3.9)
    a.add_patch(Ellipse((x0, y0), width=b[0], height=b[1], angle=90 - b[2], fc='none', ec=col, lw=2))
    a.text(x0, y0 - (0.85 if big else 0.35), f'{b[0]:.2f}$^{{\\prime\\prime}}$ $\\times$ {b[1]:.2f}$^{{\\prime\\prime}}$', color=col, ha='center', va='top', fontsize=FS - 4, fontweight='bold')

def build(YLS, CBL):
    global fig, W, H
    P, g, CBW = 7.0, 0.55, 0.32
    f0 = plt.figure(); a = f0.add_axes([0.3, 0.3, 0.4, 0.4]); a.set_xlim(5, -5); a.set_xlabel(XLAB); f0.canvas.draw()
    XL = 0.3 * f0.get_figheight() - a.get_tightbbox(f0.canvas.get_renderer()).y0 / f0.dpi; plt.close(f0)
    LM, RM, TM, BM = YLS + 0.2, 0.3, 0.2, XL + 0.2
    W = LM + 3 * P + 2 * g + g + CBW + CBL + RM; H = TM + 2 * P + XL + g + BM
    fig = plt.figure(figsize=(W, H))
    ax_at = lambda x, y, w, hh, **kw: fig.add_axes([x / W, y / H, w / W, hh / H], **kw)
    xs = [LM + k * (P + g) for k in range(3)]; y1 = BM + P + XL + g; y2 = BM; xcb = xs[2] + P + g
    aR = ax_at(xs[0], y1, P, P); aC = ax_at(xs[1], y1, P, P, sharex=aR, sharey=aR); aE = ax_at(xs[2], y1, P, P, sharex=aR, sharey=aR)
    asp = ax_at(xs[0], y2, P, P); aK = ax_at(xs[1], y2, P, P, sharex=aR, sharey=aR); aD = ax_at(xs[2], y2, P, P, sharex=aR, sharey=aR)
    for a, k, t in ((aR, 'R', r'$\ast$ Compact clean beam'), (aC, 'C', r'$\ast$ Extended clean beam'), (aE, 'E', r'$\ast$ Extended clean beam')):
        im = a.imshow(SB[k], origin='lower', extent=ext, cmap='inferno', vmin=0, vmax=vmax)
        heading(a, 'Extended (deconvolved)' if k == 'E' else COMPACT_TITLE, 'w', t); beam(a, beams[k], 'w')
    aK.imshow(SB['C'], origin='lower', extent=ext, cmap='gray_r', vmin=0, vmax=rmax)
    aK.contour(SB['E'], levels=np.array([0.15, 0.3, 0.5, 0.7, 0.9]) * SB['E'].max(), colors=[COL_E], linewidths=1.4, extent=ext, origin='lower')
    heading(aK, 'Compact + extended contours', 'k')
    aK.text(0.04, 0.885, 'both $\\ast$ extended beam; contours 15$-$90% of peak', transform=aK.transAxes, ha='left', va='top', color=COL_E, fontsize=FS - 6, fontweight='bold')
    imr = aD.imshow(D, origin='lower', extent=ext, cmap='RdBu_r', vmin=-rmax, vmax=rmax)
    heading(aD, 'Compact $-$ Extended', 'k', f'rms {D.std() / rmax:.1%} of peak')
    for a, col in ((aK, 'k'), (aD, 'k')): beam(a, BE, col)
    # boxes of the zoom figure; each letter sits on a tag just outside its box, clear of all box edges
    TAGS = {'A': (-2.12, -0.3), 'B': (-0.4, 3.22), 'C': (2.4, 2.82), 'D': (0.02, -3.9)}
    for nm, ra, dec, sz in [('A', -0.4, -0.3, 2.8), ('B', -0.4, 1.8, 2.2), ('C', 2.4, 1.0, 3.0), ('D', -1.9, -3.9, 3.2)]:
        for a in (aC, aE):
            a.add_patch(Rectangle((ra - sz / 2, dec - sz / 2), sz, sz, fc='none', ec='c', lw=1.6, ls=(0, (5, 3))))
            a.text(*TAGS[nm], nm, color='k', fontsize=FS - 6, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.25', fc='c', ec='none'))
    for a in (aR, aC, aE, aK, aD):
        frame(a); a.set_xlim(5, -5); a.set_ylim(-5, 5); a.set_xticks([4, 2, 0, -2, -4]); a.set_yticks([-4, -2, 0, 2, 4]); a.set_xlabel(XLAB)
    for a in (aC, aE, aK, aD): plt.setp(a.get_yticklabels(), visible=False)
    aR.set_ylabel(YLAB)
    c1 = ax_at(xcb, y1, CBW, P); c2 = ax_at(xcb, y2, CBW, P)
    for im_, cax in ((im, c1), (imr, c2)):
        cb = fig.colorbar(im_, cax=cax); cb.set_label(UNIT); frame(cax)
        cax.tick_params(which='both', direction='out', left=False, right=True, labelright=True)
    # spectrum
    asp.axvspan(vel[L[0]] - 12.5, vel[L[-1]] + 12.5, color='0.9', zorder=0)
    asp.step(vel, SPEC['R'], where='mid', color=COL_R, lw=2.2, ls='--')
    asp.step(vel, SPEC['E'], where='mid', color=COL_E, lw=2.6)
    asp.step(vel, SPEC['C'], where='mid', color=COL_C, lw=2.2)
    frame(asp); asp.set_xlim(vel[0] - 12.5, vel[-1] + 12.5); asp.set_ylim(0, max(v.max() for v in SPEC.values()) * 1.55)
    asp.set_xticks([4700, 4900, 5100, 5300]); asp.set_xlabel(r'$v_{\rm LSRK,\,radio}$ [km s$^{-1}$]'); asp.set_ylabel('flux density [Jy]')
    asp.text(0.04, 0.96, r'Integrated spectrum (10$^{\prime\prime}$)', transform=asp.transAxes, ha='left', va='top', fontsize=FS, fontweight='bold')
    fl = {k: v[L].sum() * 25 for k, v in SPEC.items()}
    asp.legend(handles=[Line2D([], [], color=COL_R, lw=2.2, ls='--', label=f'Comp. $\\ast$ comp. beam ({fl["R"]:.0f})'),
                        Line2D([], [], color=COL_C, lw=2.2, label=f'Comp. $\\ast$ ext. beam ({fl["C"]:.0f})'),
                        Line2D([], [], color=COL_E, lw=2.6, label=f'Ext. $\\ast$ ext. beam ({fl["E"]:.0f})'),
                        Patch(facecolor='0.9', edgecolor='0.55', label='moment-0 channels')],
               loc='upper right', bbox_to_anchor=(0.98, 0.87), frameon=True, fancybox=False, framealpha=1, edgecolor='k',
               title=r'(Jy km s$^{-1}$)', title_fontsize=FS - 5, fontsize=FS - 6)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); d = fig.dpi
    yls = max(a.get_position().x0 * W - a.get_tightbbox(r).x0 / d for a in (aR, asp))
    cbl = max(c.get_tightbbox(r).x1 / d - c.get_position().x1 * W for c in (c1, c2))
    return yls, cbl

yls, cbl = build(1.6, 1.4); plt.close(fig); build(yls, cbl)
fig.savefig(OUT, dpi=55); print('saved; 10" box flux (Jy km/s):', {k: round(v[L].sum() * 25, 1) for k, v in SPEC.items()})
