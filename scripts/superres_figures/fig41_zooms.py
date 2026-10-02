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
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
sys.path.insert(0, os.path.join(REPO, 'simple'))
from restore import convolve_with_beam
OUT = os.path.join(OUT_DIR, '41_superres_zooms.png')
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
ZOOMS = [('A: central ridge', -0.4, -0.3, 2.8), ('B: northern bright knot', -0.4, 1.8, 2.2),
         ('C: northern arm', 2.4, 1.0, 3.0), ('D: southern arm', -1.9, -3.9, 3.2)]

xc = np.load(COMPACT_MODEL).astype(np.float64); xe = np.load(EXTENDED_MODEL).astype(np.float64)
barea = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
cubeR, cubeC, cubeE = convolve_with_beam(xc, CELL, *BC), convolve_with_beam(xc, CELL, *BE), convolve_with_beam(xe, CELL, *BE)
sb = lambda cube, b: cube[L].sum(0) * 25 / barea(b)          # surface brightness [Jy km/s arcsec^-2]
R, C, E = sb(cubeR, BC), sb(cubeC, BE), sb(cubeE, BE)
spec = lambda cube, b, z: cube[:, z[0], z[1]].sum((1, 2)) / (barea(b) / CELL ** 2)   # Jy per channel in box

# ---- measured label sizes (inches), so every whitespace gap is exactly g
def _measure():
    f = plt.figure(figsize=(10, 10))
    a = f.add_axes([0.3, 0.3, 0.4, 0.4]); a.set_xlim(1, -1.5); a.set_ylim(-1.5, 1.0); a.set_xlabel(XLAB); a.set_ylabel(YLAB)
    cb = f.add_axes([0.8, 0.3, 0.03, 0.4]); cb.set_ylim(-2.2, 2.2); cb.yaxis.tick_right(); cb.yaxis.set_label_position('right'); cb.set_ylabel(UNIT)
    cb.tick_params(direction='out', left=False, right=True)
    s = f.add_axes([0.3, 0.75, 0.4, 0.2]); s.set_ylim(0, 1.9); s.set_ylabel('flux density [Jy]')
    f.canvas.draw(); r = f.canvas.get_renderer()
    ba, bc, bs = a.get_tightbbox(r), cb.get_tightbbox(r), s.get_tightbbox(r)
    xl = 0.3 * 10 - ba.y0 / f.dpi; yl = 0.3 * 10 - ba.x0 / f.dpi
    cbl = bc.x1 / f.dpi - 0.83 * 10; syl = 0.3 * 10 - bs.x0 / f.dpi
    plt.close(f); return xl, yl, cbl, syl
XL, YL, _, _ = _measure()
def build(CBL1, CBL2, SYL):
    global fig, W, H
    rows_axes = []
    P, g, CBW, SPW = 5.6, 0.55, 0.28, 5.6            # map side, common gap, colorbar width, spectrum width
    ROWLAB = 0.55                                    # rotated region label to the left of each row
    LM, RM, TM, BM = ROWLAB + YL + 0.1, 0.25, 1.35, 0.1
    x_o = LM; x_c = x_o + P + g; x_e = x_c + P + g; x_cb1 = x_e + P + g
    x_k = x_cb1 + CBW + CBL1 + g; x_r = x_k + P + g; x_cb2 = x_r + P + g
    x_s = x_cb2 + CBW + CBL2 + g + SYL
    W = x_s + SPW + RM
    rowh = P + XL + g
    H = TM + len(ZOOMS) * rowh - g + BM
    fig = plt.figure(figsize=(W, H))
    def ax_at(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)
    def heading(ax, txt, col, sub=None, subcol=None):
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha='left', va='top', color=col, fontsize=FS, fontweight='bold')
        if sub: ax.text(0.04, 0.865, sub, transform=ax.transAxes, ha='left', va='top', color=subcol or col, fontsize=FS - 5, fontweight='bold')
    def beam(ax, e, col, b=BE):
        wx, hy = e[0] - e[1], e[3] - e[2]
        big = b[0] > 1
        x0, y0 = (e[0] - max(0.27 * wx, 0.5 * b[0] + 0.1), e[2] + max(0.3 * hy, 0.62 * b[0] + 0.2)) if big else (e[0] - 0.2 * wx, e[2] + 0.2 * hy)
        ax.add_patch(Ellipse((x0, y0), width=b[0], height=b[1], angle=90 - b[2], fc='none', ec=col, lw=2))
        ax.text(x0, y0 - (max(0.2 * hy, 0.62 * b[0]) if big else 0.08 * hy), f'{b[0]:.2f}$^{{\\prime\\prime}}$ $\\times$ {b[1]:.2f}$^{{\\prime\\prime}}$', color=col, ha='center', va='top', fontsize=FS - 5, fontweight='bold')
    def frame(ax):
        for sp in ax.spines.values(): sp.set_visible(True)
    def cbar(im, cax):
        cb = fig.colorbar(im, cax=cax); cb.set_label(UNIT); frame(cax)
        cax.tick_params(which='both', direction='out', left=False, right=True, labelright=True)

    for i, (nm, ra, dec, sz) in enumerate(ZOOMS):
        y = H - TM - i * rowh - P
        r0, r1 = int(400 + dec / CELL - sz / CELL / 2), int(400 + dec / CELL + sz / CELL / 2)
        c0, c1 = int(400 - ra / CELL - sz / CELL / 2), int(400 - ra / CELL + sz / CELL / 2)
        z = np.s_[r0:r1, c0:c1]; e = [(400 - c0) * CELL, (400 - c1) * CELL, (r0 - 400) * CELL, (r1 - 400) * CELL]
        vz = max(C[z].max(), E[z].max(), R[z].max()); vr = max(C[z].max(), E[z].max()); D = (C - E)[z]
        ao = ax_at(x_o, y, P, P); ac = ax_at(x_c, y, P, P, sharex=ao, sharey=ao); ae = ax_at(x_e, y, P, P, sharex=ao, sharey=ao)
        ak = ax_at(x_k, y, P, P, sharex=ao, sharey=ao); ar = ax_at(x_r, y, P, P, sharex=ao, sharey=ao)
        c1a = ax_at(x_cb1, y, CBW, P); c2a = ax_at(x_cb2, y, CBW, P); asp = ax_at(x_s, y, SPW, P)
        ao.imshow(R[z], origin='lower', extent=e, cmap='inferno', vmin=0, vmax=vz)
        im = ac.imshow(C[z], origin='lower', extent=e, cmap='inferno', vmin=0, vmax=vz)
        ae.imshow(E[z], origin='lower', extent=e, cmap='inferno', vmin=0, vmax=vz)
        ak.imshow(C[z], origin='lower', extent=e, cmap='gray_r', vmin=0, vmax=vr)
        ak.contour(E[z], levels=np.array([0.3, 0.5, 0.7, 0.9]) * E[z].max(), colors=[COL_E], linewidths=1.6, extent=e, origin='lower')
        ak.contour(C[z], levels=np.array([0.3, 0.5, 0.7, 0.9]) * C[z].max(), colors=[COL_C], linewidths=1.4, linestyles='--', extent=e, origin='lower')
        imr = ar.imshow(D, origin='lower', extent=e, cmap='RdBu_r', vmin=-vr, vmax=vr)
        ar.text(0.05, 0.95, f'rms {D.std() / vr:.1%} of peak', transform=ar.transAxes, ha='left', va='top', fontsize=FS - 2, fontweight='bold')
        beam(ao, e, 'w', BC); frame(ao); ao.set_xlim(e[0], e[1]); ao.set_ylim(e[2], e[3]); ao.set_xlabel(XLAB)
        for a, col in ((ac, 'w'), (ae, 'w'), (ak, 'k'), (ar, 'k')):
            beam(a, e, col); frame(a); a.set_xlim(e[0], e[1]); a.set_ylim(e[2], e[3]); a.set_xlabel(XLAB)
        ao.xaxis.set_major_locator(MaxNLocator(5, prune='both')); ao.yaxis.set_major_locator(MaxNLocator(6, prune='both'))
        for a in (ac, ae, ak, ar): plt.setp(a.get_yticklabels(), visible=False)
        ao.set_ylabel(YLAB)
        cbar(im, c1a); cbar(imr, c2a)
        rows_axes.append((c1a, c2a, asp))
        fig.text((x_o - YL - 0.1 - ROWLAB / 2) / W, (y + P / 2) / H, nm, rotation=90, ha='center', va='center', fontsize=FS + 2, fontweight='bold')
        # spectrum of the zoom box
        fr, fc, fe = spec(cubeR, BC, z), spec(cubeC, BE, z), spec(cubeE, BE, z)
        asp.axvspan(vel[L[0]] - 12.5, vel[L[-1]] + 12.5, color='0.9', zorder=0)
        asp.step(vel, fr, where='mid', color=COL_R, lw=2.2, ls='--')
        asp.step(vel, fe, where='mid', color=COL_E, lw=2.6); asp.step(vel, fc, where='mid', color=COL_C, lw=2.2)
        frame(asp); asp.set_xlim(vel[0] - 12.5, vel[-1] + 12.5); asp.set_ylim(0, max(fr.max(), fc.max(), fe.max()) * 1.95)
        asp.set_xticks([4700, 4900, 5100, 5300])
        asp.set_xlabel(r'$v_{\rm LSRK,\,radio}$ [km s$^{-1}$]'); asp.set_ylabel('flux density [Jy]')
        fl = lambda f: f[L].sum() * 25
        asp.legend(handles=[Line2D([], [], color=COL_R, lw=2.2, ls='--', label=f'Comp. $\\ast$ comp. ({fl(fr):.1f})'),
                            Line2D([], [], color=COL_C, lw=2.2, label=f'Comp. $\\ast$ ext. ({fl(fc):.1f})'),
                            Line2D([], [], color=COL_E, lw=2.6, label=f'Ext. $\\ast$ ext. ({fl(fe):.1f})'),
                            Patch(facecolor='0.9', edgecolor='0.55', label='moment-0 channels')],
                   loc='upper right', frameon=True, fancybox=False, framealpha=1, edgecolor='k',
                   title=r'(Jy km s$^{-1}$)', title_fontsize=FS - 7, fontsize=FS - 7)

    # column headings above the first row
    top = H - TM
    def colhead(xmid, line1, line2=None):
        fig.text(xmid / W, (top + 0.62) / H, line1, ha='center', va='bottom', fontsize=FS + 1, fontweight='bold')
        if line2: fig.text(xmid / W, (top + 0.14) / H, line2, ha='center', va='bottom', fontsize=FS - 3, fontweight='bold')
    colhead(x_o + P / 2, COMPACT_TITLE, r'$\ast$ Compact clean beam')
    colhead(x_c + P / 2, COMPACT_TITLE, r'$\ast$ Extended clean beam')
    colhead(x_e + P / 2, 'Extended (deconvolved)', r'$\ast$ Extended clean beam')
    colhead(x_k + P / 2, r'Contours ($\ast$ ext. beam, 30$-$90%)')
    fig.text((x_k + P / 2 - 0.1) / W, (top + 0.14) / H, 'extended', ha='right', va='bottom', color=COL_E, fontsize=FS - 3, fontweight='bold')
    fig.text((x_k + P / 2 + 0.1) / W, (top + 0.14) / H, 'compact (dashed)', ha='left', va='bottom', color=COL_C, fontsize=FS - 3, fontweight='bold')
    colhead(x_r + P / 2, 'Compact $-$ Extended')
    colhead(x_s + SPW / 2, 'Integrated spectrum')
    fig.canvas.draw(); r = fig.canvas.get_renderer(); d = fig.dpi
    m1 = max(ax.get_tightbbox(r).x1 / d - (ax.get_position().x1 * W) for ax, _, _ in rows_axes)
    m2 = max(ax.get_tightbbox(r).x1 / d - (ax.get_position().x1 * W) for _, ax, _ in rows_axes)
    m3 = max((ax.get_position().x0 * W) - ax.get_tightbbox(r).x0 / d for _, _, ax in rows_axes)
    return fig, (m1, m2, m3)

fig, (m1, m2, m3) = build(1.0, 1.0, 1.2); plt.close(fig)
fig, meas = build(m1, m2, m3)
print('colorbar label widths / spectrum y-label width [in]:', [round(v, 2) for v in (m1, m2, m3)], ' check:', [round(v, 2) for v in meas])
fig.savefig(OUT, dpi=45)
print('saved', round(W, 1), 'x', round(H, 1), 'in')
