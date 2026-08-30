"""
Load real visibilities straight off a measurement set for the NUFFT
pipeline in `deconvolve_nufft.py` -- this is `notebooks/test.ipynb`'s
`load_spectral_cube_visibilities`, turned into a script, with the one real
bug that notebook had fixed for this dataset: it read every row in the MS
with no FIELD_ID filter, so on TW Hya's MS ~34% of the rows it fed to the
NUFFT were calibrator scans (quasars, an asteroid) pointing at a completely
different part of the sky, phase-referenced as if they were the science
target.

NGC 7469 / IRAS F23007+0836 does not have that particular problem -- the
split MS (`data/ngc7469_co21.ms`) already contains only the 3 science
mosaic pointings, no calibrators (verified: `fieldnames()` returns
`['IRASF23007+0836'] * 3`). But it has a DIFFERENT problem the TW Hya
notebook never had to deal with, because TW Hya is a single pointing:
combining 3 pointings into one NUFFT/image needs the visibilities of each
pointing rotated onto a COMMON phase center first.

======================================================================
WHY A MOSAIC CAN'T JUST BE NUFFT'D AS-IS
======================================================================
Every visibility is phase-referenced to its own field's pointing center --
literally what the correlator locked onto for that scan. The FIELD table
confirms the 3 pointings here really do have distinct centers ~10-13" apart
(`PHASE_DIR`), all close to one common direction (`REFERENCE_DIR`, IDENTICAL
across all 3 rows -- clearly the mosaic's intended common reference point).

If you grid visibilities from 2 different phase centers into the same
(u, v) plane without correcting for that, you are not adding noise -- you
are adding a spurious phase gradient across the field that systematically
smears/distorts the combined image (each field's data effectively
represents the sky exp(-2*pi*i*(u*dRA + v*dDec)) exp() away from where you
think it does). The fix is a per-visibility phase rotation, derived below,
onto the shared `REFERENCE_DIR`.

Derivation (small-field / tangent-plane approximation): let I_ref(l, m) be
the sky relative to the common center, and (dl, dm) the field's own phase
center offset from that common center. A visibility phase-referenced to the
field is

    V_field(u, v) = Integral I_field(l, m) exp(-2*pi*i*(u*l + v*m)) dl dm
                  = Integral I_ref(l + dl, m + dm) exp(-2*pi*i*(u*l + v*m)) dl dm

substituting l' = l + dl, m' = m + dm:

    V_field(u, v) = exp(+2*pi*i*(u*dl + v*dm)) * V_ref(u, v)

so the correction that recovers V_ref (what a NUFFT centered at the common
direction expects) is

    V_ref(u, v) = V_field(u, v) * exp(-2*pi*i*(u*dl + v*dm))

with u, v in wavelengths (baseline / lambda) and (dl, dm) the field's phase
center offset from the reference, in radians on the tangent plane.

Still missing after this correction: per-pointing PRIMARY BEAM weighting
(mosaicking proper also multiplies each field's contribution by its own PB
response before combining, so pointings agree on relative flux across the
overlap region -- this loader does not do that; see deconvolve_nufft.py's
docstring for what that costs on this dataset).
"""

import numpy as np
import torch
from casacore.tables import table

SPEED_OF_LIGHT = 299792458.0


def load_mosaic_visibilities(ms_path, spw=0, chan_start=0, chan_end=None,
                             cell_size_arcsec=0.04, correct_mosaic_phase=True):
    """
    Returns
    -------
    ktraj : (2, n_points) float32 tensor -- torchkbnufft k-space trajectory
    v_batched : (n_channels, 1, n_points) complex64 tensor -- visibilities
    cell_size_arcsec : float (echoed back; also the natural, resolution-based
        estimate is printed for comparison -- see the note below on why it
        is NOT used by default)
    meta : dict with the common phase center and per-field offsets, for
        sanity-checking the correction (see `deconvolve_nufft.py`'s
        `--phase-check` diagnostic)
    """
    # -- 1. frequency axis for this spw, and the field phase centers -------
    # `getcol` fails across all 4 rows: this MS's spws have DIFFERENT channel
    # counts (291/259/303/248), so CHAN_FREQ is a per-row variable-length
    # array, not a uniform column. `getcell` reads the one row we need.
    spw_tb = table(f"{ms_path}/SPECTRAL_WINDOW", ack=False)
    chan_freqs = spw_tb.getcell("CHAN_FREQ", spw)
    spw_tb.close()
    if chan_end is None:
        chan_end = len(chan_freqs)
    selected_freqs = chan_freqs[chan_start:chan_end]
    center_frequency = np.median(selected_freqs)          # narrowband approx
    wavelength = SPEED_OF_LIGHT / center_frequency

    field_tb = table(f"{ms_path}/FIELD", ack=False)
    phase_dir = field_tb.getcol("PHASE_DIR")[:, 0, :]      # (n_field, 2) rad
    reference_dir = field_tb.getcol("REFERENCE_DIR")[:, 0, :]
    field_tb.close()
    ra_ref, dec_ref = reference_dir[0]                     # identical every row

    # tangent-plane offset of each field's phase center from the common one
    dl = (phase_dir[:, 0] - ra_ref) * np.cos(dec_ref)
    dm = phase_dir[:, 1] - dec_ref
    print(f"[load] common phase center (REFERENCE_DIR): "
          f"RA={np.degrees(ra_ref):.6f} deg  Dec={np.degrees(dec_ref):.6f} deg")
    for f in range(len(dl)):
        print(f"[load]   field {f}: offset "
              f"({dl[f] * 206265:+.2f}\", {dm[f] * 206265:+.2f}\")")

    # -- 2. rows for this spw only ------------------------------------------
    dd_tb = table(f"{ms_path}/DATA_DESCRIPTION", ack=False)
    spw_ids = dd_tb.getcol("SPECTRAL_WINDOW_ID")
    dd_tb.close()
    ddid = int(np.nonzero(spw_ids == spw)[0][0])

    tb = table(ms_path, ack=False)
    sub = tb.query(f"DATA_DESC_ID=={ddid}")
    col_names = sub.colnames()
    data_col = "CORRECTED_DATA" if "CORRECTED_DATA" in col_names else "DATA"

    pol_idx = 0
    data = sub.getcol(data_col)[:, chan_start:chan_end, pol_idx]
    flags = sub.getcol("FLAG")[:, chan_start:chan_end, pol_idx]
    uvw = sub.getcol("UVW")
    field_id = sub.getcol("FIELD_ID")
    sub.close()
    tb.close()
    print(f"[load] spw {spw} (ddid {ddid}): {data.shape[0]} rows, "
          f"{data.shape[1]} channels, center {center_frequency / 1e9:.4f} GHz")

    # -- 3. drop rows flagged anywhere in the selected channel range --------
    valid = ~np.any(flags, axis=1)
    data, uvw, field_id = data[valid], uvw[valid], field_id[valid]
    print(f"[load] {valid.sum()}/{len(valid)} rows unflagged across all "
          f"selected channels")

    u = uvw[:, 0]
    v = uvw[:, 1]

    # -- 4. mosaic phase correction: rotate every row onto REFERENCE_DIR ---
    if correct_mosaic_phase:
        u_lambda, v_lambda = u / wavelength, v / wavelength
        row_dl, row_dm = dl[field_id], dm[field_id]
        correction = np.exp(-2j * np.pi * (u_lambda * row_dl + v_lambda * row_dm))
        data = data * correction[:, None]
        print(f"[load] applied mosaic phase correction "
              f"({(row_dl != 0).sum() + (row_dm != 0).sum()} rows offset "
              f"from the reference direction)")

    v_meas = data.T                                        # (n_chan, n_rows)

    # -- 5. NOT using the notebook's auto cell size: it is far too fine ----
    # `resolution / 8` from the max baseline gives ~0.011" here, and at the
    # notebook's im_size=256 that is a 2.7" field of view -- too small to
    # contain even the ~13" spread between mosaic pointings, let alone the
    # source (measured 6.3" from field 0's own phase center in the earlier
    # CASA-gridded pipeline). Rotating onto the shared REFERENCE_DIR (close
    # to the true source position) fixes the *centering*, but the FIELD OF
    # VIEW still has to be chosen by hand to be wide enough. `cell_size_arcsec`
    # is a plain function argument here (default 0.04", matching the
    # CASA-gridded pipeline in `../src` and `../simple` for direct
    # comparability) rather than auto-computed, precisely so this can't be
    # silently too small again.
    max_baseline = np.hypot(u, v).max()
    natural_cell = np.degrees(wavelength / max_baseline / 8.0) * 3600.0
    print(f"[load] max baseline {max_baseline:.1f} m -> resolution-based "
          f"cell would be {natural_cell:.4f}\" (NOT used -- see docstring); "
          f"using {cell_size_arcsec}\"")

    cell_size_rad = np.radians(cell_size_arcsec / 3600.0)
    u_scaled = (u / wavelength) * cell_size_rad * 2 * np.pi
    v_scaled = (v / wavelength) * cell_size_rad * 2 * np.pi
    in_nyquist = (np.abs(u_scaled) < np.pi) & (np.abs(v_scaled) < np.pi)
    print(f"[load] {in_nyquist.sum()}/{len(in_nyquist)} baselines inside the "
          f"+/-pi Nyquist limit at this cell size "
          f"({(~in_nyquist).sum()} dropped)")

    ktraj = np.vstack((u_scaled[in_nyquist], v_scaled[in_nyquist]))
    v_meas = v_meas[:, in_nyquist]

    ktraj_tensor = torch.tensor(ktraj, dtype=torch.float32)
    v_tensor = torch.tensor(v_meas, dtype=torch.complex64).unsqueeze(1)

    meta = {"reference_ra_deg": np.degrees(ra_ref),
            "reference_dec_deg": np.degrees(dec_ref),
            "field_offsets_arcsec": np.stack([dl, dm], axis=1) * 206265,
            "wavelength_m": wavelength, "center_frequency_hz": center_frequency}
    return ktraj_tensor, v_tensor, cell_size_arcsec, meta
