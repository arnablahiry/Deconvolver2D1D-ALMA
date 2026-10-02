# Overnight experiments: review (NGC 3110 / NGC 2369, CO(2-1))

Run 2026-10-01, 00:55-09:08 Athens on 4 A100s: 73 experiment tasks plus the 18-run k-sweep, none failed.
Numbers come from `data/experiments/results.json` and `data/experiments/shape_bias.json`, produced by
`scripts/experiments/analyze.py`. The figures are in the subfolders here.

## Bottom line

1. **Use primal-dual with FISTA's reweighting reference (`pdtau`, reference τλ).** It is the best method
   on both simulations and on the independent extended data, and it is nearly independent of k, so the
   threshold no longer needs tuning. The current primal-dual reweighting (reference λ) is about β ≈ 1500×
   too gentle to remove the L1 bias. FISTA is the worst of the three everywhere.
2. **Debiasing helps primal-dual and FISTA, and is roughly neutral for `pdtau`** (it slightly hurts on
   the simulations and slightly helps on the real cross-check). Once `pdtau` is used, debiasing is optional.
3. **After reweighting, the remaining residual is not mainly L1 bias.** On the extended cube,
   emission-shaped missing flux is +0.3% for primal-dual at k = 3 and +1.7% at k = 5. That is where the
   arm-shaped residual comes from. `pdtau` and FISTA slightly over-correct (−1.6 to −2.4%), and debiasing
   brings every method to about −1.2%. The structure left in the extended residual is large-scale,
   1–2″ patches, not arm-shaped.
4. **Compact total flux is biased high by 5–9% from fitted noise; flux inside the emission is accurate
   to < 1%.** The simulation and the bootstrap agree on this. Quote fluxes in apertures or the emission
   support, not over the whole field.
5. **Random errors are tiny; systematic ones dominate.** Bootstrap flux errors are 0.2–0.3% per region,
   against +0.4–1.6% systematic bias per region and +4.8% in total.
6. **The 32″ field experiment failed as designed** (see section 5), and its fluxes should not be used.
7. **NGC 2369 blind test: partial success.** The same recipe predicts the independent extended data very
   well at ≥ 0.32″, but worse than for NGC 3110 at 0.16–0.32″. Its extended model is not a reliable flux
   reference.
8. **The 0.02″ grid gives no meaningful gain over 0.04″ for compact,** and costs about 4× more.

## 1. Simulations with known truth (`01_simulations/`)

The synthetic galaxy is observed through the real PSFs, with noise drawn from the measured power
spectrum. "Error" is ‖G∗(model − truth)‖ / ‖G∗truth‖ at the extended clean beam G (lower is better).

**Compact simulation** (0.04″ grid), error vs truth:

| method | k = 2 | 3 | 4 | 5 | debiased (k = 2 … 5) |
|---|---|---|---|---|---|
| primal-dual | 0.409 | 0.420 | 0.429 | 0.437 | 0.398 0.406 0.412 0.416 |
| **pdtau** | **0.394** | **0.391** | **0.391** | **0.393** | 0.394 0.400 0.405 0.410 |
| FISTA | 0.425 | 0.440 | 0.456 | 0.466 | 0.407 0.412 0.418 0.421 |

**Extended simulation** (k = 3, support mask): primal-dual 0.103 (debiased 0.091); **pdtau 0.080**
(debiased 0.087); FISTA 0.088 (debiased 0.086).

- **k dependence:** primal-dual and FISTA get worse as k grows, which is the L1 bias growing with λ.
  `pdtau` stays flat.
- **Scales:** all methods have the same per-scale error profile. 0.04–0.16″ is essentially unrecovered
  (0.76–0.90), 0.16–0.32″ is partly recovered (0.53–0.65), and ≥ 0.32″ is good (≤ 0.26).
- **Flux, compact:** the truth is 27.5 Jy, with 23.4 Jy inside the true emission. Every model gets the
  inside flux right (23.1–23.5 Jy) but adds 1.7–3 Jy outside it.
- **Flux, extended:** the models recover 11.6–12.2 of 13.0 Jy (the largest scales are resolved out):
  primal-dual 11.6, `pdtau` 12.0, FISTA 12.2.

Figures: `01_sim_errors.png`, `01_sim_maps.png`.

## 2. Cross-validation against the independent extended data (`02_crossvalidation/`)

Each compact model is pushed through the extended PSF and compared with the extended dirty cube. The
value is the fraction of extended signal energy left unexplained at 0.16–0.32″, the extended-beam scale
(lower is better). The full table, all scales, is `xval_table.md`.

| model | k = 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| primal-dual (0.04″) | 0.513 | 0.521 | 0.528 | 0.535 |
| primal-dual (0.02″) | – | 0.507 | 0.514 | 0.520 |
| **pdtau** | 0.508 | **0.500** | **0.500** | **0.499** |
| FISTA (0.04″) | 0.552 | 0.569 | 0.584 | 0.596 |
| primal-dual, debiased | 0.496 | 0.494 | 0.496 | 0.499 |
| **pdtau, debiased** | **0.491** | **0.489** | 0.492 | 0.495 |
| FISTA, debiased | 0.510 | 0.504 | 0.506 | 0.508 |

- **The ranking matches the simulations:** pdtau ≥ primal-dual > FISTA, and debiasing helps all three.
- **The best real-data model is pdtau k = 3, debiased (0.489),** against the current recipe's 0.513.
  That's a 5% improvement in agreement at the extended-beam scale.
- **The 0.02″ grid is only 1–3% better at 0.16–0.32″** and identical at 0.08–0.16″.

## 3. Debiasing (`03_debiasing/`)

**Method:** starting from each converged model, run 5000 primal-dual iterations with λ = 0 on the
"detected" coefficients and the normal λ elsewhere. A coefficient counts as detected if the model's
predicted dirty cube is significant there, |W N x₀| > λ_b. That leaves essentially nothing free at the
unmeasured finest scales: 0.1% of coefficients at 0.04–0.08″.

**Extended residual rms** (k = 3, in units of the noise):

| method | before | after debiasing |
|---|---|---|
| primal-dual | 1.55 | 1.52 |
| FISTA | 1.81 | 1.51 |
| pdtau | 1.37 | 1.50 |

**Emission-shaped residual,** measured as the regression slope of the residual moment 0 on the predicted
emission (> 0 means emission is missing in its own shape, i.e. L1 bias):

| extended model | k = 3 | k = 5 |
|---|---|---|
| primal-dual, before reweighting existed | +0.30% | – |
| primal-dual (current reweighting) | +0.26% | **+1.67%** |
| pdtau | −1.84% | −1.60% |
| FISTA | −2.36% | −0.80% |
| any method, debiased | ≈ −1.2% | – |

On the compact cube, the emission-shaped residual is +0.1 to +0.4% for every model: negligible.

- **Where the arm-shaped residual comes from:** primal-dual at high k. The arm-shaped residual seen in
  the extended FISTA 3σ GIFs came from the run without reweighting. That model was later overwritten by
  the reweighted sweep run, so it can't be re-measured. Its GIFs are kept in `figures/fista/extended/3-sigma/`.
- **Effect on extended flux:** debiasing moves every method to about 24 Jy (primal-dual 21.9 → 23.8;
  pdtau 27.5 → 24.1).

Figures: `03_extended_before_after.png`, `03_compact_before_after.png`, `03_bias_summary.png`.

## 4. Primal-dual with FISTA's reweighting reference (`04_pd_tau_reweighting/`)

GIFs for every k and configuration are in `<k>_sigma/<compact|extended>/`.

- **Extended flux:** pdtau recovers much more of it: 27.5 / 25.3 / 23.5 Jy at k = 3 / 4 / 5, against
  21.9 / 20.3 / 18.9 for primal-dual. At k = 3 that matches the compact flux (27.7 Jy).
- **Extended residual:** the lowest of all methods (1.37× noise).
- **Line-free flux:** stays negligible (0.008 Jy).
- **Compact:** fluxes of 28.2 to 27.9 Jy, slightly higher than primal-dual.

On the extended simulation it has the lowest error (0.080), and its flux (12.0 of 13.0 Jy) is close to
FISTA's 12.2 and above primal-dual's 11.6. Figure: `04_extended_k3_methods.png`.

## 5. 32″ field for compact (`05_large_field/`): failed as designed

The new tclean cube is in `data/ngc3110/compact/cube_32arcsec/`. The deconvolution of it is not usable:

- **It over-fits:** inside the primary-beam circle the residual is about 10× *below* the noise.
- **It builds a pedestal:** 9.9 Jy in line-free channels, and 52 Jy inside the central 16″ against 27.7 Jy
  for the 16″ run.

Two design errors cause it:
1. **The field is larger than the compact array's largest recoverable scale (9.8″),** so smooth
   large-scale flux is unconstrained. This is the same failure as the unmasked extended cube.
2. **The region outside the primary-beam cutoff** (4.8% of pixels, NaN → 0) is treated as real
   zero-valued data.

A proper version needs data weights that exclude the cut-off region, plus a support mask or a
large-scale constraint, or ACA data. The 16″ run's residual rising toward its field edge
(`05_large_field.png`, panel 3) remains real but unexplained.

## 7. NGC 2369 blind test (`07_ngc2369/`)

The recipe was applied unchanged (compact k = 2; extended k = 3 inside a newly grown support mask).

**Line channels:** detected automatically from the compact cube, 2–28 (2893–3543 km/s). That leaves only
5 line-free channels (0, 1, 29, 30, 31) for the noise estimate, which is thin.

- **At the extended beam the maps agree** (compact − extended rms 4.2% of peak, against 3.7% for NGC 3110).
- **Cross-validation is excellent at ≥ 0.32″** (0.05–0.16 unexplained) but worse at 0.16–0.32″ (0.654,
  against 0.513 for NGC 3110).
- **At < 0.16″ the compact model adds structure the extended data don't have** (1.2–1.7, against
  1.07–1.09 for NGC 3110).

So super-resolution transfers only partly to this nearly edge-on, narrow galaxy.

**The extended model is not a flux reference:** it has more flux than compact (105 against 70 Jy;
1506 against 1178 Jy km/s in the central 10″). Its mask grew to 48% of voxels, so it lets in a line-shaped
pedestal, as NGC 3110's mask did once its noise was estimated correctly (36%).

Figure: `07_ngc2369_blind_test.png`.

## 8. Error maps (`08_error_maps/`)

Parametric bootstrap: data = N × (final compact model) + new noise with the measured power spectrum,
deconvolved 10 times with the same recipe.

| | input model | bootstrap mean ± std |
|---|---|---|
| total flux [Jy] | 27.69 | 29.01 ± 0.05 (**+4.8% bias**) |
| A: central ridge [Jy km/s] | 122.6 | 123.6 ± 0.25 |
| B: northern knot | 69.6 | 69.9 ± 0.11 |
| C: northern arm | 66.1 | 67.0 ± 0.16 |
| D: southern arm | 69.4 | 70.6 ± 0.21 |

- **Systematic error dominates:** the bias is 0.4–1.6% per region and 4.8% in total, against 0.2–0.3%
  random error. It's the fitted-noise pedestal again, consistent with the simulation's excess outside the
  emission.
- **Knot peaks at the extended beam have S/N ≈ 15** (median of mean/std). Knot positions and shapes vary
  between realizations at the ±7%-of-peak level.
- **The bootstrap ignores calibration error, the negative bowl and edge effects,** so these are lower
  limits on the uncertainty.

## Also from the overnight k-sweep (`figures/{pd,fista}/<k>_sigma/`, `figures/0.02arcsec/`)

- **Compact flux hardly depends on k** (27.6 to 27.5 Jy for k = 3 to 5).
- **Extended flux does,** by 14% between k = 3 and k = 5: the L1 shrinkage at low S/N.
- **The 0.02″ compact runs** change the flux by < 0.4% and cross-validation by 1–3%.

## Suggested next steps

1. **Make `pdtau` (k = 3) the default** and re-make figures 40–44 and the notebook with it. Optionally
   debias it.
2. **Fix the compact flux bias:** measure fluxes inside a support region, or add the support mask to
   compact too.
3. **Joint compact + extended deconvolution** (not run yet): the natural way to get the diffuse flux and
   the 0.2″ detail at once.
4. **If the larger field is needed,** redo it with primary-beam weights and a support mask.

## Where things are

| | figures (`figures/experiments/`) | models (`data/experiments/`) |
|---|---|---|
| 1 | `01_simulations/` | `01_simulations/{compact,extended}/<method>_k<k>.npy`; inputs in `01_simulations/inputs/` |
| 2 | `02_crossvalidation/` (incl. `xval_table.md`) | – (uses all compact models) |
| 3 | `03_debiasing/` | `03_debiasing/{real,sim}/{compact,extended}/<method>_k<k>_debiased.npy` |
| 4 | `04_pd_tau_reweighting/` (+ GIFs per k) | `04_pd_tau_reweighting/{compact,extended}/pdtau_k<k>.npy` |
| 5 | `05_large_field/` | `05_large_field/pd_k{2,3}.npy` (32″, 0.04″) |
| 7 | `07_ngc2369/` | `07_ngc2369/{compact_pd_k2,extended_pd_k3}.npy`; mask in `data/ngc2369/extended/pd/` |
| 8 | `08_error_maps/` | `08_error_maps/bootstrap_<i>.npy` |

- **Models:** real-data models are in Jy/pixel at 0.02″; simulation models are on their own grid.
- **Code:** `scripts/experiments/` (`common.py`, `run_model.py`, `grow_mask.py`, `run_queue.py`,
  `analyze.py`).
- **Logs:** `logs/experiments/`.
