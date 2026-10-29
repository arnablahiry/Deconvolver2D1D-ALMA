# 2D dictionaries

Spatial (2D) dictionaries for the 2D-1D deconvolution, selected by name with no code change:

```python
D = gpu_pd.Dict2D1D(shape, J, L, dict_2d="learnlet")      # or "starlet" (default), "learnlet:16", a path to an .npz
```

or, for scripts that build `Dict2D1D(shape, J, L)` themselves, `DICT_2D=learnlet python scripts/...`;
in the GUI, the "2D dictionary" field.

| name | what |
|---|---|
| `starlet` | built in: isotropic undecimated starlet (B3 spline), J detail planes + coarse |
| `learnlet` | the starlet split + 64 learnt 5×5 filters per detail scale, from the learnlet trained on 10,000 ImageNet images ([vicbonj/learnlet](https://github.com/vicbonj/learnlet); Ramzi et al. 2020, Bonjean et al. 2025) |

`<name>:N` keeps the N filters per scale with the largest share of the reconstruction (‖S_k ∗ A_k‖).

## File format

`<name>.npz` with

* `analysis`, `synthesis`: float32 `(scales, F, k, k)`, k odd. At detail scale j < scales, the starlet plane
  w_j gives the planes w_j, A_j1 ∗ w_j, …, A_jF ∗ w_j (filters dilated by 2^j, reflect boundaries), and the
  inverse is w_j = c_0 + Σ_k S_jk ∗ (c_k − A_jk ∗ c_0), exact for any filters. Deeper scales are plain starlet planes.
* optional: `threshold_multipliers` (the learnlet's learnt per-scale thresholds, unused), `meta` (JSON provenance).

Regenerate / add another trained learnlet: `python scripts/export_learnlet_dictionary.py [--repo ... --weights ... --name ...]`.

## Cost

Each filter is a full plane per scale: with J = 7, `learnlet` has 264 planes vs the starlet's 8. On one A100
(40 GB), per primal-dual iteration:

| cube | starlet | learnlet:16 | learnlet |
|---|---|---|---|
| 32 × 400 × 400 | 92 ms, 1.5 GB | 0.6 s, 7.8 GB | 2.0 s, 26 GB |
| 32 × 800 × 800 | 156 ms, 6.0 GB | 1.1 s, 31 GB | out of memory |

‖W‖² is also larger (starlet 3.6, learnlet:16 78, learnlet 232), so the primal-dual dual step σ = β / (2‖W‖²) is smaller.
