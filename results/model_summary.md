# Model summary

6 systems (13, 28, 30, 33, 46, 64), unit cycles only, test split = last 20% of cycles. All checks pass.

## Test-split metrics (mean over the 6 systems; per-system rows in `model_metrics.csv`)

| model | APFD | recall@0.2 | detect@0.2 | recall_new@0.2 | ROC-AUC | AP | Brier | log loss |
|---|---|---|---|---|---|---|---|---|
| random | 0.501 | 0.190 | 0.626 | 0.189 | 0.502 | 0.069 | | |
| id_order | 0.486 | 0.245 | 0.652 | 0.219 | 0.500 | 0.069 | | |
| dfe | 0.774 | 0.627 | 0.855 | 0.550 | 0.830 | 0.246 | | |
| lr | 0.795 | 0.634 | 0.856 | 0.560 | 0.874 | 0.327 | 0.058 | 0.204 |
| lgbm | 0.784 | 0.623 | 0.859 | 0.556 | 0.888 | 0.458 | 0.050 | 0.179 |

LR and LightGBM are within 0.03 of DFE on every ranking metric (APFD, recall@0.2, detect@0.2, recall_new@0.2). LightGBM is clearly better on the probability metrics (AP, Brier, log loss) and ROC-AUC.

## Paired block bootstrap (block 10, 2000 resamples; mean difference [95% CI])

| system | failing cycles | APFD dfe / lr / lgbm | lgbm − dfe, APFD | lgbm − lr, APFD | lgbm − dfe, recall@0.2 | lgbm − lr, recall@0.2 |
|---|---|---|---|---|---|---|
| system13 | 147 | 0.773 / 0.778 / 0.782 | +0.009 [−0.006, +0.026] | +0.004 [−0.006, +0.014] | −0.009 [−0.058, +0.035] | +0.016 [−0.003, +0.032] |
| system28 | 50 | 0.726 / 0.776 / 0.767 | +0.041 [−0.010, +0.108] | −0.009 [−0.033, +0.013] | +0.045 [−0.051, +0.160] | +0.004 [−0.057, +0.068] |
| system30 | 107 | 0.764 / 0.780 / 0.788 | +0.024 [+0.006, +0.042] | +0.008 [−0.005, +0.019] | +0.024 [−0.021, +0.075] | −0.002 [−0.039, +0.036] |
| system33 | 92 | 0.706 / 0.729 / 0.707 | +0.001 [−0.019, +0.020] | −0.022 [−0.045, +0.000] | −0.023 [−0.057, +0.007] | −0.038 [−0.081, −0.004] |
| system46 | 108 | 0.836 / 0.858 / 0.824 | −0.012 [−0.037, +0.011] | −0.034 [−0.060, −0.010] | −0.058 [−0.108, −0.012] | −0.056 [−0.102, −0.006] |
| system64 | 93 | 0.838 / 0.848 / 0.839 | +0.001 [−0.022, +0.022] | −0.009 [−0.025, +0.006] | −0.004 [−0.036, +0.023] | +0.007 [−0.023, +0.035] |

Of 24 intervals, 5 exclude 0, with no multiplicity correction:
- one favours LightGBM: system30, vs DFE, on APFD;
- four favour the simpler model, all against LightGBM: in system33 vs LR on recall@0.2, and in system46 vs DFE and LR.

Neither model beats DFE clearly.

## Checks
- **Leakage tests** (`tests/test_features.py`: brute force and perturbation, 6 systems): pass.
- **Label-flip check** (`tests/test_train.py`, system28): pass. The chosen C, config and best iteration are identical, and lr and lgbm scores are identical on every row.
- **Random-score check** (50 seeds): mean APFD 0.497–0.503 in all systems, within 0.5 ± 0.02.

## Notes
- LightGBM picks the smallest trees (15 leaves) in every system and stops early (best iteration 17–89), so the grid's lower edge is binding.
- LR picks C = 0.001, the lower edge of the grid, for system30 and system64, and C = 1, the upper edge, for system28.
- DFE has a much lower AP (0.246) than LightGBM (0.458) despite similar APFD.
- LR under-predicts its top-scored bin in system46: predicted 0.14, observed 0.27 (`figures/reliability_system46.png`).
