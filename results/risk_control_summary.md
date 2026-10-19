# Risk control summary — empirical Bayes, conformal risk control, drift

Defaults: score = lgbm, loss = frac, alpha = 0.05 and 0.10, W = 50, gamma = 0.05, moving window = 50 failing cycles. Six systems, unit cycles only. Every grid value is in `eb_*.csv`, `crc_*.csv` and `drift*.csv`; this page uses only the defaults.

## A. Empirical Bayes

a and b are fitted on train-split blocks of 20 runs only. For rows with the same n, raw, Laplace and EB rank rows the same way, and n = 20 for almost every row. EB's gain is calibrated probabilities, not ranking (APFD differs by at most 0.004 and ROC-AUC by at most 0.003 across the three scores).

| system | a | b | prior mean | blocks | converged | at bound |
|---|---|---|---|---|---|---|
| system13 | 0.093 | 3.670 | 0.025 | 1741 | True | False |
| system28 | 0.077 | 0.761 | 0.091 | 1648 | True | False |
| system30 | 0.076 | 2.219 | 0.033 | 3266 | True | False |
| system33 | 0.049 | 1.841 | 0.026 | 3195 | True | False |
| system46 | 0.090 | 1.305 | 0.064 | 3336 | True | False |
| system64 | 0.069 | 1.831 | 0.037 | 3036 | True | False |

Brier and log loss on test unit rows, each as raw / Laplace / EB. Left: all rows. Right: rows with k = 0.

| system | Brier raw | Brier Lap | Brier EB | logloss raw | logloss Lap | logloss EB | k=0 Brier raw | k=0 Brier Lap | k=0 Brier EB | k=0 logloss raw | k=0 logloss Lap | k=0 logloss EB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system13 | 0.072 | 0.073 | 0.071 | 0.294 | 0.249 | 0.241 | 0.011 | 0.012 | 0.011 | 0.150 | 0.080 | 0.064 |
| system28 | 0.048 | 0.050 | 0.048 | 0.263 | 0.197 | 0.190 | 0.015 | 0.016 | 0.015 | 0.203 | 0.091 | 0.085 |
| system30 | 0.067 | 0.068 | 0.066 | 0.273 | 0.236 | 0.223 | 0.010 | 0.011 | 0.010 | 0.138 | 0.077 | 0.060 |
| system33 | 0.081 | 0.082 | 0.080 | 0.372 | 0.281 | 0.290 | 0.021 | 0.021 | 0.021 | 0.290 | 0.110 | 0.130 |
| system46 | 0.028 | 0.029 | 0.028 | 0.157 | 0.130 | 0.109 | 0.007 | 0.009 | 0.007 | 0.103 | 0.069 | 0.045 |
| system64 | 0.053 | 0.054 | 0.052 | 0.255 | 0.204 | 0.194 | 0.012 | 0.013 | 0.012 | 0.165 | 0.083 | 0.072 |
| mean | 0.058 | 0.059 | 0.058 | 0.269 | 0.216 | 0.208 | 0.013 | 0.014 | 0.013 | 0.175 | 0.085 | 0.076 |

| system | APFD raw | APFD Lap | APFD EB | AUC raw | AUC Lap | AUC EB |
|---|---|---|---|---|---|---|
| system13 | 0.753 | 0.753 | 0.753 | 0.837 | 0.838 | 0.837 |
| system28 | 0.714 | 0.713 | 0.714 | 0.756 | 0.758 | 0.757 |
| system30 | 0.747 | 0.751 | 0.751 | 0.836 | 0.836 | 0.836 |
| system33 | 0.688 | 0.690 | 0.690 | 0.773 | 0.774 | 0.774 |
| system46 | 0.836 | 0.836 | 0.836 | 0.849 | 0.849 | 0.849 |
| system64 | 0.833 | 0.833 | 0.833 | 0.825 | 0.826 | 0.826 |
| mean | 0.762 | 0.763 | 0.763 | 0.813 | 0.813 | 0.813 |

## B. Conformal risk control (lgbm, frac loss, test split)

### alpha = 0.05: static threshold and fraction CRC

| system | risk | sel_frac | time_frac | fraction CRC sel_frac | n_cal |
|---|---|---|---|---|---|
| system13 | 0.014 | 0.724 | 0.890 | 0.497 | 185 |
| system28 | 0.034 | 0.702 | 0.790 | 0.467 | 67 |
| system30 | 0.003 | 0.892 | 0.951 | 0.784 | 150 |
| system33 | 0.027 | 0.865 | 0.896 | 0.613 | 129 |
| system46 | 0.007 | 0.841 | 0.931 | 0.793 | 143 |
| system64 | 0.051 | 0.576 | 0.679 | 0.533 | 197 |
| mean | 0.023 | 0.767 | 0.856 | 0.614 |  |

lgbm − dfe, static threshold, alpha = 0.05 (paired block bootstrap, tau fixed at its cal value, so the CI covers test-period variation only):

| system | sel_frac diff [95% CI] | risk diff [95% CI] |
|---|---|---|
| system13 | -0.277 [-0.326, -0.224] | +0.013 [+0.005, +0.024] |
| system28 | -0.299 [-0.393, -0.211] | +0.035 [+0.000, +0.089] |
| system30 | -0.108 [-0.129, -0.086] | +0.003 [+0.001, +0.007] |
| system33 | -0.135 [-0.156, -0.115] | +0.027 [+0.007, +0.057] |
| system46 | -0.160 [-0.180, -0.137] | +0.007 [+0.000, +0.019] |
| system64 | -0.425 [-0.482, -0.369] | +0.052 [+0.027, +0.083] |
| mean of differences | -0.234 | +0.023 |

### alpha = 0.1: static threshold and fraction CRC

| system | risk | sel_frac | time_frac | fraction CRC sel_frac | n_cal |
|---|---|---|---|---|---|
| system13 | 0.042 | 0.619 | 0.774 | 0.356 | 185 |
| system28 | 0.139 | 0.519 | 0.603 | 0.347 | 67 |
| system30 | 0.044 | 0.703 | 0.799 | 0.444 | 150 |
| system33 | 0.033 | 0.828 | 0.867 | 0.544 | 129 |
| system46 | 0.031 | 0.775 | 0.849 | 0.644 | 143 |
| system64 | 0.147 | 0.295 | 0.309 | 0.364 | 197 |
| mean | 0.073 | 0.623 | 0.700 | 0.450 |  |

lgbm − dfe, static threshold, alpha = 0.1 (paired block bootstrap, tau fixed at its cal value, so the CI covers test-period variation only):

| system | sel_frac diff [95% CI] | risk diff [95% CI] |
|---|---|---|
| system13 | -0.382 [-0.441, -0.314] | +0.042 [+0.019, +0.070] |
| system28 | -0.481 [-0.614, -0.360] | +0.140 [+0.037, +0.287] |
| system30 | -0.297 [-0.362, -0.238] | +0.045 [+0.020, +0.075] |
| system33 | -0.172 [-0.204, -0.141] | +0.034 [+0.011, +0.066] |
| system46 | -0.225 [-0.258, -0.188] | +0.032 [+0.008, +0.061] |
| system64 | +0.005 [-0.049, +0.063] | +0.008 [-0.013, +0.030] |
| mean of differences | -0.259 | +0.050 |

### Rolling-50 and ACI-0.05

Test labels become known after each cycle, which assumes skipped tests run later, for example in a following full session. The ACI bound (max(alpha, 1−alpha) + gamma)/(gamma·T) is about 0.13 or more at gamma = 0.05 for T = 50–150. It checks the implementation, not performance.

alpha = 0.05; maxwin = max_window_risk

| system | static risk | static sel_frac | static maxwin | rolling-50 risk | rolling-50 sel_frac | rolling-50 maxwin | aci-0.05 risk | aci-0.05 sel_frac | aci-0.05 maxwin |
|---|---|---|---|---|---|---|---|---|---|
| system13 | 0.014 | 0.724 | 0.024 | 0.023 | 0.741 | 0.054 | 0.050 | 0.621 | 0.076 |
| system28 | 0.034 | 0.702 | 0.034 | 0.027 | 0.771 | 0.027 | 0.042 | 0.763 | 0.042 |
| system30 | 0.003 | 0.892 | 0.005 | 0.026 | 0.812 | 0.050 | 0.056 | 0.757 | 0.086 |
| system33 | 0.027 | 0.865 | 0.040 | 0.028 | 0.899 | 0.040 | 0.042 | 0.838 | 0.057 |
| system46 | 0.007 | 0.841 | 0.014 | 0.015 | 0.796 | 0.032 | 0.044 | 0.670 | 0.092 |
| system64 | 0.051 | 0.576 | 0.063 | 0.042 | 0.672 | 0.048 | 0.052 | 0.646 | 0.055 |
| mean | 0.023 | 0.767 | 0.030 | 0.027 | 0.782 | 0.042 | 0.048 | 0.716 | 0.068 |

alpha = 0.1; maxwin = max_window_risk

| system | static risk | static sel_frac | static maxwin | rolling-50 risk | rolling-50 sel_frac | rolling-50 maxwin | aci-0.05 risk | aci-0.05 sel_frac | aci-0.05 maxwin |
|---|---|---|---|---|---|---|---|---|---|
| system13 | 0.042 | 0.619 | 0.056 | 0.071 | 0.575 | 0.108 | 0.106 | 0.488 | 0.145 |
| system28 | 0.139 | 0.519 | 0.139 | 0.077 | 0.554 | 0.077 | 0.077 | 0.538 | 0.077 |
| system30 | 0.044 | 0.703 | 0.055 | 0.064 | 0.623 | 0.092 | 0.097 | 0.543 | 0.123 |
| system33 | 0.033 | 0.828 | 0.051 | 0.059 | 0.752 | 0.070 | 0.088 | 0.680 | 0.117 |
| system46 | 0.031 | 0.775 | 0.041 | 0.049 | 0.638 | 0.105 | 0.088 | 0.499 | 0.174 |
| system64 | 0.147 | 0.295 | 0.198 | 0.104 | 0.379 | 0.117 | 0.105 | 0.389 | 0.118 |
| mean | 0.073 | 0.623 | 0.090 | 0.071 | 0.587 | 0.095 | 0.093 | 0.523 | 0.126 |

## Exchangeable gate (lgbm, 1000 random 50/50 splits of cal + test cycles)

| alpha | gate (all systems, both losses) | max(mean risk − alpha) | max(mean risk − alpha − 2·SE) |
|---|---|---|---|
| 0.01 | PASS | -0.002 | -0.003 |
| 0.02 | PASS | -0.002 | -0.003 |
| 0.05 | PASS | -0.001 | -0.003 |
| 0.1 | PASS | -0.001 | -0.003 |
| 0.2 | PASS | -0.000 | -0.003 |

## Ranking comparison: lgbm vs dfe on test unit cycles (`model_bootstrap.csv`)

| system | APFD lgbm | APFD dfe | lgbm − dfe [95% CI] |
|---|---|---|---|
| system13 | 0.782 | 0.773 | +0.009 [-0.006, +0.026] |
| system28 | 0.767 | 0.726 | +0.041 [-0.010, +0.108] |
| system30 | 0.788 | 0.764 | +0.024 [+0.006, +0.042] |
| system33 | 0.707 | 0.706 | +0.001 [-0.019, +0.020] |
| system46 | 0.824 | 0.836 | -0.012 [-0.037, +0.011] |
| system64 | 0.839 | 0.838 | +0.001 [-0.022, +0.022] |
| mean | 0.784 | 0.774 | +0.011 |

## Check statuses

| check | status |
|---|---|
| pytest -q (56 tests incl. test_crc.py) | PASS |
| beta fit converged | PASS (6/6; at bound: 0) |
| exchangeable gate, every alpha | PASS |
| bound_ok, every aci run | PASS (36 runs) |
| static lgbm threshold sel_frac non-increasing in alpha (both losses) | PASS |
