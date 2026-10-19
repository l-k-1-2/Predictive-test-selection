# Predictive test selection with conformal risk control 

## Purpose

Software organizations run a large number of tests. With AI generating code at an unprecedented rate, testing has become all the more frequent and important. This project asks:

> Based on testing history, can we predict which tests are likely to fail, and run only those, with a promise that, on average, no more than a certain percentage of failures will be missed?

The promise comes from a statistical method called **conformal risk control**. It sets the cut-off for "run this test" so that the share of missed failures stays below the set limit on average.


## Data

**Source:** the Enhanced Westermo dataset v1.0 ([Zenodo, DOI 10.5281/zenodo.8401356](https://doi.org/10.5281/zenodo.8401356), CC BY 4.0), derived from the Westermo test results data set (Strandberg, 2022). It records nightly regression testing of industrial network equipment.

There is one CSV file per test system. Each row is one execution of a test within a nightly test session, called a *cycle*. Columns:

| column | type | meaning, as found in the data |
|---|---|---|
| `Id` | int | identifier of a test execution; within a session it follows execution order |
| `Name` | int | identifier of the test case |
| `Duration` | int | runtime of the execution |
| `CalcPrio` | int | output column for prioritisation algorithms, always 0; dropped |
| `LastRun` | datetime | timestamp of this execution |
| `LastResults` | list of 0/1 | verdict history, oldest first, **including this execution's verdict** |
| `Verdict` | 0/1 | 1 = failed; only 0 and 1 occur, with no missing values |
| `Cycle` | int | test session number |

It covers 9 test systems, about 1 million test results and roughly 18 months of testing.

**Shape.** A *unit cycle* is a cycle with at least 20 tests, and repeated runs of a test within a cycle are merged; both are explained under Data Preprocessing. Systems marked ✓ are the six used for modelling.

| system | raw rows | rows after merging repeats | cycles | unit cycles | tests | failure rate | days covered |
|---|---|---|---|---|---|---|---|
| system11 | 41,976 | 16,669 | 326 | 133 | 167 | 0.113 | 184 |
| system13 ✓ | 81,858 | 78,116 | 2,281 | 896 | 127 | 0.042 | 546 |
| system21 | 65,796 | 48,989 | 1,105 | 415 | 145 | 0.051 | 353 |
| system28 ✓ | 68,905 | 56,362 | 1,165 | 452 | 241 | 0.069 | 410 |
| system30 ✓ | 184,815 | 147,783 | 2,295 | 867 | 243 | 0.039 | 545 |
| system33 ✓ | 167,260 | 131,801 | 2,469 | 847 | 217 | 0.042 | 545 |
| system46 ✓ | 187,431 | 140,396 | 2,407 | 851 | 240 | 0.039 | 546 |
| system64 ✓ | 165,978 | 133,576 | 2,228 | 822 | 259 | 0.055 | 544 |
| system96 | 72,799 | 54,713 | 921 | 323 | 216 | 0.072 | 214 |
| **total** | **1,036,818** | **808,405** | | | | | |

## Approach

The project was built in 3 steps. First, data preprocessing and exploratory analysis, to understand the data, remove noise and prevent data leakage. Second, modelling: logistic regression and LightGBM were trained and compared with a simple baseline, the decayed failure score (DFE). Third, conformal risk control was applied to the models' scores to choose which tests to run, with a 5% limit on missed failures.

Testing:
- The last 20% of the data, by time, is used for testing.
- Nothing is tuned on the test period: features, model settings and risk-control settings are all fixed before it is looked at.

## Data Preprocessing

### Cleaning
- **Repeated runs.** 22% of raw rows repeat a (cycle, test) pair: they are back-to-back re-runs, a median 1.2–4.2 minutes apart, and the verdicts disagree in only 1.4–7.0% of groups. Each repeated pair is merged into one row: failed if any run failed, with the mean duration.
- **Time order.** `cycle_order` is the rank of `Cycle`. It agrees with cycle time, the median `LastRun` of the cycle's rows (Spearman ρ ≥ 0.9999 in every system). Only 16 cycles, in 5 systems, are dated before the cycle before them.
- **Tie-break.** `idx` is the rank of `Id` within a cycle, i.e. execution order. It is used only to break ties between equal scores. Ranking by `idx` alone gives an APFD of 0.43–0.56, so execution order carries no information about failures. APFD (Average Percentage of Faults Detected) measures how early a ranking places the failing tests; 0.5 is random order, and it is defined under Modelling.

### Data insights
- **`LastResults` leaks the label.** Its last entry equals the current verdict in 97.5–99.9% of rows, against 91.9–95.6% for the previous verdict. Hence, omitted to avoid data leakage.
- **`LastRun` is the current execution's timestamp**, so it is not used for modelling.
- **The current `Duration`** can reflect the outcome (aborted or timed-out runs), so it is used only to compute the time cost.
- **Small cycles are mostly re-runs.** In the six systems, 61–66% of cycles have fewer than 20 tests. In 16–48% of their rows the test failed in its previous run, against 2.5–4.9% in larger cycles. They hold 4–7% of rows but 18–46% of failures. So:
  - every cycle is used to build the history features (see Feature selection);
  - training, evaluation and test selection use **unit cycles** (≥ 20 tests) only.
- **System selection.** A system is used if it has at least 300 failing unit cycles (unit cycles with at least one failed test), at least 20 tests and ρ ≥ 0.9. Six systems qualify.

### Feature selection

All 13 features describe a test's history before the current cycle, plus the cycle's size. They are computed per test in time order, with each value shifted by one run, so cycle c only sees cycles before c. Every feature is used by both models; there is no automatic feature selection.

Features engineered:

| feature | definition |
|---|---|
| `h_runs`, `h_fails` | number of prior runs, and of prior failures |
| `h_rate` | `h_fails / h_runs` |
| `h_last1` | 1 if the latest prior run failed |
| `h_fails5`, `h_fails20` | failures in the last 5 / 20 prior runs |
| `h_since_fail` | prior runs since the latest failure |
| `h_flips20` | pass↔fail changes within the last 20 prior runs |
| `h_dfe` | decayed failure score (DFE): each test's score P starts at 0; after every cycle, P ← 0.2·P, then P += 0.8 for each test that failed |
| `h_dur` | mean log(1 + duration) over prior runs |
| `h_gap` | cycles since the test's previous run |
| `h_age` | cycles since the test's first appearance |
| `j_ntests` | number of tests in the current cycle |

## Modelling

**Splits.** Each system is split by time over all its cycles: train 50%, validation 10%, calibration 20%, test 20%. Only unit-cycle rows are used.

**Models** are fitted separately for each system on train rows, with every setting chosen on validation rows. No class weights are used, so the model outputs stay probabilities.

| model | score / setup | tuning | chosen |
|---|---|---|---|
| DFE | the `h_dfe` feature, used directly as the score | – | – |
| logistic regression | median imputation with missing-value indicators → standardisation → L2 logistic regression | C ∈ {0.001, 0.01, 0.1, 1} by validation average precision | C = 0.001 (systems 30, 64), 0.01 (33, 46), 0.1 (13), 1 (28) |
| LightGBM | learning rate 0.05, feature and bagging fraction 0.8, early stopping after 100 rounds on validation log loss (max 3000) | leaves ∈ {15, 63} × min data in leaf ∈ {20, 200}, by validation average precision | 15 leaves in every system; min data 200 (13, 28, 30) or 20 (33, 46, 64); 17–89 rounds |

**Metrics** on test unit cycles:
- **Ranking**, averaged over failing cycles; tests are ranked by score within each cycle:
  - APFD = 1 − (sum of failed tests' positions) / (n·m) + 1/(2n), for n tests and m failures;
  - recall@20% (share of failures in the top 20% of tests) and detect@20% (at least one failure in the top 20%);
  - recall of new failures, i.e. tests whose previous run passed.
- **Probability**, over all test rows: ROC-AUC, average precision (AP), Brier score and log loss. Brier and log loss are reported for the two models only, as the DFE score is not a probability.

**Results** (mean over the six systems):

| model | APFD | recall@20% | detect@20% | new-failure recall@20% | ROC-AUC | AP | Brier | log loss |
|---|---|---|---|---|---|---|---|---|
| DFE | 0.774 | 0.627 | 0.855 | 0.550 | 0.830 | 0.246 | | |
| logistic regression | 0.795 | 0.634 | 0.856 | 0.560 | 0.874 | 0.327 | 0.058 | 0.204 |
| **LightGBM** | 0.784 | 0.623 | 0.859 | 0.556 | **0.888** | **0.458** | **0.050** | **0.179** |

On ranking, the three are effectively tied. LightGBM has the best probability metrics, and it is the main score for risk control, chosen before the test period was looked at; logistic regression and DFE are kept for comparison.

## Conformal Risk Control

### What it is
Conformal risk control (Angelopoulos et al.) turns a model's scores into a run/skip rule that comes with a statistical promise. In each session, every test whose score s is at or above a threshold τ is run. For a session whose failed tests are F, the loss is the share of failures missed:

$$L(\tau) = \frac{|\{f \in F : s_f < \tau\}|}{|F|}$$

The promise: if the calibration sessions and a new session are exchangeable (roughly, the new session behaves like the calibration period), the calibrated threshold τ̂ gives E[L(τ̂)] ≤ α, where α is the chosen limit on missed failures. It holds for any model, because the loss lies between 0 and 1 and can only fall as more tests run. It is a promise about the average over sessions, not about any single session.

### Calibration
τ̂ is set on the n failing sessions of the calibration period, which comes after training and before testing. L_i is the loss in the i-th of them:

$$\hat\tau = \max\Big\{\tau : \frac{\sum_{i=1}^{n} L_i(\tau) + 1}{n+1} \le \alpha\Big\}$$

This is the highest threshold, and so the fewest tests, whose average calibration loss stays within α after a small correction for the unseen session (the +1). The main target is α = 0.05, and α = 0.10 is also reported. τ̂ is then fixed and applied unchanged to the test period.

The systems change over time (drift), so two variants re-calibrate during the test period as each session's results arrive:
- **Rolling:** τ̂ is recomputed from the latest 50 failing sessions.
- **Adaptive (ACI, Gibbs & Candès 2021):** the same, but the working target α is also nudged after each session: up after a session with few misses, down after one with many.

### Implementation
- **Closed form.** τ̂ is found by sorting all calibration failure scores, weighting each by 1/|F| so every session counts equally, and taking a cumulative sum. No grid search is needed.
- **Unit tests** compare it with an exact brute-force search.
- **Comparator.** A fixed-share rule runs the top share of tests in every session, with the share calibrated by the same formula.
- **Validity check.** The calibration and test sessions are pooled and split at random 50/50, 1,000 times. τ̂ is computed on one half and the loss measured on the other. The mean loss never exceeded α for any system or α, which confirms both the implementation and the promise when the data are exchangeable.

## Models with Conformal Risk Control
Test period, α = 0.05, mean over the six systems:

| score | missed failures | tests run | test time used |
|---|---|---|---|
| **LightGBM** | 2.3% | 77% | 86% |
| logistic regression | 4.2% | 70% | 75% |
| DFE | 0% | 100% | 100% |

- **The promise held at the 5% target.** LightGBM missed 2.3% of failures on average, at most 5.1% in any system, while skipping 23% of tests and 14% of testing time.
- **The model is what makes savings possible.** DFE gives the same lowest score to most tests that haven't failed recently, including most new failures, so it has to run everything to stay within 5%. Logistic regression saves more, but goes over 5% in two systems.
- **Drift weakens the promise.** At α = 0.10, two systems went over the limit (13.9% and 14.7%). Rolling re-calibration brought them down to 7.7% and 10.4%. ACI met the average target almost exactly (4.8% at α = 0.05), but had worse stretches, up to 9% missed over 50 sessions.
- **A fixed share per session fails.** The fixed-share rule missed far more failures than allowed (above 5% in 4 of 6 systems), because it can't run more tests when a session has more risky ones.

## How to run

Python 3.11. From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download `TSP_Westermo_Dataset.zip` from the Zenodo link above and unzip it into `data/raw/westermo/`, so the nine CSV files sit directly in that folder.

Run the scripts in this order. Each one reads the outputs of the ones before it:

| # | script | what it does |
|---|---|---|
| 1 | `src/schema.py` | records the raw columns and their observed meaning |
| 2 | `src/prepare.py` | merges repeated runs, orders cycles in time, flags unit cycles |
| 3 | `src/lastresults.py` | checks what the `LastResults` column contains |
| 4 | `src/system_stats.py` | per-system statistics |
| 5 | `src/select_systems.py` | chooses the six systems to model |
| 6 | `src/baselines.py` | APFD of execution order and DFE |
| 7 | `src/features.py` | builds the 13 history features |
| 8 | `src/train.py` | trains logistic regression and LightGBM, saves their scores |
| 9 | `src/evaluate.py` | test-period metrics and block bootstrap |
| 10 | `src/empirical_bayes.py` | smoothed failure rates over each test's last 20 runs |
| 11 | `src/crc.py` | conformal risk control and the validity check |
| 12 | `src/drift.py` | rolling and adaptive re-calibration over time |

Then run the tests with `pytest -q`. Some of them read the pipeline's outputs, so run them after step 12.

Results are written to `results/` and figures to `figures/`.