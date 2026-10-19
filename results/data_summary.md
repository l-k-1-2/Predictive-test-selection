# Data summary

**Outcome:** 9 systems cleaned and checked, 6 selected. Training, model choice and evaluation use **unit cycles**, sessions with at least 20 tests.

## Schema and cleaning
- Columns and dtypes are listed in `data_schema.md`. `Name` is already an integer; `test_id` is a dense 0..T−1 code of it. `CalcPrio` is dropped.
- `LastRun` is the timestamp of the row's **own** execution, not the previous one (`data_schema.md`). So `cycle_time` is the median `LastRun` of the cycle's own rows.
- Timestamps are shifted to 1970–1971, so only relative time is meaningful.
- `Verdict` is only 0 or 1, with no nulls, in all 1,036,818 raw rows. Duration is never missing or negative. LastRun and LastResults all parse, and LastResults holds only 0 and 1.
- Merging duplicate (Cycle, Name) rows leaves 808,405 rows. Per system, 3,742 rows merge away in system13 (4.6%), up to 47,035 in system46 (25%), and 25,307 in system11 (60%).
- Duplicates are repeat runs in one session: a median 1.2–4.2 min from first to last run, with differing verdicts in only 1.4–7.0% of groups. `duration` is their mean, so it understates a repeated test's cost. Time-based results inherit this.
- Cached intermediate files: `data/processed/prepare_checks.csv` and `data/processed/lastresults.csv`.

## Order check
Spearman ρ(Cycle, cycle_time) ≥ 0.9999 in all 9 systems, so `cycle_order` is the dense rank of Cycle. LastRun decreases in at most 0.3% of a test's consecutive rows (system21, whose worst test reaches 3.7%). 16 cycles in 5 systems are dated before the previous cycle, by at most 217 h (system21). That is too few to matter.

## Cycle sizes and unit cycles
In the selected systems:
- There are 3.3–5.2 cycles per calendar date, and 61–66% of cycles have fewer than 20 tests.
- Small cycles are mostly re-runs. In 16–48% of their rows the test failed in its previous run, against 2.5–4.9% in unit cycles. They hold 4–7% of rows but 18–46% of failures.

A unit cycle has ≥ 20 tests; there are 1.3–1.8 per date, i.e. the nightly sessions.
- History uses every cycle.
- Selection, training and evaluation use unit cycles only.
- Reason: a 1-test cycle scores APFD 0.5 under any model, and small cycles would dominate per-cycle cost.

The failure rate per unit cycle rises late in the data (`figures/failing_rate.png`): it averages 0.062–0.129 in the test split, against 0.014–0.101 in cal.

## LastResults (every cleaned row with an earlier observed row)
- The most recent entry is the **end** of the list (oldest first).
- `lr_current_match` is 0.975–0.999 and `lr_prev_match` 0.919–0.956. The current match is higher in every system.
- **The column contains the current outcome. Using LastResults as a feature on this dataset would leak the label.**
- The best-fitting alignment is the current verdict followed by the earlier verdicts. `lr_prefix_match` is 0.662–0.954 (system11: 0.377) and `lr_longer` is 0.047–0.435 (system11: 0.956). Lists reaching before the dataset window explain part of this; repeat runs within a session probably explain the rest (not tested).

## Tie-break
Id-order mean APFD over failing unit cycles is 0.431–0.562, so execution order carries no outcome information. `idx`, the rank of `Id` within its cycle, is used only to break ties between equal scores.

## Baselines and selected systems
Mean APFD over failing unit cycles. DFE (α = 0.8) scores each cycle with P from before it, and P is updated after every cycle.

| system | cycles | unit cycles | tests | failing unit cycles | new-failure share | id-order APFD | DFE APFD |
|---|---|---|---|---|---|---|---|
| system11 (excluded) | 326 | 133 | 167 | 125 | 0.245 | 0.562 | 0.864 |
| system13 | 2281 | 896 | 127 | 616 | 0.716 | 0.439 | 0.848 |
| system21 (excluded) | 1105 | 415 | 145 | 282 | 0.445 | 0.454 | 0.873 |
| system28 | 1165 | 452 | 241 | 371 | 0.358 | 0.431 | 0.863 |
| system30 | 2295 | 867 | 243 | 554 | 0.724 | 0.464 | 0.861 |
| system33 | 2469 | 847 | 217 | 498 | 0.761 | 0.534 | 0.847 |
| system46 | 2407 | 851 | 240 | 626 | 0.578 | 0.558 | 0.872 |
| system64 | 2228 | 822 | 259 | 633 | 0.528 | 0.474 | 0.888 |
| system96 (excluded) | 921 | 323 | 216 | 268 | 0.573 | 0.480 | 0.865 |

Criteria: failing unit cycles ≥ 300, n_tests ≥ 20, ρ ≥ 0.9. Three systems fall short on failing unit cycles: system11, system21 and system96.
