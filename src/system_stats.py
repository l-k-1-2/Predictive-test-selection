import glob
import os

import pandas as pd

ROWS = "data/processed/rows"
CYCLES = "data/processed/cycles"
OUT = "results/system_stats.csv"
COLUMNS = ["system", "n_cycles", "n_tests", "rows", "tests_per_cycle", "failure_rate", "failing_cycles", "failing_cycle_rate",
           "failed_per_failing_cycle", "new_failure_share", "first_time", "last_time", "spearman_cycle_time", "lastrun_decrease_share",
           "n_dup", "n_bad_duration", "lr_prev_match", "lr_prefix_match", "lr_longer", "lr_current_match",
           "unit_cycles", "failing_unit_cycles", "small_cycle_share", "small_row_share", "small_fail_share", "small_prev_fail_rate",
           "unit_prev_fail_rate", "cycles_per_date", "unit_cycles_per_date", "n_cycle_inversions", "max_inversion_h",
           "dup_minutes_median", "dup_mixed_share"]


def prev_failed(d):
    # d is sorted by test, then cycle_order
    same = (d.test_id == d.test_id.shift()).to_numpy()
    return d.failed.shift(fill_value=False).to_numpy() & same


def system_stats(name, checks, lr):
    d = pd.read_parquet(f"{ROWS}/{name}.parquet", columns=["cycle_order", "test_id", "failed"])
    d = d.sort_values(["test_id", "cycle_order"], kind="stable")
    c = pd.read_parquet(f"{CYCLES}/{name}.parquet")
    failed, pf = d.failed.to_numpy(), prev_failed(d)
    small = ~c.unit.to_numpy()[d.cycle_order.to_numpy()]
    failing = int(c.failing.sum())
    dates = c.cycle_time.dt.date.nunique()
    return dict(system=name, n_cycles=len(c), n_tests=d.test_id.nunique(), rows=len(d), tests_per_cycle=len(d) / len(c),
                failure_rate=failed.mean(), failing_cycles=failing, failing_cycle_rate=failing / len(c),
                failed_per_failing_cycle=c.n_failed.sum() / failing, new_failure_share=(failed & ~pf).sum() / failed.sum(),
                first_time=c.cycle_time.min(), last_time=c.cycle_time.max(),
                spearman_cycle_time=checks["spearman_cycle_time"], lastrun_decrease_share=checks["lastrun_decrease_share"],
                n_dup=int(checks["n_dup"]), n_bad_duration=int(checks["n_bad_duration"]),
                lr_prev_match=lr["lr_prev_match"], lr_prefix_match=lr["lr_prefix_match"], lr_longer=lr["lr_longer"],
                lr_current_match=lr["lr_current_match"],
                unit_cycles=int(c.unit.sum()), failing_unit_cycles=int((c.unit & c.failing).sum()), small_cycle_share=1 - c.unit.mean(),
                small_row_share=small.mean(), small_fail_share=failed[small].sum() / failed.sum(),
                small_prev_fail_rate=pf[small].mean(), unit_prev_fail_rate=pf[~small].mean(),
                cycles_per_date=len(c) / dates, unit_cycles_per_date=c.unit.sum() / dates,
                n_cycle_inversions=int(checks["n_cycle_inversions"]), max_inversion_h=checks["max_inversion_h"],
                dup_minutes_median=checks["dup_minutes_median"], dup_mixed_share=checks["dup_mixed_share"])


def main():
    checks = pd.read_csv("data/processed/prepare_checks.csv").set_index("system")
    lr = pd.read_csv("data/processed/lastresults.csv").set_index("system")
    out = []
    for path in sorted(glob.glob(f"{ROWS}/*.parquet")):
        name = os.path.basename(path)[:-len(".parquet")]
        out.append(system_stats(name, checks.loc[name], lr.loc[name]))
    assert len(out) == 9
    pd.DataFrame(out)[COLUMNS].to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
