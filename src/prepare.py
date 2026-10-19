import gc
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

RAW = "data/raw/westermo"
ROWS = "data/processed/rows"
CYCLES = "data/processed/cycles"
CHECKS = "data/processed/prepare_checks.csv"  # per-system counts and order checks, read again by system_stats.py
COLS = ["Id", "Name", "Duration", "LastRun", "LastResults", "Verdict", "Cycle"]
EPOCH = pd.Timestamp("1970-01-01")
CHUNK = 20000
MIN_TESTS = 20  # unit cycles: smaller cycles are mostly re-runs of recent failures


def system_name(path):
    return os.path.basename(path).split("_")[1]


def load_raw(path):
    d = pd.read_csv(path, sep=";", usecols=COLS, engine="pyarrow")
    assert d.Id.is_unique, "Id is not unique, lowest-Id rule is ambiguous"
    assert d.Verdict.isin([0, 1]).all(), "Verdict outside {0, 1}"
    d["LastRun"] = pd.to_datetime(d.LastRun, format="%Y-%m-%d %H:%M:%S")
    d = d.sort_values("Id", kind="stable").reset_index(drop=True)
    d["Duration"] = d.Duration.astype(float)
    bad = d.Duration.isna() | (d.Duration < 0)
    d.loc[bad, "Duration"] = np.nan
    return d, int(bad.sum())


def duplicate_checks(d):
    g = d[d.duplicated(["Cycle", "Name"], keep=False)].groupby(["Cycle", "Name"])
    spread_min = (g.LastRun.max() - g.LastRun.min()).dt.total_seconds() / 60
    return spread_min.median(), (g.Verdict.nunique() > 1).mean()


def merge_duplicates(d):
    keys = ["Cycle", "Name"]
    # d is sorted by Id, so the first row of each group is the lowest Id
    first = d.drop_duplicates(keys, keep="first").set_index(keys)
    agg = d.groupby(keys).agg(Verdict=("Verdict", "max"), Duration=("Duration", "mean"))
    first["Verdict"] = agg.Verdict
    first["Duration"] = agg.Duration
    return first.reset_index()


def parse_results(strings):
    values, counts = [], []
    for i in range(0, len(strings), CHUNK):
        s = strings.iloc[i : i + CHUNK]
        lens = s.str.len().to_numpy()
        arr = np.frombuffer("".join(s).encode("ascii"), dtype=np.uint8)
        digit = (arr >= 48) & (arr <= 57)
        comma = arr == 44
        allowed = digit | comma | (arr == 91) | (arr == 93) | (arr == 32)
        assert allowed.all(), "unexpected character in LastResults"
        assert not (digit[1:] & digit[:-1]).any(), "multi-digit entry in LastResults"
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        nd = np.add.reduceat(digit.astype(np.int64), starts)
        nc = np.add.reduceat(comma.astype(np.int64), starts)
        assert ((nd == nc + 1) | ((nd == 0) & (nc == 0))).all(), "LastResults list is malformed"
        values.append((arr[digit] - 48).astype(np.int8))
        counts.append(nd)
    values = np.concatenate(values)
    counts = np.concatenate(counts)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(values)), np.unique(values)


def clean(d):
    d = d.rename(columns={"Cycle": "cycle", "Name": "test", "Id": "id", "Duration": "duration", "LastRun": "last_run"})
    d["failed"] = d.Verdict == 1
    _, d["test_id"] = np.unique(d.test.to_numpy(), return_inverse=True)
    d = d.sort_values(["cycle", "id"], kind="stable").reset_index(drop=True)
    d["idx"] = d.groupby("cycle").id.rank(method="first").astype(int) - 1
    return d


def cycle_table(d):
    # LastRun is the timestamp of the row's own execution (see results/data_schema.md)
    d = d.assign(run_s=(d.last_run - EPOCH).dt.total_seconds())
    g = d.groupby("cycle")
    c = pd.DataFrame({"cycle_time_s": g.run_s.median(), "n_tests": g.size(), "n_failed": g.failed.sum()}).reset_index()
    c["cycle_time"] = EPOCH + pd.to_timedelta(c.cycle_time_s, unit="s")
    c["failing"] = c.n_failed > 0
    c["unit"] = c.n_tests >= MIN_TESTS
    return c


def order_checks(d, c):
    s = d.sort_values(["test_id", "cycle"], kind="stable")
    same = (s.test_id == s.test_id.shift()).to_numpy()
    dec = (s.last_run < s.last_run.shift()).to_numpy()
    pooled = dec[same].mean()
    per_test = pd.Series(dec[same]).groupby(s.test_id.to_numpy()[same]).mean()
    rho = spearmanr(c.cycle, c.cycle_time_s).statistic
    # cycles whose time is earlier than the previous cycle's
    gap_h = c.cycle_time_s.diff().to_numpy()[1:] / 3600
    return rho, pooled, per_test.max(), int((gap_h < 0).sum()), max(0.0, -gap_h.min())


def check_integrity(d, c):
    assert not d.duplicated(["cycle", "test"]).any()
    assert (c.cycle_order.to_numpy() == np.arange(len(c))).all()
    assert c.cycle.is_monotonic_increasing


def process(path):
    name = system_name(path)
    raw, n_bad = load_raw(path)
    n_raw = len(raw)
    dup_min, dup_mixed = duplicate_checks(raw)
    d = merge_duplicates(raw)
    del raw
    gc.collect()
    d = clean(d)
    c = cycle_table(d)
    rho, dec_pooled, dec_max, n_inv, max_inv = order_checks(d, c)
    row = dict(system=name, rows_raw=n_raw, rows=len(d), n_cycles=len(c), n_tests=d.test_id.nunique(), n_dup=n_raw - len(d),
               n_bad_duration=n_bad, spearman_cycle_time=rho, lastrun_decrease_share=dec_pooled,
               lastrun_decrease_max_test=dec_max, n_cycle_inversions=n_inv, max_inversion_h=max_inv,
               dup_minutes_median=dup_min, dup_mixed_share=dup_mixed)
    print(row)
    if not rho >= 0.9:
        return row, False

    c["cycle_order"] = c.cycle.rank(method="dense").astype(int) - 1
    d = d.merge(c[["cycle", "cycle_order"]], on="cycle", how="left")
    check_integrity(d, c)
    lr, lr_values = parse_results(d.LastResults)
    row["lr_values"] = " ".join(map(str, lr_values))
    out = d[["cycle", "cycle_order", "test", "test_id", "idx", "duration", "failed", "last_run"]]
    t = pa.Table.from_pandas(out, preserve_index=False).append_column("last_results", lr)
    pq.write_table(t, f"{ROWS}/{name}.parquet")
    c[["cycle", "cycle_order", "cycle_time", "n_tests", "n_failed", "failing", "unit"]].to_parquet(f"{CYCLES}/{name}.parquet", index=False)
    return row, True


def main():
    os.makedirs(ROWS, exist_ok=True)
    os.makedirs(CYCLES, exist_ok=True)
    done = pd.read_csv(CHECKS).set_index("system").to_dict("index") if os.path.exists(CHECKS) else {}
    failed_order = []
    for path in sorted(glob.glob(f"{RAW}/*.csv")):
        name = system_name(path)
        if name in done and os.path.exists(f"{ROWS}/{name}.parquet") and os.path.exists(f"{CYCLES}/{name}.parquet"):
            print(name, "cached")
            continue
        row, ok = process(path)
        if not ok:
            failed_order.append(name)
        done[name] = {k: v for k, v in row.items() if k != "system"}
        pd.DataFrame.from_dict(done, orient="index").rename_axis("system").reset_index().to_csv(CHECKS, index=False)
        gc.collect()
    if failed_order:
        print("STOP: spearman(cycle, cycle_time) < 0.9 for", failed_order)
        sys.exit(1)


if __name__ == "__main__":
    main()
