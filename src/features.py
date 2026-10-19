import json
import os

import numpy as np
import pandas as pd

from common import dfe_update

ROWS = "data/processed/rows"
CYCLES = "data/processed/cycles"
OUT = "data/processed/features"
FEATURES = ["h_runs", "h_fails", "h_rate", "h_last1", "h_fails5", "h_fails20", "h_since_fail", "h_flips20", "h_dfe", "h_dur", "h_gap",
            "h_age", "j_ntests"]
KEYS = ["cycle_order", "split", "unit", "test", "test_id", "idx", "duration", "failed"]


def split_of(cycle_order, n_cycles):
    # integer arithmetic keeps the floor exact
    b1, b2, b3 = 5 * n_cycles // 10, 6 * n_cycles // 10, 8 * n_cycles // 10
    return np.select([cycle_order < b1, cycle_order < b2, cycle_order < b3], ["train", "val", "cal"], "test")


def load_table(name):
    d = pd.read_parquet(f"{ROWS}/{name}.parquet", columns=["cycle_order", "test", "test_id", "idx", "duration", "failed"])
    c = pd.read_parquet(f"{CYCLES}/{name}.parquet", columns=["cycle_order", "unit"])
    d = d.merge(c, on="cycle_order", how="left").sort_values(["cycle_order", "idx"], kind="stable").reset_index(drop=True)
    d["split"] = split_of(d.cycle_order.to_numpy(), len(c))
    return d


def dfe_before(d):
    # d is sorted by cycle_order; each row gets P as it was before its cycle
    out = np.zeros(len(d))
    P = np.zeros(d.test_id.max() + 1)
    ids, failed = d.test_id.to_numpy(), d.failed.to_numpy()
    bounds = np.flatnonzero(np.diff(d.cycle_order.to_numpy())) + 1
    for s, e in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(d)]])):
        out[s:e] = P[ids[s:e]]
        dfe_update(P, ids[s:e][failed[s:e]])
    return out


def rolling_sum(x, tid, window):
    return x.groupby(tid).transform(lambda v: v.rolling(window, min_periods=1).sum())


def history_features(d):
    s = d.sort_values(["test_id", "cycle_order"], kind="stable")
    tid, co = s.test_id, s.cycle_order
    f = s.failed.astype(float)
    # shift(1) comes before every cumulative or rolling operation
    prev = f.groupby(tid).shift(1)
    pos = s.groupby("test_id").cumcount().astype(float)
    out = pd.DataFrame(index=s.index)
    out["h_runs"] = pos
    out["h_fails"] = prev.fillna(0).groupby(tid).cumsum()
    out["h_rate"] = out.h_fails / pos.where(pos > 0)
    out["h_last1"] = prev
    out["h_fails5"] = rolling_sum(prev, tid, 5).fillna(0)
    out["h_fails20"] = rolling_sum(prev, tid, 20).fillna(0)
    last_fail = (pos - 1).where(prev == 1).groupby(tid).ffill()
    out["h_since_fail"] = (pos - 1) - last_fail
    # change between run j and j-1; shifted so row k sees changes among runs k-20..k-1 (19 pairs)
    change = f.groupby(tid).diff().abs().groupby(tid).shift(1)
    out["h_flips20"] = rolling_sum(change, tid, 19).fillna(0)
    out["h_dfe"] = pd.Series(dfe_before(d), index=d.index).reindex(s.index)
    prev_ld = np.log1p(s.duration).groupby(tid).shift(1)
    n_prev = prev_ld.notna().astype(float).groupby(tid).cumsum()
    out["h_dur"] = prev_ld.fillna(0).groupby(tid).cumsum() / n_prev.where(n_prev > 0)
    out["h_gap"] = co - co.groupby(tid).shift(1)
    out["h_age"] = co - co.groupby(tid).transform("first")
    out["j_ntests"] = s.groupby("cycle_order").cycle_order.transform("size").astype(float)
    return out.reindex(d.index)[FEATURES].astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in json.load(open("results/selected_systems.json"))["systems"]:
        d = load_table(name)
        t = pd.concat([d[KEYS], history_features(d)], axis=1)
        t.to_parquet(f"{OUT}/{name}.parquet", index=False)
        print(name, t.shape, t.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
