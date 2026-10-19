import glob
import os

import numpy as np
import pandas as pd

from common import apfd, dfe_update, rank

ROWS = "data/processed/rows"
CYCLES = "data/processed/cycles"
OUT = "results/baselines.csv"


def cycles_of(d):
    d = d.sort_values(["cycle_order", "idx"], kind="stable")
    co, ids, idx, failed = d.cycle_order.to_numpy(), d.test_id.to_numpy(), d.idx.to_numpy(), d.failed.to_numpy()
    bounds = np.flatnonzero(np.diff(co)) + 1
    for s, e in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(d)]])):
        yield co[s], ids[s:e], idx[s:e], failed[s:e]


def baselines(d, unit):
    P = np.zeros(d.test_id.max() + 1)
    id_order, dfe = [], []
    for co, ids, idx, failed in cycles_of(d):
        # scored on unit cycles only; DFE still learns from every cycle
        if unit[co] and failed.any():
            id_order.append(apfd(failed[rank(np.zeros(len(ids)), idx)]))
            dfe.append(apfd(failed[rank(P[ids], idx)]))
        dfe_update(P, ids[failed])
    return {"id_order": id_order, "dfe": dfe}


def main():
    out = []
    for path in sorted(glob.glob(f"{ROWS}/*.parquet")):
        name = os.path.basename(path)[:-len(".parquet")]
        d = pd.read_parquet(path, columns=["cycle_order", "test_id", "idx", "failed"])
        unit = pd.read_parquet(f"{CYCLES}/{name}.parquet").sort_values("cycle_order").unit.to_numpy()
        for method, scores in baselines(d, unit).items():
            out.append(dict(system=name, method=method, mean_apfd=np.mean(scores), n_failing_cycles=len(scores)))
        print(out[-2], out[-1])
    pd.DataFrame(out).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
