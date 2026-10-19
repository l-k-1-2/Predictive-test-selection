import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROWS = "data/processed/rows"
OUT = "data/processed/lastresults.csv"  # per-system LastResults findings, read again by system_stats.py


def load(path):
    t = pq.read_table(path)
    lr = t.column("last_results").combine_chunks()
    d = t.select(["cycle_order", "test_id", "failed"]).to_pandas()
    return d, lr.values.to_numpy(), lr.offsets.to_numpy()


def history_checks(d, vals, offs):
    # rows ordered by test then time, so H_prev of a row is the reversed earlier verdicts of its test
    order = np.lexsort((d.cycle_order.to_numpy(), d.test_id.to_numpy()))
    test = d.test_id.to_numpy()[order]
    failed = d.failed.to_numpy()[order].astype(np.int8)
    bounds = np.flatnonzero(np.diff(test)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(test)]])
    rec = []
    for s, e in zip(starts, ends):
        v = failed[s:e]
        for k in range(1, e - s):
            i = order[s + k]
            lst = vals[offs[i] : offs[i + 1]]
            h = {"prev": v[k - 1 :: -1], "cur": v[k::-1]}
            row = [len(lst), v[k], v[k - 1]]
            for end, lo in (("start", lst), ("end", lst[::-1])):
                row.append(lo[0] if len(lo) else -1)
                for a in ("prev", "cur"):
                    n = min(len(lo), len(h[a]))
                    row.append(n > 0 and np.array_equal(lo[:n], h[a][:n]))
            row += [len(lst) > len(h["prev"]), len(lst) > len(h["cur"])]
            rec.append(row)
    cols = ["m", "cur", "prev", "recent_start", "pre_start_prev", "pre_start_cur", "recent_end", "pre_end_prev", "pre_end_cur",
            "longer_prev", "longer_cur"]
    return pd.DataFrame(rec, columns=cols)


def summarize(name, r):
    out = {"system": name, "n_eligible": len(r), "n_empty_lists": int((r.m == 0).sum())}
    for end in ("start", "end"):
        out[f"lr_current_match_{end}"] = (r[f"recent_{end}"] == r.cur).mean()
        out[f"lr_prev_match_{end}"] = (r[f"recent_{end}"] == r.prev).mean()
        for a in ("prev", "cur"):
            out[f"prefix_{a}_{end}"] = r[f"pre_{end}_{a}"].mean()
    # the recent end is the one whose newest entry fits either history better
    fit = {e: max(out[f"lr_current_match_{e}"], out[f"lr_prev_match_{e}"]) for e in ("start", "end")}
    end = max(fit, key=fit.get)
    align = max(("prev", "cur"), key=lambda a: out[f"prefix_{a}_{end}"])
    out.update(lr_end=end, lr_alignment=align, lr_current_match=out[f"lr_current_match_{end}"], lr_prev_match=out[f"lr_prev_match_{end}"],
               lr_prefix_match=out[f"prefix_{align}_{end}"], lr_longer=r[f"longer_{align}"].mean())
    print(out)
    return out


def main():
    done = pd.read_csv(OUT).set_index("system").to_dict("index") if os.path.exists(OUT) else {}
    for path in sorted(glob.glob(f"{ROWS}/*.parquet")):
        name = os.path.basename(path)[:-len(".parquet")]
        if name in done:
            print(name, "cached")
            continue
        d, vals, offs = load(path)
        done[name] = {k: v for k, v in summarize(name, history_checks(d, vals, offs)).items() if k != "system"}
        pd.DataFrame.from_dict(done, orient="index").rename_axis("system").reset_index().to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
