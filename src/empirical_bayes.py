import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from evaluate import per_cycle_table

FEAT = "data/processed/features"
PREDS = "data/processed/preds"
BLOCK = 20
LO, HI = np.log(1e-3), np.log(1e6)
SCORES = ["raw", "laplace", "eb"]


def fit_beta_binomial(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    m = np.clip(k.sum() / n.sum(), 1e-6, 1 - 1e-6)
    lo, hi = np.log(1e-3), np.log(1e6)
    nll = lambda x: -(betaln(k + np.exp(x[0]), n - k + np.exp(x[1]))
                      - betaln(np.exp(x[0]), np.exp(x[1]))).sum()
    x0 = np.clip(np.log([10 * m, 10 * (1 - m)]), lo, hi)
    r = minimize(nll, x0, method="L-BFGS-B", bounds=[(lo, hi)] * 2)
    return np.exp(r.x), r


def train_blocks(feat):
    # all train-split rows, unit or not, in cycle_order
    t = feat[feat.split == "train"].sort_values(["test_id", "cycle_order"], kind="stable")
    block = t.groupby("test_id").cumcount() // BLOCK
    g = t.failed.astype(int).groupby([t.test_id, block])
    return g.sum().to_numpy(), g.size().to_numpy()


def fit_system(name):
    feat = pd.read_parquet(f"{FEAT}/{name}.parquet")
    k, n = train_blocks(feat)
    (a, b), r = fit_beta_binomial(k, n)
    at_bound = bool(np.any(np.isclose(r.x, LO, atol=1e-6) | np.isclose(r.x, HI, atol=1e-6)))
    row = dict(system=name, a=a, b=b, prior_mean=a / (a + b), a_plus_b=a + b, n_blocks=len(k), converged=bool(r.success),
               at_bound=at_bound, pooled_train_rate=k.sum() / n.sum())
    return row, feat


def add_scores(name, row, feat):
    d = pd.read_parquet(f"{PREDS}/{name}.parquet").drop(columns=[f"score_{s}" for s in SCORES], errors="ignore")
    f = feat[["cycle_order", "test", "h_runs", "h_fails20"]].rename(columns={"h_runs": "runs"})
    d = d.merge(f, on=["cycle_order", "test"], how="left", validate="one_to_one")
    k, n = d.h_fails20.to_numpy().astype(float), np.minimum(d.runs.to_numpy(), BLOCK).astype(float)
    a, b = row["a"], row["b"]
    d["score_raw"] = np.where(n > 0, k / np.maximum(n, 1), row["pooled_train_rate"])
    d["score_laplace"] = (k + 1) / (n + 2)
    d["score_eb"] = (k + a) / (n + a + b)
    d["k"] = k
    return d


def evaluate(name, d):
    t = d[d.split == "test"].sort_values(["cycle_order", "idx"], kind="stable").reset_index(drop=True)
    out = []
    for s in SCORES:
        pc = per_cycle_table(t, s)
        p = t[f"score_{s}"].clip(1e-6, 1 - 1e-6)
        zero = t.k == 0
        out.append(dict(system=name, score=s, apfd=pc.apfd.mean(), **{"recall@0.2": pc["recall@0.2"].mean()},
                        roc_auc=roc_auc_score(t.failed, p), brier=brier_score_loss(t.failed, p), log_loss=log_loss(t.failed, p),
                        k0_n_rows=int(zero.sum()), k0_n_failed=int(t.failed[zero].sum()),
                        k0_brier=brier_score_loss(t.failed[zero], p[zero]), k0_log_loss=log_loss(t.failed[zero], p[zero], labels=[False, True])))
    return out


def main():
    params, evals = [], []
    for name in json.load(open("results/selected_systems.json"))["systems"]:
        row, feat = fit_system(name)
        d = add_scores(name, row, feat)
        d.drop(columns=["runs", "h_fails20", "k"]).to_parquet(f"{PREDS}/{name}.parquet", index=False)
        params.append(row)
        evals += evaluate(name, d)
        print(row)
    pd.DataFrame(params).to_csv("results/eb_params.csv", index=False)
    e = pd.DataFrame(evals)
    mean = e.drop(columns="system").groupby("score", sort=False).mean().reset_index()
    mean.insert(0, "system", "mean")
    pd.concat([e, mean], ignore_index=True).to_csv("results/eb_eval.csv", index=False)


if __name__ == "__main__":
    main()
