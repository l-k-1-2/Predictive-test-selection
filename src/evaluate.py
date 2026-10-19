import json
import sys

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from features import FEATURES
from common import apfd, rank

PREDS = "data/processed/preds"
MODELS = ["random", "id_order", "dfe", "lr", "lgbm"]
PROB_MODELS = ["lr", "lgbm"]
BUDGETS = (1, 2, 3)  # tenths, so ceil(b*n) is exact integer arithmetic
BLOCK, N_BOOT = 10, 2000
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load_test(name):
    d = pd.read_parquet(f"{PREDS}/{name}.parquet")
    return d[d.split == "test"].sort_values(["cycle_order", "idx"], kind="stable").reset_index(drop=True)


def cycles_of(d):
    bounds = np.flatnonzero(np.diff(d.cycle_order.to_numpy())) + 1
    for s, e in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(d)]])):
        yield d.iloc[s:e]


def cycle_metrics(c, score):
    order = rank(score, c.idx.to_numpy())
    failed = c.failed.to_numpy()[order]
    new = (c.failed.to_numpy() & ((c.h_runs.to_numpy() == 0) | (c.h_last1.to_numpy() == 0)))[order]
    n = len(c)
    out = {"apfd": apfd(failed)}
    for b in BUDGETS:
        k = -(-n * b // 10)
        out[f"recall@{b / 10}"] = failed[:k].sum() / failed.sum()
        out[f"detect@{b / 10}"] = float(failed[:k].any())
    k2 = -(-n * 2 // 10)
    out["recall_new@0.2"] = new[:k2].sum() / new.sum() if new.any() else np.nan
    return out


def per_cycle_table(d, model, score=None):
    score = d[f"score_{model}"].to_numpy() if score is None else score
    rows = []
    for c in cycles_of(d):
        if c.failed.any():
            rows.append(dict(cycle_order=c.cycle_order.iloc[0], **cycle_metrics(c, score[c.index.to_numpy()])))
    return pd.DataFrame(rows)


def system_metrics(name, d):
    out, per_cycle = [], {}
    for m in MODELS:
        pc = per_cycle[m] = per_cycle_table(d, m)
        row = dict(system=name, model=m, **pc.drop(columns="cycle_order").mean().to_dict())
        s, y = d[f"score_{m}"], d.failed
        row.update(roc_auc=roc_auc_score(y, s), average_precision=average_precision_score(y, s))
        if m in PROB_MODELS:
            row.update(brier=brier_score_loss(y, s), log_loss=log_loss(y, s))
        row.update(n_cycles=d.cycle_order.nunique(), n_failing_cycles=len(pc))
        out.append(row)
    return out, per_cycle


def block_resamples(n, rng):
    nb = -(-n // BLOCK)
    starts = rng.integers(0, n, (N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)) % n
    return idx.reshape(N_BOOT, nb * BLOCK)[:, :n]


def bootstrap(name, per_cycle):
    n = len(per_cycle["lgbm"])
    idx = block_resamples(n, np.random.default_rng(0))
    out = []
    for other in ("dfe", "lr"):
        for metric in ("apfd", "recall@0.2"):
            diff = per_cycle["lgbm"][metric].to_numpy() - per_cycle[other][metric].to_numpy()
            boot = diff[idx].mean(axis=1)
            out.append(dict(system=name, comparison=f"lgbm_minus_{other}", metric=metric, mean_diff=boot.mean(),
                            ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5), n_failing_cycles=n))
    return out


def random_check(d):
    means = []
    for seed in range(50):
        score = np.random.default_rng(seed).random(len(d))
        means.append(per_cycle_table(d, "random", score).apfd.mean())
    return float(np.mean(means))


def reliability_figure(name, d):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    top = 0
    for m, color, label in (("lr", BLUE, "lr"), ("lgbm", ORANGE, "lgbm")):
        bins = pd.qcut(d[f"score_{m}"], 10, duplicates="drop")
        g = d.groupby(bins, observed=True).agg(pred=(f"score_{m}", "mean"), obs=("failed", "mean"))
        ax.plot(g.pred, g.obs, marker="o", ms=6, lw=1.5, color=color, markeredgecolor=SURFACE, label=label)
        top = max(top, g.pred.max(), g.obs.max())
    top *= 1.05
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls="--", label="perfect")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("mean predicted failure rate (10 quantile bins)", fontsize=9, color=INK2)
    ax.set_ylabel("observed failure rate", fontsize=9, color=INK2)
    ax.set_title(f"{name}: reliability on test unit rows", loc="left", fontsize=11, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"figures/reliability_{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def importance(name):
    b = lgb.Booster(model_file=f"models/{name}/lgbm.txt")
    gain = pd.DataFrame({"feature": b.feature_name(), "gain": b.feature_importance("gain")})
    assert set(gain.feature) == set(FEATURES)
    gain = gain.sort_values("gain", ascending=False).reset_index(drop=True)
    gain.insert(0, "rank", gain.index + 1)
    gain.insert(0, "system", name)
    gain["gain_share"] = gain.gain / gain.gain.sum()
    return gain


def main():
    systems = json.load(open("results/selected_systems.json"))["systems"]
    metrics, boot, imp, rand = [], [], [], {}
    for name in systems:
        d = load_test(name)
        rows, per_cycle = system_metrics(name, d)
        metrics += rows
        boot += bootstrap(name, per_cycle)
        imp.append(importance(name))
        reliability_figure(name, d)
        rand[name] = random_check(d)
        print(name, "random-score mean APFD over 50 seeds", rand[name])
    m = pd.DataFrame(metrics)
    mean = m.drop(columns="system").groupby("model", sort=False).mean().reset_index()
    mean.insert(0, "system", "mean")
    pd.concat([m, mean], ignore_index=True).to_csv("results/model_metrics.csv", index=False)
    pd.DataFrame(boot).to_csv("results/model_bootstrap.csv", index=False)
    pd.concat(imp, ignore_index=True).to_csv("results/feature_importance.csv", index=False)
    bad = {n: v for n, v in rand.items() if not 0.48 <= v <= 0.52}
    if bad:
        print("STOP: random-score check outside 0.5 +- 0.02:", bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
