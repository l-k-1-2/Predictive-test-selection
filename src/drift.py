import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crc import COLORS, GRID, INK, INK2, MUTED, SURFACE, crc_threshold, fail_scores, load_cycles, style, unit_loss

ALPHAS = (0.05, 0.10)
MOVING = 50
METHODS = {"static": dict(window=None, gamma=0.0, update=False),
           **{f"rolling-{w}": dict(window=w, gamma=0.0, update=True) for w in (25, 50, 100)},
           **{f"aci-{g}": dict(window=50, gamma=g, update=True) for g in (0.01, 0.05, 0.1)}}
LINES = {"static": "#2a78d6", "rolling-50": "#eb6834", "aci-0.05": "#1baf7a"}


def run_online(cycles, cal_fail, alpha, kind="frac", window=None, gamma=0.0, update=True):
    hist, a, out = list(cal_fail), alpha, []
    for c in cycles:
        tau = crc_threshold(hist[-window:] if window else hist, a, kind)
        run = c["scores"] >= tau
        failing = c["failed"].any()
        L = unit_loss(c["scores"][c["failed"]], tau, kind) if failing else np.nan
        out.append((run.mean(), (c["duration"] * run).sum(), c["duration"].sum(), L, a))
        if failing:
            a += gamma * (alpha - L)
            if update:
                hist.append(c["scores"][c["failed"]])
    return out


def moving_mean(x, w=MOVING):
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else np.array([])


def metrics(out, alpha, method):
    share, dur_run, dur_tot, loss, _ = (np.array(v) for v in zip(*out))
    lf = loss[~np.isnan(loss)]
    mw = moving_mean(lf)
    row = dict(alpha=alpha, method=method, n_failing_cycles=len(lf), risk=lf.mean(), sel_frac=share.mean(),
               time_frac=dur_run.sum() / dur_tot.sum(), max_window_risk=mw.max() if mw.size else np.nan, bound=np.nan, bound_ok=np.nan)
    if method.startswith("aci"):
        g = METHODS[method]["gamma"]
        row["bound"] = (max(alpha, 1 - alpha) + g) / (g * len(lf))
        row["bound_ok"] = bool(abs(row["risk"] - alpha) <= row["bound"])
    return row


def block_rows(name, test, cycles, runs):
    rows = []
    for alpha in ALPHAS:
        for b, ids in enumerate(np.array_split(np.arange(len(test)), 10)):
            y = np.concatenate([test[i]["failed"] for i in ids])
            s = np.concatenate([test[i]["lgbm"] for i in ids])
            nf = sum(test[i]["failed"].any() for i in ids)
            risk = {m: np.nanmean([runs[(alpha, m)][i][3] for i in ids]) if nf else np.nan for m in ("static", "rolling-50")}
            rows.append(dict(system=name, alpha=alpha, block=b, n_cycles=len(ids), n_failing=int(nf),
                             roc_auc=roc_auc_score(y, s) if 0 < y.sum() < y.size else np.nan, failing_cycle_rate=nf / len(ids),
                             static_risk=risk["static"], rolling50_risk=risk["rolling-50"]))
    return rows


def time_figure(name, test, runs, alpha=0.05):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), facecolor=SURFACE, sharex=True)
    order = np.array([c["order"] for c in test])
    fail = np.array([c["failed"].any() for c in test])
    for ax in axes:
        style(ax)
    for m, color in LINES.items():
        out = runs[(alpha, m)]
        loss = np.array([o[3] for o in out])
        share = np.array([o[0] for o in out])
        ml = moving_mean(loss[fail])
        axes[0].plot(order[fail][MOVING - 1:], ml, color=color, lw=1.5, label=m)
        axes[1].plot(order[MOVING - 1:], moving_mean(share), color=color, lw=1.5, label=m)
    axes[0].axhline(alpha, color=MUTED, lw=1, ls="--", label=f"alpha = {alpha}")
    axes[0].set_ylabel("moving mean loss (50 failing cycles)", fontsize=9, color=INK2)
    axes[1].set_ylabel("moving mean sel_frac (50 cycles)", fontsize=9, color=INK2)
    axes[1].set_xlabel("cycle_order", fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncol=4)
    fig.suptitle(f"{name}: risk and cost over the test split, alpha = {alpha}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"figures/risk_time_{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    rows, blocks = [], []
    for name in json.load(open("results/selected_systems.json"))["systems"]:
        cal, test = load_cycles(name, "cal"), load_cycles(name, "test")
        view = [dict(scores=c["lgbm"], failed=c["failed"], duration=c["duration"]) for c in test]
        cal_fail = fail_scores(cal, "lgbm")
        runs = {(a, m): run_online(view, cal_fail, a, "frac", **kw) for a in ALPHAS for m, kw in METHODS.items()}
        for (a, m), out in runs.items():
            rows.append(dict(system=name, **metrics(out, a, m)))
        blocks += block_rows(name, test, view, runs)
        time_figure(name, test, runs)
        print(name, "done")
    d = pd.DataFrame(rows)
    d.to_csv("results/drift.csv", index=False)
    pd.DataFrame(blocks).to_csv("results/drift_blocks.csv", index=False)
    aci = d[d.method.str.startswith("aci")]
    print("bound_ok for every aci run:", bool(aci.bound_ok.astype(bool).all()))
    if not aci.bound_ok.astype(bool).all():
        print(aci[~aci.bound_ok.astype(bool)])
        sys.exit(1)


if __name__ == "__main__":
    main()
