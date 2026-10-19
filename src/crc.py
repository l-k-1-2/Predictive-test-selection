import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import block_resamples
from common import rank

PREDS = "data/processed/preds"
SCORES = ["lgbm", "lr", "dfe", "eb"]
LOSSES = ["frac", "any"]
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.20]
LAMBDAS = np.arange(101) / 100
N_SPLITS = 1000
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLORS = {"lgbm": "#2a78d6", "lr": "#eb6834", "dfe": "#1baf7a", "lgbm fraction": "#eda100"}


def crc_threshold(fail_scores, alpha, kind="frac"):
    n = len(fail_scores)
    budget = alpha * (n + 1) - 1
    if budget < 0:
        return -np.inf
    if kind == "frac":
        s = np.concatenate(fail_scores)
        w = np.concatenate([np.full(f.size, 1 / f.size) for f in fail_scores])
    else:
        s, w = np.array([f.max() for f in fail_scores]), np.ones(n)
    o = np.argsort(s, kind="stable")
    k = np.searchsorted(np.cumsum(w[o]), budget + 1e-9, side="right")
    return np.inf if k == s.size else s[o][k]


def crc_grid(L, lambdas, alpha):
    ok = (L.sum(0) + 1) / (L.shape[0] + 1) <= alpha
    return lambdas[ok.argmax()] if ok.any() else lambdas[-1]


def unit_loss(f, tau, kind="frac"):
    miss = f < tau
    return miss.mean() if kind == "frac" else float(miss.all())


def load_cycles(name, split):
    d = pd.read_parquet(f"{PREDS}/{name}.parquet")
    d = d[d.split == split].sort_values(["cycle_order", "idx"], kind="stable").reset_index(drop=True)
    bounds = np.flatnonzero(np.diff(d.cycle_order.to_numpy())) + 1
    cycles = []
    for s, e in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(d)]])):
        c = d.iloc[s:e]
        cyc = dict(order=int(c.cycle_order.iloc[0]), idx=c.idx.to_numpy(), failed=c.failed.to_numpy(bool),
                   duration=np.nan_to_num(c.duration.to_numpy(float)))
        cyc.update({m: c[f"score_{m}"].to_numpy(float) for m in SCORES})
        cycles.append(cyc)
    return cycles


def failing(cycles):
    return [c for c in cycles if c["failed"].any()]


def fail_scores(cycles, model):
    return [c[model][c["failed"]] for c in failing(cycles)]


def summarize(share, dur_run, dur_tot, loss):
    return dict(risk=np.nanmean(loss), sel_frac=share.mean(), time_frac=dur_run.sum() / dur_tot.sum())


def threshold_arrays(cycles, model, tau, kind):
    n = len(cycles)
    share, dur_run, dur_tot, loss = np.empty(n), np.empty(n), np.empty(n), np.full(n, np.nan)
    for i, c in enumerate(cycles):
        run = c[model] >= tau
        share[i], dur_run[i], dur_tot[i] = run.mean(), c["duration"][run].sum(), c["duration"].sum()
        if c["failed"].any():
            loss[i] = unit_loss(c[model][c["failed"]], tau, kind)
    return share, dur_run, dur_tot, loss


def rank_info(c, model):
    # ranked positions of the failed tests and cumulative duration along the ranking
    key = ("rank", model)
    if key not in c:
        order = rank(c[model], c["idx"])
        inv = np.empty(len(order), int)
        inv[order] = np.arange(len(order))
        c[key] = (inv[c["failed"]], np.cumsum(c["duration"][order]))
    return c[key]


def fraction_arrays(cycles, model, j, kind):
    # runs the top ceil(j/100 * n) tests; integer arithmetic keeps the ceiling exact
    n = len(cycles)
    share, dur_run, dur_tot, loss = np.empty(n), np.empty(n), np.empty(n), np.full(n, np.nan)
    for i, c in enumerate(cycles):
        m = len(c["idx"])
        k = -(-j * m // 100)
        pos, cum = rank_info(c, model)
        share[i], dur_run[i], dur_tot[i] = k / m, (cum[k - 1] if k > 0 else 0.0), c["duration"].sum()
        if pos.size:
            loss[i] = (pos >= k).mean() if kind == "frac" else float((pos >= k).all())
    return share, dur_run, dur_tot, loss


def fraction_loss_matrix(cal_failing, model, kind):
    js = np.arange(101)
    L = np.empty((len(cal_failing), 101))
    for i, c in enumerate(cal_failing):
        k = -(-js * len(c["idx"]) // 100)
        miss = rank_info(c, model)[0][:, None] >= k[None, :]
        L[i] = miss.mean(0) if kind == "frac" else miss.all(0)
    return L


def static_rows(name, cal, test):
    rows = []
    cal_f = failing(cal)
    for model in SCORES:
        for kind in LOSSES:
            r = summarize(*fraction_arrays(test, model, 20, kind))
            rows.append(dict(system=name, score=model, method="top20", loss=kind, alpha=np.nan, n_cal=len(cal_f), param=0.2, **r))
            L = fraction_loss_matrix(cal_f, model, kind)
            for alpha in ALPHAS:
                tau = crc_threshold(fail_scores(cal, model), alpha, kind)
                r = summarize(*threshold_arrays(test, model, tau, kind))
                rows.append(dict(system=name, score=model, method="threshold", loss=kind, alpha=alpha, n_cal=len(cal_f), param=tau, **r))
                lam = crc_grid(L, LAMBDAS, alpha)
                r = summarize(*fraction_arrays(test, model, int(round(lam * 100)), kind))
                rows.append(dict(system=name, score=model, method="fraction", loss=kind, alpha=alpha, n_cal=len(cal_f), param=lam, **r))
    return rows


def cost_ci_rows(name, cal, test):
    idx = block_resamples(len(test), np.random.default_rng(0))
    out = []
    for alpha in (0.05, 0.10):
        arr = {}
        for model in ("lgbm", "dfe"):
            tau = crc_threshold(fail_scores(cal, model), alpha, "frac")
            arr[model] = threshold_arrays(test, model, tau, "frac")
        share = {m: a[0][idx].mean(axis=1) for m, a in arr.items()}
        risk = {}
        for m, a in arr.items():
            is_f = ~np.isnan(a[3])
            risk[m] = np.nan_to_num(a[3])[idx].sum(axis=1) / np.where(is_f[idx].sum(axis=1) > 0, is_f[idx].sum(axis=1), np.nan)
        for metric, diff in (("sel_frac", share["lgbm"] - share["dfe"]), ("risk", risk["lgbm"] - risk["dfe"])):
            out.append(dict(system=name, alpha=alpha, metric=metric, mean_diff=np.nanmean(diff),
                            ci_low=np.nanpercentile(diff, 2.5), ci_high=np.nanpercentile(diff, 97.5)))
    return out


def gate_rows(name, cycles):
    # lgbm only: random 50/50 splits of the pooled cal and test cycles
    n = len(cycles)
    sizes = np.array([len(c["idx"]) for c in cycles])
    s_all = np.concatenate([c["lgbm"] for c in cycles])
    cid = np.repeat(np.arange(n), sizes)
    f_idx = np.array([i for i, c in enumerate(cycles) if c["failed"].any()])
    f_lists = [cycles[i]["lgbm"][cycles[i]["failed"]] for i in f_idx]
    sf = np.concatenate(f_lists)
    fcid = np.repeat(np.arange(len(f_idx)), [f.size for f in f_lists])
    wf = np.concatenate([np.full(f.size, 1 / f.size) for f in f_lists])
    mx = np.array([f.max() for f in f_lists])
    risks, sels = {}, {}
    for seed in range(N_SPLITS):
        in_a = np.zeros(n, bool)
        in_a[np.random.default_rng(seed).permutation(n)[: n // 2]] = True
        a_f, b_f = np.flatnonzero(in_a[f_idx]), np.flatnonzero(~in_a[f_idx])
        for kind in LOSSES:
            for alpha in ALPHAS:
                tau = crc_threshold([f_lists[i] for i in a_f], alpha, kind)
                loss = np.bincount(fcid, weights=wf * (sf < tau), minlength=len(f_idx)) if kind == "frac" else (mx < tau).astype(float)
                share = np.bincount(cid, weights=(s_all >= tau), minlength=n) / sizes
                risks.setdefault((kind, alpha), []).append(loss[b_f].mean())
                sels.setdefault((kind, alpha), []).append(share[~in_a].mean())
    return [dict(system=name, loss=k, alpha=a, mean_risk=np.mean(r), se=np.std(r, ddof=1) / np.sqrt(N_SPLITS), mean_sel_frac=np.mean(sels[(k, a)]))
            for (k, a), r in risks.items()]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def cost_figure(name, static):
    s = static[(static.system == name) & (static.loss == "frac")]
    lines = [("lgbm", s[(s.score == "lgbm") & (s.method == "threshold")], "o"), ("lr", s[(s.score == "lr") & (s.method == "threshold")], "s"),
             ("dfe", s[(s.score == "dfe") & (s.method == "threshold")], "^"),
             ("lgbm fraction", s[(s.score == "lgbm") & (s.method == "fraction")], "D")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, col, label in zip(axes, ("risk", "sel_frac"), ("realized test risk (frac loss)", "share of tests run (sel_frac)")):
        style(ax)
        for lab, g, mk in lines:
            ax.plot(g.alpha, g[col], marker=mk, ms=6, lw=1.5, color=COLORS[lab], ls="--" if "fraction" in lab else "-",
                    markeredgecolor=SURFACE, label=lab)
        ax.set_xlabel("target alpha", fontsize=9, color=INK2)
        ax.set_ylabel(label, fontsize=9, color=INK2)
    axes[0].plot([0, 0.2], [0, 0.2], color=MUTED, lw=1, ls=":", label="target = realized")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.suptitle(f"{name}: conformal risk control on the test split", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"figures/risk_cost_{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def checks(static, gate):
    gate = gate.assign(passed=gate.mean_risk <= gate.alpha + 2 * gate.se)
    t = static[(static.score == "lgbm") & (static.method == "threshold")].sort_values("alpha")
    mono = all((g.sel_frac.diff().dropna() <= 1e-12).all() for _, g in t.groupby(["system", "loss"]))
    return gate, mono


def main():
    systems = json.load(open("results/selected_systems.json"))["systems"]
    static, ci, gate = [], [], []
    for name in systems:
        cal, test = load_cycles(name, "cal"), load_cycles(name, "test")
        static += static_rows(name, cal, test)
        ci += cost_ci_rows(name, cal, test)
        gate += gate_rows(name, cal + test)
        print(name, "done")
    static, gate = pd.DataFrame(static), pd.DataFrame(gate)
    static.to_csv("results/crc_static.csv", index=False)
    pd.DataFrame(ci).to_csv("results/crc_cost_ci.csv", index=False)
    gate.to_csv("results/crc_exchangeable.csv", index=False)
    for name in systems:
        cost_figure(name, static)
    gate, mono = checks(static, gate)
    print("gate failures:", gate[~gate.passed][["system", "loss", "alpha", "mean_risk", "se"]].to_dict("records"))
    print("static lgbm sel_frac non-increasing in alpha:", mono)
    if not gate.passed.all() or not mono:
        sys.exit(1)


if __name__ == "__main__":
    main()
