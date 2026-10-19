import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STATS = "results/system_stats.csv"
OUT = "results/selected_systems.json"
FIG = "figures/failing_rate.png"
CRITERIA = ("failing_unit_cycles >= 300 and n_tests >= 20 and spearman_cycle_time >= 0.9; all eligible systems are used; "
            "unit cycles have at least 20 tests")
WINDOW = 100
# reference palette, light mode: chart surface, ink tokens, categorical slots 1 and 2
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def select(stats):
    ok = (stats.failing_unit_cycles >= 300) & (stats.n_tests >= 20) & (stats.spearman_cycle_time >= 0.9)
    return stats[ok].system.tolist()


def plot(systems):
    cols = 4
    rows = -(-len(systems) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.1 * rows + 1), sharey=True, facecolor=SURFACE, squeeze=False)
    for ax, name in zip(axes.ravel(), systems):
        c = pd.read_parquet(f"data/processed/cycles/{name}.parquet").sort_values("cycle_order")
        c = c[c.unit]
        fail_share = (c.n_failed / c.n_tests).rolling(WINDOW).mean()
        failing = c.failing.astype(float).rolling(WINDOW).mean()
        ax.set_facecolor(SURFACE)
        ax.plot(c.cycle_order, failing, color=BLUE, lw=1.5, label="failing cycles (share)")
        ax.plot(c.cycle_order, fail_share, color=ORANGE, lw=1.5, label="failed tests per cycle (share)")
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    for ax in axes.ravel()[len(systems):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("rolling mean", fontsize=8, color=INK2)
    for ax in axes[-1]:
        ax.set_xlabel("cycle_order", fontsize=8, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Unit cycles (20+ tests): failing cycles and per-cycle failure rate, rolling mean over {WINDOW} unit cycles",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=150, facecolor=SURFACE)


def main():
    stats = pd.read_csv(STATS)
    systems = select(stats)
    if len(systems) < 3:
        print(stats.to_string())
        raise SystemExit("STOP: fewer than 3 eligible systems")
    with open(OUT, "w") as fh:
        json.dump({"systems": systems, "criteria": CRITERIA}, fh, indent=2)
    print(systems, "excluded:", sorted(set(stats.system) - set(systems)))
    plot(systems)


if __name__ == "__main__":
    main()
