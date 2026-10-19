import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(os.path.join(HERE, ".."))
from features import FEATURES, history_features, load_table
from common import dfe_update

SYSTEMS = json.load(open("results/selected_systems.json"))["systems"]
H_COLS = [c for c in FEATURES if c.startswith("h_")]


def brute_features(d, test_id, c):
    own = d[d.test_id == test_id]
    prior = own[own.cycle_order < c].sort_values("cycle_order")
    f, co = prior.failed.to_numpy().astype(int), prior.cycle_order.to_numpy()
    n = len(f)
    fails = np.flatnonzero(f)
    return dict(
        h_runs=n,
        h_fails=f.sum(),
        h_rate=f.sum() / n if n else np.nan,
        h_last1=f[-1] if n else np.nan,
        h_fails5=f[-5:].sum(),
        h_fails20=f[-20:].sum(),
        h_since_fail=n - 1 - fails[-1] if len(fails) else np.nan,
        h_flips20=int((f[-20:][1:] != f[-20:][:-1]).sum()),
        h_dur=np.log1p(prior.duration.to_numpy()).mean() if n else np.nan,
        h_gap=c - co[-1] if n else np.nan,
        h_age=c - own[own.cycle_order <= c].cycle_order.min(),
        j_ntests=(d.cycle_order == c).sum(),
    )


def brute_dfe(d, rows):
    # P before cycle c, built from cycles < c only
    out = {}
    P = np.zeros(d.test_id.max() + 1)
    for c in range(rows.cycle_order.max() + 1):
        for i in rows.index[rows.cycle_order == c]:
            out[i] = P[rows.test_id[i]]
        cyc = d[d.cycle_order == c]
        dfe_update(P, cyc.test_id.to_numpy()[cyc.failed.to_numpy()])
    return out


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array_equal(np.isnan(a), np.isnan(b)) and (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) <= 1e-6).all()


@pytest.mark.parametrize("name", SYSTEMS)
def test_brute_force(name):
    t = pd.read_parquet(f"data/processed/features/{name}.parquet")
    d = load_table(name)
    rng = np.random.default_rng(0)
    test_rows = t.index[t.split == "test"].to_numpy()
    other = t.index[t.split != "test"].to_numpy()
    pick = np.concatenate([rng.choice(test_rows, 50, replace=False), rng.choice(other, 250, replace=False)])
    dfe = brute_dfe(d, t.loc[pick])
    for i in pick:
        want = brute_features(d, t.test_id[i], t.cycle_order[i])
        want["h_dfe"] = dfe[i]
        for col in FEATURES:
            assert same(t.loc[i, col], want[col]), (name, i, col, t.loc[i, col], want[col])


@pytest.mark.parametrize("name", SYSTEMS)
def test_perturbation(name):
    d = load_table(name)
    before = history_features(d)
    cal = np.sort(d.cycle_order[d.split == "cal"].unique())
    c = int(np.random.default_rng(0).choice(cal))
    d2 = d.copy()
    hit = d2.cycle_order == c
    d2.loc[hit, "failed"] = ~d2.loc[hit, "failed"]
    after = history_features(d2)
    early = (d.cycle_order <= c).to_numpy()
    for col in H_COLS:
        assert same(before.loc[early, col], after.loc[early, col]), (name, col)
    late = ~early
    assert any(not same(before.loc[late, col], after.loc[late, col]) for col in H_COLS)
