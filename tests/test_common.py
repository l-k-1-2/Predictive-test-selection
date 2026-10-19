import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import apfd, dfe_update, rank

ROWS = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "rows")
CYCLES = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cycles")


def test_apfd_known_values():
    assert apfd([1, 0, 0, 0, 0]) == pytest.approx(0.9)
    assert apfd([0, 0, 0, 0, 1]) == pytest.approx(0.1)
    assert apfd([0, 1, 0, 1]) == pytest.approx(0.375)


def test_apfd_random_order_is_half():
    rng = np.random.default_rng(0)
    f = np.zeros(50, bool)
    f[:3] = True
    mean = np.mean([apfd(rng.permutation(f)) for _ in range(20000)])
    assert mean == pytest.approx(0.5, abs=0.005)


def test_rank_breaks_ties_by_tiebreak():
    assert list(rank([1, 1, 1, 2], [2, 0, 1, 9])) == [3, 1, 2, 0]
    assert list(rank([0, 0, 0], [5, 3, 4])) == [1, 2, 0]


def test_dfe_scores():
    P = np.zeros(1)
    seen = []
    for failed in ([0], [], []):
        seen.append(P[0])
        dfe_update(P, np.array(failed, dtype=int))
    assert seen == pytest.approx([0, 0.8, 0.16])


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROWS, "*.parquet"))))
def test_one_row_per_cycle_test_and_cycle_order(path):
    d = pd.read_parquet(path, columns=["cycle", "cycle_order", "test"])
    c = pd.read_parquet(os.path.join(CYCLES, os.path.basename(path)))
    assert not d.duplicated(["cycle", "test"]).any()
    assert c.cycle_order.tolist() == list(range(len(c)))
    assert sorted(d.cycle_order.unique()) == list(range(len(c)))
    assert (c.unit == (c.n_tests >= 20)).all()
