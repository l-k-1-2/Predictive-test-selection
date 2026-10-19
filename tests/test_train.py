import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(os.path.join(HERE, ".."))
from train import fit_models, scores


def test_label_flip_does_not_change_training():
    d = pd.read_parquet("data/processed/features/system28.parquet")
    flipped = d.copy()
    later = flipped.split.isin(["cal", "test"])
    flipped.loc[later, "failed"] = ~flipped.loc[later, "failed"]
    assert flipped.failed.ne(d.failed).sum() == later.sum()

    lr1, b1, info1 = fit_models(d)
    lr2, b2, info2 = fit_models(flipped)
    assert info1 == info2

    rows = d[d.unit & (d.split != "train")].reset_index(drop=True)
    s1 = scores(rows, lr1, b1)
    s2 = scores(rows, lr2, b2)
    for col in ("score_lr", "score_lgbm"):
        assert np.array_equal(s1[col].to_numpy(), s2[col].to_numpy()), col
