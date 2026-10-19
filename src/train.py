import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

FEAT = "data/processed/features"
PREDS = "data/processed/preds"
MODELS = "models"
PARAMS = "results/model_params.json"
BASE = dict(objective="binary", learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=0,
            deterministic=True, num_threads=4, verbose=-1, metric="binary_logloss")
GRID = [dict(num_leaves=n, min_data_in_leaf=m) for n in (15, 63) for m in (20, 200)]
LR_C = (0.001, 0.01, 0.1, 1)


def fit_lr(train, val, C):
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    m.fit(train[FEATURES], train.failed)
    return m, average_precision_score(val.failed, m.predict_proba(val[FEATURES])[:, 1])


def choose_lr(train, val):
    fits = {C: fit_lr(train, val, C) for C in LR_C}
    best = max(fits, key=lambda C: fits[C][1])
    return fits[best][0], dict(C=best, val_ap={str(C): ap for C, (_, ap) in fits.items()})


def fit_lgbm(train, val, cfg):
    dtrain = lgb.Dataset(train[FEATURES], train.failed.astype(int))
    dval = lgb.Dataset(val[FEATURES], val.failed.astype(int), reference=dtrain)
    b = lgb.train({**BASE, **cfg}, dtrain, 3000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
    ap = average_precision_score(val.failed, b.predict(val[FEATURES], num_iteration=b.best_iteration))
    return b, ap


def choose_lgbm(train, val):
    fits = [fit_lgbm(train, val, cfg) for cfg in GRID]
    i = int(np.argmax([ap for _, ap in fits]))
    info = dict(**GRID[i], best_iteration=fits[i][0].best_iteration,
                grid=[dict(**cfg, best_iteration=b.best_iteration, val_ap=ap) for cfg, (b, ap) in zip(GRID, fits)])
    return fits[i][0], info


def fit_models(d):
    # only train and val unit rows are read here, so cal and test labels cannot reach fitting or model choice
    d = d[d.unit]
    train, val = d[d.split == "train"], d[d.split == "val"]
    lr, lr_info = choose_lr(train, val)
    booster, lgbm_info = choose_lgbm(train, val)
    return lr, booster, dict(lr=lr_info, lgbm=lgbm_info)


def scores(d, lr, booster):
    out = d[["cycle_order", "split", "test", "idx", "duration", "failed", "h_runs", "h_fails", "h_last1"]].copy()
    out["score_random"] = np.random.default_rng(0).random(len(d))
    out["score_id_order"] = 0.0
    out["score_dfe"] = d.h_dfe.to_numpy()
    out["score_lr"] = lr.predict_proba(d[FEATURES])[:, 1]
    out["score_lgbm"] = booster.predict(d[FEATURES], num_iteration=booster.best_iteration)
    return out


def train_system(name):
    d = pd.read_parquet(f"{FEAT}/{name}.parquet")
    lr, booster, info = fit_models(d)
    os.makedirs(f"{MODELS}/{name}", exist_ok=True)
    joblib.dump(lr, f"{MODELS}/{name}/lr.joblib")
    booster.save_model(f"{MODELS}/{name}/lgbm.txt")
    d = d[d.unit & (d.split != "train")].reset_index(drop=True)
    scores(d, lr, booster).to_parquet(f"{PREDS}/{name}.parquet", index=False)
    print(name, info["lr"]["C"], {k: info["lgbm"][k] for k in ("num_leaves", "min_data_in_leaf", "best_iteration")})
    return info


def main():
    os.makedirs(PREDS, exist_ok=True)
    params = json.load(open(PARAMS)) if os.path.exists(PARAMS) else {}
    for name in json.load(open("results/selected_systems.json"))["systems"]:
        if name in params and os.path.exists(f"{PREDS}/{name}.parquet") and os.path.exists(f"{MODELS}/{name}/lgbm.txt"):
            print(name, "cached")
            continue
        params[name] = train_system(name)
        json.dump(params, open(PARAMS, "w"), indent=2)


if __name__ == "__main__":
    main()
