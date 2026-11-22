"""Next-lap model zoo: the classic toolbox next to AutoGluon, on the same honest split.

Same question and same features as :mod:`motorsport_lab.ml.tabular` (next racing lap of a
GT3 car, train on race hours 0-16, test on hours 16-24), so every number is comparable:

* baselines: previous lap, median of the previous three laps;
* linear: ridge regression and a Huber regression (robust to the long tail of slow laps);
* trees: random forest, scikit-learn histogram gradient boosting, LightGBM, XGBoost,
  CatBoost (native categorical car model);
* LightGBM tuned with Optuna on a time-ordered validation window (hours 12-16 only, so
  the test hours are never seen while tuning);
* explanations: SHAP values from LightGBM's built-in TreeSHAP (``pred_contrib``);
* uncertainty: conformalised quantile regression (CQR), an 80 % interval whose coverage is
  checked on the unseen hours 16-24.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .tabular import FEATURES, make_table

NUMERIC = [f for f in FEATURES if f != "car_model"]
SEED = 0


def _frames(
    table: pd.DataFrame, split_hour: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = table.copy()
    data["car_model"] = data["car_model"].astype("category")
    train = data[data["race_hour"] < split_hour]
    test = data[data["race_hour"] >= split_hour]
    return train[FEATURES], test[FEATURES], train["lap_time_s"], test["lap_time_s"]


def _linear(model: Any) -> Any:
    encode = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["car_model"]),
        ]
    )
    return make_pipeline(encode, model)


def _ordinal(X: pd.DataFrame) -> pd.DataFrame:
    return X.assign(car_model=X["car_model"].cat.codes)


def _lgbm_params(**extra: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "l1",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": SEED,
        "verbose": -1,
    }
    params.update(extra)
    return params


def _as_text(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype({"car_model": str})


Spec = tuple[Any, Callable[[pd.DataFrame], pd.DataFrame], dict[str, Any]]


def _same(X: pd.DataFrame) -> pd.DataFrame:
    return X


def _lightgbm() -> Spec:
    import lightgbm as lgb

    return lgb.LGBMRegressor(**_lgbm_params()), _same, {}


def _xgboost() -> Spec:
    import xgboost as xgb

    booster = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        enable_categorical=True,
        tree_method="hist",
        random_state=SEED,
    )
    return booster, _same, {}


def _catboost() -> Spec:
    from catboost import CatBoostRegressor

    cat = CatBoostRegressor(
        loss_function="MAE",
        iterations=600,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    return cat, _as_text, {"cat_features": ["car_model"]}


_FACTORIES: dict[str, Callable[[], Spec]] = {
    "ridge": lambda: (_linear(RidgeCV(alphas=np.logspace(-2, 3, 12))), _same, {}),
    "huber": lambda: (
        _linear(HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=2000)),
        _same,
        {},
    ),
    "random_forest": lambda: (
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        _ordinal,
        {},
    ),
    "hist_gradient_boosting": lambda: (
        HistGradientBoostingRegressor(
            loss="absolute_error", categorical_features=["car_model"], random_state=SEED
        ),
        _same,
        {},
    ),
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "catboost": _catboost,
}


def fit_predict(name: str, X: pd.DataFrame, y: pd.Series, X_new: pd.DataFrame) -> np.ndarray:
    """Fit one model of the zoo on (X, y) and predict X_new."""
    if name not in _FACTORIES:
        raise ValueError(f"unknown model {name!r}")
    model, prepare, fit_args = _FACTORIES[name]()
    model.fit(prepare(X), y, **fit_args)
    return np.asarray(model.predict(prepare(X_new)))


ZOO = tuple(_FACTORIES)


def tune_lightgbm(
    X: pd.DataFrame, y: pd.Series, race_hour: pd.Series, n_trials: int = 40, val_from: float = 12.0
) -> dict[str, Any]:
    """Optuna search on a time-ordered split inside the training hours (fit < 12 h, val 12-16)."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fit_mask = (race_hour < val_from).to_numpy()

    def objective(trial: Any) -> float:
        params = _lgbm_params(
            n_estimators=trial.suggest_int("n_estimators", 100, 1200, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        model = lgb.LGBMRegressor(**params).fit(X[fit_mask], y[fit_mask])
        return float(np.mean(np.abs(model.predict(X[~fit_mask]) - y[~fit_mask])))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params)


def block_bootstrap(
    err: pd.Series, ref_err: pd.Series, blocks: pd.Series, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """MAE difference (model - reference) and its 95 % interval, resampling whole blocks."""
    diff = (err - ref_err).groupby(blocks)
    sums = np.asarray(diff.sum(), dtype=float)
    counts = np.asarray(diff.count(), dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return (
        float(sums.sum()) / float(counts.sum()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def evaluate(
    df: pd.DataFrame,
    split_hour: float = 16.0,
    models: tuple[str, ...] = ZOO,
    n_trials: int = 40,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """MAE of every model on hours 16-24, with a block-bootstrap interval vs the previous lap.

    Returns ``(scores, best_lightgbm_params)``. Blocks are (car, race hour) so correlated
    laps of the same car in the same hour are resampled together.
    """
    table = make_table(df)
    X, X_test, y, y_test = _frames(table, split_hour)
    test = table[table["race_hour"] >= split_hour]
    preds: dict[str, np.ndarray] = {
        "previous_lap (baseline)": test["prev_lap_s"].to_numpy(float),
        "median_3_laps (baseline)": test["prev3_median_s"].to_numpy(float),
    }
    for name in models:
        preds[name] = fit_predict(name, X, y, X_test)
    best: dict[str, Any] = {}
    if n_trials > 0:
        import lightgbm as lgb

        best = tune_lightgbm(X, y, table.loc[X.index, "race_hour"], n_trials=n_trials)
        tuned = lgb.LGBMRegressor(**_lgbm_params(subsample_freq=1, **best))
        preds["lightgbm_optuna"] = np.asarray(tuned.fit(X, y).predict(X_test))
    blocks = test["car"].astype(str) + "_" + test["race_hour"].astype(int).astype(str)
    ref_err = pd.Series(np.abs(preds["previous_lap (baseline)"] - y_test.to_numpy()), test.index)
    clean = test["clean"].to_numpy(bool)
    rows = []
    for name, p in preds.items():
        err = pd.Series(np.abs(p - y_test.to_numpy()), test.index)
        diff, low, high = block_bootstrap(err, ref_err, blocks)
        rows.append(
            {
                "model": name,
                "mae_all_s": err.mean(),
                "mae_clean_s": err[clean].mean(),
                "diff_vs_previous_lap_s": diff,
                "ci_low_s": low,
                "ci_high_s": high,
                "n_laps": len(err),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_all_s").reset_index(drop=True), best


def shap_importance(df: pd.DataFrame, split_hour: float = 16.0) -> pd.DataFrame:
    """Mean |SHAP| (seconds) per feature for LightGBM on the test hours (TreeSHAP)."""
    import lightgbm as lgb

    X, X_test, y, _ = _frames(make_table(df), split_hour)
    model = lgb.LGBMRegressor(**_lgbm_params()).fit(X, y)
    contrib = np.asarray(model.predict(X_test, pred_contrib=True))[:, :-1]  # last = bias
    out = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_abs_shap_s": np.abs(contrib).mean(axis=0),
            "mean_shap_s": contrib.mean(axis=0),
        }
    )
    return out.sort_values("mean_abs_shap_s", ascending=False).reset_index(drop=True)


def conformal_intervals(
    df: pd.DataFrame,
    fit_until: float = 12.0,
    calibrate_until: float = 16.0,
    alpha: float = 0.2,
) -> pd.DataFrame:
    """Conformalised quantile regression (Romano et al., 2019) with LightGBM quantiles.

    Quantile models for ``alpha/2`` and ``1-alpha/2`` are fitted on hours 0-12, the
    conformity scores on hours 12-16 widen (or shrink) the band, and the coverage is
    measured on hours 16-24. Raw quantiles are reported next to it for comparison.
    """
    import lightgbm as lgb

    table = make_table(df)
    data = table.assign(car_model=table["car_model"].astype("category"))
    fit = data[data["race_hour"] < fit_until]
    cal = data[(data["race_hour"] >= fit_until) & (data["race_hour"] < calibrate_until)]
    test = data[data["race_hour"] >= calibrate_until]

    def quantile(q: float) -> Callable[[pd.DataFrame], np.ndarray]:
        params = _lgbm_params(objective="quantile", alpha=q)
        model = lgb.LGBMRegressor(**params).fit(fit[FEATURES], fit["lap_time_s"])
        return lambda frame: np.asarray(model.predict(frame[FEATURES]))

    low, high = quantile(alpha / 2), quantile(1 - alpha / 2)
    y_cal = cal["lap_time_s"].to_numpy()
    scores = np.maximum(low(cal) - y_cal, y_cal - high(cal))
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    q_hat = float(np.quantile(scores, level, method="higher"))
    y = test["lap_time_s"].to_numpy()
    clean = test["clean"].to_numpy(bool)
    rows = []
    for name, lo, hi in (
        ("raw_quantiles", low(test), high(test)),
        ("conformal (CQR)", low(test) - q_hat, high(test) + q_hat),
    ):
        inside = (y >= lo) & (y <= hi)
        rows.append(
            {
                "method": name,
                "target_coverage": 1 - alpha,
                "coverage_all": inside.mean(),
                "coverage_clean": inside[clean].mean(),
                "median_width_s": float(np.median(hi - lo)),
                "median_width_clean_s": float(np.median((hi - lo)[clean])),
            }
        )
    return pd.DataFrame(rows)
