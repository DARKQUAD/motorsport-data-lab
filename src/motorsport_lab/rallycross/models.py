"""What wins a rallycross race? A logistic regression and a gradient-boosting check.

Explanatory question (not a pre-race prediction: the pace is measured inside the race):
at equal pace, does *when* a driver takes the joker change the odds of winning?

* features: gap of the driver's best lap to the fastest lap of the race (s), joker timing
  (lap 1 / middle / last lap), race type (heat, semi-final, final), category;
* statsmodels logit: odds ratios with 95 % intervals, clustered by race;
* scikit-learn comparison with leave-one-event-out cross-validation (a whole meeting is
  held out at a time): pace-only logit, full logit, histogram gradient boosting, scored with
  log loss, Brier score and "did the favourite win" accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

RACE = ["event", "category", "race"]
TIMING = ["lap 1", "middle", "last lap"]


def win_table(races: pd.DataFrame) -> pd.DataFrame:
    """One row per driver and race (from :func:`analysis.per_race`) with model inputs."""
    d = races[races["joker_when"].isin(TIMING) & races["position"].notna()].copy()
    d["pace_gap_s"] = d["best_lap_s"] - d.groupby(RACE)["best_lap_s"].transform("min")
    d = d[d["pace_gap_s"] < 5]  # retirements and spins are not a pace signal
    d["win"] = (d["position"] == 1).astype(int)
    d["cars"] = d.groupby(RACE)["driver"].transform("count")
    d["race_id"] = d[RACE].astype(str).agg(" | ".join, axis=1)
    keep = d.groupby("race_id")["win"].transform("sum") == 1
    return d[keep].reset_index(drop=True)


def odds_ratios(table: pd.DataFrame) -> pd.DataFrame:
    """Logit of winning, odds ratios with race-clustered 95 % intervals."""
    import statsmodels.formula.api as smf

    fit = smf.logit(
        "win ~ pace_gap_s + C(joker_when, Treatment('middle')) + C(kind) + C(category)",
        table,
    ).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(table["race_id"])[0]},
    )
    ci = fit.conf_int()
    out = pd.DataFrame(
        {
            "term": fit.params.index,
            "odds_ratio": np.exp(fit.params.to_numpy()),
            "low": np.exp(ci[0].to_numpy()),
            "high": np.exp(ci[1].to_numpy()),
            "p_value": fit.pvalues.to_numpy(),
        }
    )
    return out[out["term"] != "Intercept"].reset_index(drop=True)


def _design(table: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.get_dummies(
        table[["pace_gap_s", "cars", "joker_when", "kind", "category"]],
        columns=["joker_when", "kind", "category"],
        dtype=float,
    )
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def cross_validate(table: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-event-out scores for three classifiers."""
    X_all = _design(table)
    pace_cols = ["pace_gap_s", "cars"]
    probs: dict[str, np.ndarray] = {
        m: np.zeros(len(table)) for m in ("logit_pace_only", "logit_full", "gradient_boosting")
    }
    for event in table["event"].unique():
        test = (table["event"] == event).to_numpy()
        X, y = X_all[~test], table["win"][~test]
        models = {
            "logit_pace_only": (LogisticRegression(max_iter=1000), pace_cols),
            "logit_full": (LogisticRegression(max_iter=1000, C=1.0), list(X_all.columns)),
            "gradient_boosting": (
                HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0),
                list(X_all.columns),
            ),
        }
        for name, (model, cols) in models.items():
            model.fit(X[cols], y)
            probs[name][test] = model.predict_proba(X_all.loc[test, cols])[:, 1]
    rows = []
    for name, p in probs.items():
        fav = table.assign(p=p).groupby("race_id")["p"].transform("max") == p
        rows.append(
            {
                "model": name,
                "log_loss": log_loss(table["win"], p),
                "brier": brier_score_loss(table["win"], p),
                "favourite_won": float(table.loc[fav, "win"].mean()),
                "races": int(table["race_id"].nunique()),
            }
        )
    base = float(table["win"].mean())
    rows.append(
        {
            "model": "base_rate (no model)",
            "log_loss": log_loss(table["win"], np.full(len(table), base)),
            "brier": brier_score_loss(table["win"], np.full(len(table), base)),
            "favourite_won": np.nan,
            "races": int(table["race_id"].nunique()),
        }
    )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def holdout(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Blind test: fit on ``train`` races, score on ``test`` races of unseen categories.

    The category dummies of unseen categories are all zero, so the models rely on pace gap,
    joker timing, race type and field size only.
    """
    columns = list(_design(train).columns)
    X, X_test = _design(train), _design(test, columns)
    pace_cols = ["pace_gap_s", "cars"]
    models = {
        "logit_pace_only": (LogisticRegression(max_iter=1000), pace_cols),
        "logit_full": (LogisticRegression(max_iter=1000), columns),
        "gradient_boosting": (
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0),
            columns,
        ),
    }
    y_test = test["win"]
    rows = []
    for name, (model, cols) in models.items():
        p = model.fit(X[cols], train["win"]).predict_proba(X_test[cols])[:, 1]
        fav = test.assign(p=p).groupby("race_id")["p"].transform("max") == p
        rows.append(
            {
                "model": name,
                "log_loss": log_loss(y_test, p),
                "brier": brier_score_loss(y_test, p),
                "favourite_won": float(test.loc[fav, "win"].mean()),
                "races": int(test["race_id"].nunique()),
            }
        )
    base = np.full(len(test), float(train["win"].mean()))
    rows.append(
        {
            "model": "base_rate (no model)",
            "log_loss": log_loss(y_test, base),
            "brier": brier_score_loss(y_test, base),
            "favourite_won": np.nan,
            "races": int(test["race_id"].nunique()),
        }
    )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
