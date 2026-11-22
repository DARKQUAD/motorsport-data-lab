"""AutoGluon-Tabular lap-time model with an honest time split.

Target: the next racing lap of a GT3 car (in-laps and out-laps excluded). Features are
only things known when the lap starts: car model, driver slot, lap in stint, race hour,
the previous lap, the median of the last three laps, and the track index of the previous
10-minute window. Train on hours 0-16, test on hours 16-24 (the future), compare with the
"same as the previous lap" baseline and a linear regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = [
    "car_model",
    "driver_idx",
    "lap_in_stint",
    "race_hour",
    "prev_lap_s",
    "prev3_median_s",
    "prev_track_index",
]


def make_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per racing lap with features known before the lap starts."""
    d = df.sort_values(["car", "lap"]).copy()
    racing = ~(d["in_pit"] | d["out_lap"])
    d["racing_time"] = d["lap_time_s"].where(racing)
    g = d.groupby("car")["racing_time"]
    d["prev_lap_s"] = g.transform(lambda s: s.ffill().shift(1))
    d["prev3_median_s"] = g.transform(
        lambda s: s.ffill().shift(1).rolling(3, min_periods=1).median()
    )
    bucket_index = d.groupby("bucket")["track_index"].first()
    d["prev_track_index"] = (d["bucket"] - 1).map(bucket_index).fillna(1.0)
    table = d[racing & (d["lap"] > 3)].dropna(subset=["prev_lap_s"])
    return table[["car", *FEATURES, "lap_time_s", "clean"]].reset_index(drop=True)


def evaluate(
    df: pd.DataFrame, split_hour: float = 16.0, time_limit: int = 180
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (MAE table, AutoGluon feature importance)."""
    from autogluon.tabular import TabularPredictor

    table = make_table(df)
    train, test = table[table["race_hour"] < split_hour], table[table["race_hour"] >= split_hour]
    y = test["lap_time_s"]
    preds = {"previous_lap (baseline)": test["prev_lap_s"]}
    numeric = [
        "driver_idx",
        "lap_in_stint",
        "race_hour",
        "prev_lap_s",
        "prev3_median_s",
        "prev_track_index",
    ]
    lin = LinearRegression().fit(train[numeric], train["lap_time_s"])
    preds["linear_regression"] = pd.Series(lin.predict(test[numeric]), index=test.index)
    predictor = TabularPredictor(
        label="lap_time_s", eval_metric="mean_absolute_error", verbosity=0, path=None
    )
    predictor.fit(train[[*FEATURES, "lap_time_s"]], presets="medium_quality", time_limit=time_limit)
    preds["autogluon_tabular"] = predictor.predict(test[FEATURES])
    rows = []
    for name, p in preds.items():
        err = (p - y).abs()
        rows.append(
            {
                "model": name,
                "mae_all_s": err.mean(),
                "mae_clean_s": err[test["clean"]].mean(),
                "n_laps": len(err),
            }
        )
    importance = predictor.feature_importance(
        test[[*FEATURES, "lap_time_s"]], num_shuffle_sets=3, silent=True
    )
    scores = pd.DataFrame(rows).sort_values("mae_all_s").reset_index(drop=True)
    scores["skill_vs_previous_lap"] = 1 - scores["mae_all_s"] / float(
        scores.loc[scores["model"].str.startswith("previous"), "mae_all_s"].to_numpy()[0]
    )
    return scores, importance.reset_index(names="feature")[
        ["feature", "importance", "stddev"]
    ].round(3).assign(importance=lambda t: np.round(t["importance"], 3))
