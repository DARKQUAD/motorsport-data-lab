"""Forecast the next stint (8 laps, about 70 minutes) of every GT3 car.

Question a strategist asks: "what lap times will this car do until its next stop?"

Protocol (rolling origin, no look-ahead in the inputs):

* cut-offs every few hours of race time; for each car, the history is every lap that
  started before the cut-off, the target is the next ``HORIZON`` laps;
* known future inputs: which upcoming laps are in-laps / out-laps (the team plans its
  stops); past-only input: the field-wide track index (Code 60, rain, night);
* models: two race-engineer baselines, Chronos-2 zero-shot, AutoGluon-TimeSeries;
* metric: mean absolute error in seconds, on all laps and on clean laps only.

Known limitation: the ``clean`` flag and the track-index scale are computed once on the
whole race, so the baselines get a slightly better filter than they would live.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

HORIZON = 8
START = pd.Timestamp("2026-01-01")
KNOWN = ["in_pit", "out_lap"]
ALL_MODELS = (
    "last_clean_lap",
    "median_5_clean_laps",
    "auto_ets",
    "auto_theta",
    "auto_arima_x",
    "chronos2",
    "autogluon",
    "autogluon_best",
)


CAP = 2.0  # laps slower than 2x the car's reference (red flag, garage repairs) are capped


def to_series(df: pd.DataFrame) -> pd.DataFrame:
    """Long format expected by Chronos / AutoGluon: item_id, timestamp (1 lap = 1 minute).

    Lap times are capped at ``CAP`` x the car's reference lap: a 7-hour red-flag lap or a
    garage repair cannot be forecast from lap history and would dominate any average.
    Truth and inputs are capped the same way for every model.
    """
    out = pd.DataFrame(
        {
            "item_id": df["car"].astype(str),
            "timestamp": START + pd.to_timedelta(df["lap"], unit="min"),
            "target": np.minimum(df["lap_time_s"], CAP * df["car_ref_s"]).astype(float),
            "in_pit": df["in_pit"].astype(float),
            "out_lap": df["out_lap"].astype(float),
            "track_index": df["track_index"].astype(float),
            "clean": df["clean"].astype(bool),
            "race_hour": df["race_hour"].astype(float),
        }
    )
    return out.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def split(
    series: pd.DataFrame, cutoff_hour: float, horizon: int = HORIZON
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History = laps started before the cut-off; target = the next ``horizon`` laps."""
    lap_start = series["race_hour"] - series["target"] / 3600
    history = series[lap_start < cutoff_hour]
    future = series[lap_start >= cutoff_hour].groupby("item_id").head(horizon)
    counts = future.groupby("item_id")["target"].count()
    keep = counts[counts == horizon].index.intersection(
        history.groupby("item_id")["target"].count().loc[lambda s: s >= 3 * horizon].index
    )
    return history[history["item_id"].isin(keep)], future[future["item_id"].isin(keep)]


def _pit_offsets(history: pd.DataFrame) -> tuple[float, float]:
    clean_med = history[history["clean"]]["target"].median()
    in_lap = history[history["in_pit"] > 0]["target"].median() - clean_med
    out_lap = history[history["out_lap"] > 0]["target"].median() - clean_med
    return float(np.nan_to_num(in_lap, nan=20.0)), float(np.nan_to_num(out_lap, nan=200.0))


def baseline(history: pd.DataFrame, future: pd.DataFrame, window: int = 1) -> pd.Series:
    """Median of the last ``window`` clean laps (+ typical in/out-lap loss on planned stops)."""
    in_off, out_off = _pit_offsets(history)
    preds = []
    for item, fut in future.groupby("item_id", sort=False):
        hist = history[history["item_id"] == item]
        clean = hist[hist["clean"]]["target"].tail(window)
        level = clean.median() if not clean.empty else hist["target"].tail(5).median()
        preds.append(level + fut["in_pit"] * in_off + fut["out_lap"] * out_off)
    return pd.concat(preds)


def chronos2(
    history: pd.DataFrame, future: pd.DataFrame, model: str = "amazon/chronos-2"
) -> pd.Series:
    """Zero-shot Chronos-2 with known future pit laps and the past track index."""
    from chronos import Chronos2Pipeline

    pipeline = _cached(model, lambda: Chronos2Pipeline.from_pretrained(model, device_map="cpu"))
    context = history[["item_id", "timestamp", "target", *KNOWN, "track_index"]]
    future_cov = future[["item_id", "timestamp", *KNOWN]]
    pred = pipeline.predict_df(
        context,
        future_df=future_cov,
        prediction_length=HORIZON,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    )
    pred = pred.set_index(["item_id", "timestamp"])["predictions"]
    keys = pd.MultiIndex.from_frame(future[["item_id", "timestamp"]])
    return pd.Series(pred.reindex(keys).to_numpy(), index=future.index)


def autogluon(
    history: pd.DataFrame,
    future: pd.DataFrame,
    time_limit: int = 120,
    presets: str = "fast_training",
) -> pd.Series:
    """AutoGluon-TimeSeries, pit laps as known covariates (``best_quality`` = every model)."""
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    train = TimeSeriesDataFrame.from_data_frame(history[["item_id", "timestamp", "target", *KNOWN]])
    known = TimeSeriesDataFrame.from_data_frame(future[["item_id", "timestamp", *KNOWN]])
    predictor = TimeSeriesPredictor(
        prediction_length=HORIZON,
        target="target",
        known_covariates_names=KNOWN,
        eval_metric="MAE",
        freq="min",
        verbosity=0,
        path=None,
    ).fit(train, presets=presets, time_limit=time_limit, enable_ensemble=True)
    pred = predictor.predict(train, known_covariates=known)["mean"]
    keys = pd.MultiIndex.from_frame(future[["item_id", "timestamp"]])
    return pd.Series(pred.reindex(keys).to_numpy(), index=future.index)


def statistical(history: pd.DataFrame, future: pd.DataFrame, model: str) -> pd.Series:
    """Classical per-car forecasts with Nixtla statsforecast.

    * ``auto_ets`` / ``auto_theta``: fitted on each car's clean laps only (the green-flag
      pace level and its drift), then the typical in-lap / out-lap loss is added on the
      planned stops, like the race-engineer baseline;
    * ``auto_arima_x``: AutoARIMA on the full (capped) lap series with the pit laps as
      exogenous regressors.

    A car with no clean lap before the cut-off gets the "last clean lap" rule instead.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive

    keys = pd.MultiIndex.from_frame(future[["item_id", "timestamp"]])
    if model == "auto_arima_x":
        train = history.rename(columns={"item_id": "unique_id", "timestamp": "ds", "target": "y"})
        x_future = future.rename(columns={"item_id": "unique_id", "timestamp": "ds"})
        sf = StatsForecast(
            models=[AutoARIMA(max_p=3, max_q=2)], freq="min", n_jobs=1, fallback_model=Naive()
        )
        pred = sf.forecast(
            df=train[["unique_id", "ds", "y", *KNOWN]],
            h=HORIZON,
            X_df=x_future[["unique_id", "ds", *KNOWN]],
        )
        pred = pred.set_index(["unique_id", "ds"])["AutoARIMA"]
        return pd.Series(pred.reindex(keys).to_numpy(), index=future.index)
    clean = history[history["clean"]].copy()
    clean["ds"] = clean.groupby("item_id").cumcount()
    clean = clean.rename(columns={"item_id": "unique_id", "target": "y"})
    stat = {"auto_ets": AutoETS(model="ZZN"), "auto_theta": AutoTheta()}[model]
    sf = StatsForecast(models=[stat], freq=1, n_jobs=1, fallback_model=Naive())
    level = sf.forecast(df=clean[["unique_id", "ds", "y"]], h=HORIZON)
    level["step"] = level.groupby("unique_id").cumcount()
    column = str(stat)
    in_off, out_off = _pit_offsets(history)
    step = future.groupby("item_id").cumcount()
    base = level.set_index(["unique_id", "step"])[column]
    lookup = pd.MultiIndex.from_arrays([future["item_id"], step])
    values = base.reindex(lookup).to_numpy()
    pred = pd.Series(
        values + future["in_pit"].to_numpy() * in_off + future["out_lap"].to_numpy() * out_off,
        index=future.index,
    )
    # a car without clean laps in its history (long neutralisation): fall back on the rule
    return pred.fillna(baseline(history, future, 1))


_MODELS: dict[str, Any] = {}


def _cached(key: str, factory: Callable[[], Any]) -> Any:
    if key not in _MODELS:
        _MODELS[key] = factory()
    return _MODELS[key]


def backtest(
    series: pd.DataFrame,
    cutoffs: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0, 20.0),
    models: tuple[str, ...] = ("last_clean_lap", "median_5_clean_laps", "chronos2", "autogluon"),
    ag_time_limit: int = 120,
) -> pd.DataFrame:
    """Return one row per (cut-off, car, lap) with the truth and each model's forecast."""
    runners: dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.Series]] = {
        "last_clean_lap": lambda h, f: baseline(h, f, 1),
        "median_5_clean_laps": lambda h, f: baseline(h, f, 5),
        "chronos2": chronos2,
        "autogluon": lambda h, f: autogluon(h, f, ag_time_limit),
        "autogluon_best": lambda h, f: autogluon(h, f, ag_time_limit, "best_quality"),
        "auto_ets": lambda h, f: statistical(h, f, "auto_ets"),
        "auto_theta": lambda h, f: statistical(h, f, "auto_theta"),
        "auto_arima_x": lambda h, f: statistical(h, f, "auto_arima_x"),
    }
    frames = []
    for cutoff in cutoffs:
        history, future = split(series, cutoff)
        if future.empty:
            continue
        out = future[
            ["item_id", "timestamp", "target", "clean", "in_pit", "out_lap", "track_index"]
        ].copy()
        out["cutoff_hour"] = cutoff
        for name in models:
            out[name] = runners[name](history, future).to_numpy()
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def score(results: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """MAE (s) per model on all laps, clean laps and pit laps, plus skill vs the best baseline."""
    models = models or [c for c in results.columns if c in ALL_MODELS]
    pit = (results["in_pit"] > 0) | (results["out_lap"] > 0)
    rows = []
    for m in models:
        err = (results[m] - results["target"]).abs()
        rows.append(
            {
                "model": m,
                "mae_all_s": err.mean(),
                "mae_clean_s": err[results["clean"]].mean(),
                "mae_pit_laps_s": err[pit].mean(),
                "n_laps": int(err.notna().sum()),
            }
        )
    table = pd.DataFrame(rows)
    best_baseline = table[table["model"].isin(["last_clean_lap", "median_5_clean_laps"])][
        "mae_all_s"
    ].min()
    table["skill_vs_best_baseline"] = 1 - table["mae_all_s"] / best_baseline
    return table.sort_values("mae_all_s").reset_index(drop=True)


def paired_bootstrap(
    results: pd.DataFrame, model: str, reference: str, n: int = 2000, seed: int = 0
) -> dict[str, float]:
    """95 % interval of MAE(model) - MAE(reference), resampling whole (cut-off, car) blocks.

    Laps of the same car and cut-off are correlated, so blocks are resampled, not laps.
    A negative interval that excludes zero means ``model`` is reliably better.
    """
    err = results.assign(
        diff=(results[model] - results["target"]).abs()
        - (results[reference] - results["target"]).abs()
    )
    grouped = err.groupby(["cutoff_hour", "item_id"])["diff"]
    sums = grouped.sum().to_numpy(dtype=float)
    counts = grouped.count().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n, len(sums)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "mean_diff_s": float(sums.sum()) / float(counts.sum()),
        "ci_low_s": float(np.percentile(boot, 2.5)),
        "ci_high_s": float(np.percentile(boot, 97.5)),
        "blocks": float(len(sums)),
    }
