"""Blind tests across seasons: does what was learnt on past 24h races hold on the next one?

Six editions of the Nürburgring 24h have a public lap-by-lap file (2017, 2018, 2020, 2022,
2023, 2026). Every test here is *forward-chaining*: a model only sees seasons strictly
before the season it is scored on, exactly as a team preparing next year's race would.

* ``next_lap``: the next-lap model zoo trained on all earlier seasons, tested on the
  whole next race (block bootstrap vs "same as the previous lap");
* ``conformal``: 80 % intervals fitted on earlier seasons, calibrated on the latest
  earlier season, coverage measured on the next one;
* ``forecast``: the rolling next-stint backtest re-run on every edition (Chronos-2 is
  zero-shot, so every season is blind for it);
* ``rating_stability``: do mixed-model driver ratings of one season predict the next?
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..endurance.laps import annotate
from .classic import ZOO, _lgbm_params, block_bootstrap, fit_predict
from .tabular import FEATURES, make_table

YEARS = (2017, 2018, 2020, 2022, 2023, 2026)
BASELINES = ("previous_lap (baseline)", "median_3_laps (baseline)")


def load_all(data_dir: Path, years: Sequence[int] = YEARS) -> pd.DataFrame:
    """Annotated GT3 laps of every edition, with a ``season`` column."""
    frames = []
    for year in years:
        laps = pd.read_csv(data_dir / f"n24_{year}_gt3_laps.csv", dtype={"car": str})
        frames.append(annotate(laps).assign(season=year))
    return pd.concat(frames, ignore_index=True)


RED_FLAG = 2.0  # laps slower than 2x the car's reference: red flag or repairs, not forecastable


def season_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Next-lap feature table for every season, with one shared car-model category list.

    Laps slower than ``RED_FLAG`` x the car's reference lap (a 2020 red-flag "lap" lasted
    10 hours) are dropped from training and scoring, as in the forecasting backtest.
    """
    tables = []
    for season, g in laps.groupby("season"):
        table = make_table(g).assign(season=season)
        ref = table["car"].map(g.groupby("car")["car_ref_s"].first())
        tables.append(table[table["lap_time_s"] <= RED_FLAG * ref])
    table = pd.concat(tables, ignore_index=True)
    table["car_model"] = table["car_model"].fillna("unknown").astype("category")
    table["block"] = (
        table["season"].astype(str)
        + "_"
        + table["car"].astype(str)
        + "_"
        + table["race_hour"].astype(int).astype(str)
    )
    return table


def _autogluon_tabular(train: pd.DataFrame, test: pd.DataFrame, time_limit: int) -> np.ndarray:
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label="lap_time_s", eval_metric="mean_absolute_error", verbosity=0, path=None
    )
    predictor.fit(train[[*FEATURES, "lap_time_s"]], presets="medium_quality", time_limit=time_limit)
    return np.asarray(predictor.predict(test[FEATURES]))


def next_lap(
    table: pd.DataFrame,
    models: Sequence[str] = ZOO,
    autogluon_time_limit: int = 0,
) -> pd.DataFrame:
    """Per-lap absolute errors: train on every season before ``s``, test on season ``s``."""
    seasons = sorted(table["season"].unique())
    frames = []
    for season in seasons[1:]:
        train, test = table[table["season"] < season], table[table["season"] == season]
        preds: dict[str, np.ndarray] = {
            BASELINES[0]: test["prev_lap_s"].to_numpy(float),
            BASELINES[1]: test["prev3_median_s"].to_numpy(float),
        }
        for name in models:
            preds[name] = fit_predict(name, train[FEATURES], train["lap_time_s"], test[FEATURES])
        if autogluon_time_limit > 0:
            preds["autogluon_tabular"] = _autogluon_tabular(train, test, autogluon_time_limit)
        y = test["lap_time_s"].to_numpy(float)
        for name, p in preds.items():
            frames.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "model": name,
                        "abs_err": np.abs(p - y),
                        "clean": test["clean"].to_numpy(bool),
                        "block": test["block"].to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def summarise(errors: pd.DataFrame, reference: str = BASELINES[0]) -> pd.DataFrame:
    """MAE per model and season, the pooled MAE, seasons won, and a bootstrap vs ``reference``.

    The bootstrap resamples whole (season, car, race hour) blocks of the pooled test laps.
    """
    per = errors.pivot_table(index="model", columns="season", values="abs_err", aggfunc="mean")
    per.columns = [f"mae_{c}" for c in per.columns]
    pooled = errors.groupby("model")["abs_err"].mean().rename("mae_all_seasons")
    clean = errors[errors["clean"]].groupby("model")["abs_err"].mean().rename("mae_clean")
    ranks = errors.groupby(["season", "model"])["abs_err"].mean().groupby(level=0).rank()
    wins = (ranks == 1).groupby(level=1).sum().rename("seasons_won")
    beats = (
        errors.groupby(["season", "model"])["abs_err"]
        .mean()
        .unstack()
        .lt(errors[errors["model"] == reference].groupby("season")["abs_err"].mean(), axis=0)
        .sum()
        .rename("seasons_better_than_reference")
    )
    ref = errors[errors["model"] == reference].set_index(["season", "block"], append=True)
    rows = []
    for model, g in errors.groupby("model"):
        cur = g.set_index(["season", "block"], append=True)
        diff, low, high = block_bootstrap(
            pd.Series(cur["abs_err"].to_numpy()),
            pd.Series(ref["abs_err"].to_numpy()),
            pd.Series(cur.index.get_level_values("block").to_numpy()),
        )
        rows.append(
            {"model": model, "diff_vs_reference_s": diff, "ci_low_s": low, "ci_high_s": high}
        )
    out = pd.concat([pooled, clean, wins, beats, per], axis=1).join(
        pd.DataFrame(rows).set_index("model")
    )
    return out.sort_values("mae_all_seasons").reset_index(names="model")


def conformal(table: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """CQR across seasons: fit on seasons < s-1, calibrate on season s-1, test on season s."""
    import lightgbm as lgb

    seasons = sorted(table["season"].unique())
    rows: list[dict[str, Any]] = []
    for i in range(2, len(seasons)):
        fit = table[table["season"] < seasons[i - 1]]
        cal = table[table["season"] == seasons[i - 1]]
        test = table[table["season"] == seasons[i]]
        bands: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for q in (alpha / 2, 1 - alpha / 2):
            model = lgb.LGBMRegressor(**_lgbm_params(objective="quantile", alpha=q))
            model.fit(fit[FEATURES], fit["lap_time_s"])
            bands[q] = (
                np.asarray(model.predict(cal[FEATURES]), dtype=float),
                np.asarray(model.predict(test[FEATURES]), dtype=float),
            )
        y_cal = cal["lap_time_s"].to_numpy()
        lo_cal, lo_test = bands[alpha / 2]
        hi_cal, hi_test = bands[1 - alpha / 2]
        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
        q_hat = float(np.quantile(scores, level, method="higher"))
        y = test["lap_time_s"].to_numpy()
        clean = test["clean"].to_numpy(bool)
        for name, lo, hi in (
            ("raw_quantiles", lo_test, hi_test),
            ("conformal (CQR)", lo_test - q_hat, hi_test + q_hat),
        ):
            inside = (y >= lo) & (y <= hi)
            rows.append(
                {
                    "season": int(seasons[i]),
                    "method": name,
                    "coverage_all": float(inside.mean()),
                    "coverage_clean": float(inside[clean].mean()),
                    "median_width_s": float(np.median(hi - lo)),
                }
            )
    return pd.DataFrame(rows)


def forecast(laps: pd.DataFrame, models: Sequence[str], ag_time_limit: int = 60) -> pd.DataFrame:
    """The rolling next-stint backtest of :mod:`forecasting`, run on every season."""
    from . import forecasting as fc

    frames = []
    for season in sorted(laps["season"].unique()):
        g = laps[laps["season"] == season]
        res = fc.backtest(fc.to_series(g), models=tuple(models), ag_time_limit=ag_time_limit)
        frames.append(res.assign(season=int(season)))
    return pd.concat(frames, ignore_index=True)


def forecast_summary(results: pd.DataFrame, reference: str = "last_clean_lap") -> pd.DataFrame:
    """MAE per model and season, pooled MAE and a pooled block bootstrap vs ``reference``."""
    from . import forecasting as fc

    models = [m for m in fc.ALL_MODELS if m in results.columns]
    abs_err = results[models].sub(results["target"], axis=0).abs()
    per = abs_err.groupby(results["season"]).mean()
    keyed = results.assign(item_id=results["season"].astype(str) + "_" + results["item_id"])
    rows = []
    for m in models:
        err = (results[m] - results["target"]).abs()
        sig = fc.paired_bootstrap(keyed, m, reference) if m != reference else {}
        rows.append(
            {
                "model": m,
                "mae_all_seasons": err.mean(),
                "mae_clean": err[results["clean"]].mean(),
                "seasons_better_than_last_clean_lap": int((per[m] < per[reference]).sum()),
                "diff_vs_last_clean_lap_s": sig.get("mean_diff_s", 0.0),
                "ci_low_s": sig.get("ci_low_s", 0.0),
                "ci_high_s": sig.get("ci_high_s", 0.0),
                **{f"mae_{s}": per.loc[s, m] for s in per.index},
            }
        )
    return pd.DataFrame(rows).sort_values("mae_all_seasons").reset_index(drop=True)


def rating_stability(laps: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of driver ratings between consecutive seasons with driver names."""
    from ..endurance.laps import driver_deltas
    from ..endurance.ratings import driver_ratings

    named = [
        int(s)
        for s in sorted(laps["season"].unique())
        if laps.loc[laps["season"] == s, "driver"].notna().mean() > 0.5
    ]
    ratings, raw = {}, {}
    for season in named:
        g = laps[laps["season"] == season]
        ratings[season] = driver_ratings(g)[0].set_index("driver")["effect_s"]
        raw[season] = driver_deltas(g).groupby("driver")["delta_s"].mean()
    rows = []
    for a, b in pairwise(named):
        for name, table in (("mixed_model", ratings), ("raw_teammate_delta", raw)):
            both = pd.concat([table[a], table[b]], axis=1, join="inner").dropna()
            rows.append(
                {
                    "from_season": a,
                    "to_season": b,
                    "method": name,
                    "drivers_in_both": len(both),
                    "spearman": float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")),
                }
            )
    return pd.DataFrame(rows)
