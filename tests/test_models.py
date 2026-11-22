"""Classic ML and statistics on the real datasets (needs the ``[models]`` extra)."""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motorsport_lab.endurance.laps import annotate
from motorsport_lab.rallycross.analysis import per_race

DATA = Path(__file__).resolve().parents[1] / "data"


@cache
def n24() -> pd.DataFrame:
    return annotate(pd.read_csv(DATA / "n24/n24_2026_gt3_laps.csv", dtype={"car": str}))


def test_block_bootstrap_detects_a_real_gap() -> None:
    from motorsport_lab.ml.classic import block_bootstrap

    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(10, 1, 400)).abs()
    blocks = pd.Series(np.repeat(np.arange(40), 10))
    diff, low, high = block_bootstrap(ref - 2, ref, blocks)
    assert diff == pytest.approx(-2)
    assert low <= -2 <= high
    _, low0, high0 = block_bootstrap(ref, ref, blocks)
    assert low0 == high0 == 0


def test_model_zoo_beats_previous_lap() -> None:
    pytest.importorskip("lightgbm")
    from motorsport_lab.ml import classic

    scores, best = classic.evaluate(n24(), models=("ridge", "lightgbm"), n_trials=0)
    mae = scores.set_index("model")["mae_all_s"]
    assert best == {}
    assert mae["lightgbm"] < mae["previous_lap (baseline)"]
    assert scores["ci_low_s"].le(scores["diff_vs_previous_lap_s"]).all()


def test_shap_values_add_up_and_track_index_matters() -> None:
    pytest.importorskip("lightgbm")
    from motorsport_lab.ml import classic

    shap = classic.shap_importance(n24())
    assert set(shap["feature"]) == set(classic.FEATURES)
    assert "prev_track_index" in shap["feature"].head(3).to_list()


def test_conformal_widens_raw_quantiles_towards_target() -> None:
    pytest.importorskip("lightgbm")
    from motorsport_lab.ml import classic

    table = classic.conformal_intervals(n24()).set_index("method")
    raw, cqr = table.loc["raw_quantiles"], table.loc["conformal (CQR)"]
    assert cqr["coverage_all"] > raw["coverage_all"]
    assert abs(cqr["coverage_all"] - 0.8) < 0.1


def test_statistical_forecasts_cover_every_future_lap() -> None:
    pytest.importorskip("statsforecast")
    from motorsport_lab.ml import forecasting as fc

    history, future = fc.split(fc.to_series(n24()), 12.0)
    for model in ("auto_ets", "auto_arima_x"):
        pred = fc.statistical(history, future, model)
        assert pred.notna().all()
        assert pred.between(300, 2000).all()  # seconds per Nürburgring lap


def test_driver_ratings_shrink_and_rank() -> None:
    pytest.importorskip("statsmodels")
    from motorsport_lab.endurance.ratings import driver_ratings

    ratings, summary = driver_ratings(n24())
    assert 0 < summary["icc"] < 1
    assert ratings["effect_s"].is_monotonic_increasing
    assert (ratings["low_s"] <= ratings["effect_s"]).all()
    # partial pooling: few laps -> wider interval
    few, many = ratings.nsmallest(10, "clean_laps"), ratings.nlargest(10, "clean_laps")
    assert few["sd_s"].mean() > many["sd_s"].mean()


def test_rallycross_win_model() -> None:
    pytest.importorskip("statsmodels")
    from motorsport_lab.rallycross import models as rx

    table = rx.win_table(per_race(pd.read_csv(DATA / "rx_france/rx_france_2026_laps.csv")))
    assert table.groupby("race_id")["win"].sum().eq(1).all()
    odds = rx.odds_ratios(table).set_index("term")
    assert odds.loc["pace_gap_s", "odds_ratio"] < 1  # slower = fewer wins
    cv = rx.cross_validate(table).set_index("model")
    assert cv.loc["logit_full", "log_loss"] < cv.loc["base_rate (no model)", "log_loss"]
