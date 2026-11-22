from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from motorsport_lab.endurance import laps as lp
from motorsport_lab.endurance import strategy as strat

DATA = Path(__file__).resolve().parents[1] / "data"


def toy_laps() -> pd.DataFrame:
    times = [500, 498, 497, 520, 700, 499, 498, 496, 515, 690, 497]
    in_pit = [False, False, False, True, False, False, False, False, True, False, False]
    drivers = [1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1]
    return pd.DataFrame(
        {
            "car": "9",
            "car_model": "Test GT3",
            "lap": range(1, 12),
            "lap_time_s": times,
            "in_pit": in_pit,
            "driver_idx": drivers,
            "driver": [f"D{d}" for d in drivers],
        }
    )


def test_annotate_stints_and_out_laps() -> None:
    df = lp.annotate(toy_laps())
    assert df["stint"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
    assert df.loc[df["out_lap"], "lap"].tolist() == [5, 10]
    assert df["elapsed_s"].iat[-1] == sum(toy_laps()["lap_time_s"])
    assert not df.loc[df["in_pit"] | df["out_lap"], "clean"].any()


def test_pit_stops_detect_driver_change() -> None:
    stops = lp.pit_stops(lp.annotate(toy_laps()))
    assert stops["lap"].tolist() == [4, 9]
    assert stops["driver_change"].tolist() == [True, True]
    assert (stops["loss_s"] > 150).all()


@pytest.fixture(scope="module")
def n24() -> pd.DataFrame:
    return lp.annotate(pd.read_csv(DATA / "n24/n24_2026_gt3_laps.csv", dtype={"car": str}))


def test_calibrated_simulator_reproduces_the_real_winner(n24: pd.DataFrame) -> None:
    params = strat.calibrate(n24)
    assert params.fuel_stint_laps == 8
    sim = strat.simulate(params, strat.Strategy("base"), n=500, seed=1)
    real_best = n24[n24["status"] == "classified"]["laps_completed"].max()
    assert abs(sim["laps"].median() - real_best) <= 3


def test_simulator_is_monotonic_in_stop_loss(n24: pd.DataFrame) -> None:
    params = strat.calibrate(n24)
    fast = strat.simulate(replace(params, stop_loss_s=100), strat.Strategy("x"), n=300, seed=2)[
        "laps"
    ].mean()
    slow = strat.simulate(replace(params, stop_loss_s=400), strat.Strategy("x"), n=300, seed=2)[
        "laps"
    ].mean()
    assert fast > slow


def test_compare_uses_common_random_numbers() -> None:
    p = strat.PRESETS["spa_24h"]
    a = strat.compare(p, [strat.Strategy("a"), strat.Strategy("b")], n=200, seed=3)
    assert a["mean_laps"].nunique() == 1  # identical strategies -> identical results


def test_small_le_mans_file_does_not_crash() -> None:
    """A short Al Kamel extract (few laps, no full stint) must still give empty, typed tables."""
    small = toy_laps().head(6)
    df = lp.annotate(small)
    assert lp.stint_evolution(df).empty
    assert list(lp.stint_evolution(df).columns)[:2] == ["car_model", "slope_s_per_lap"]
    assert not lp.car_model_pace(df).empty or df["clean"].sum() == 0
