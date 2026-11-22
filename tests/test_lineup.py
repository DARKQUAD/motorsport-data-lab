"""Driver-style profiles and the who-drives-next simulator."""

from pathlib import Path

import numpy as np
import pandas as pd

from motorsport_lab.endurance import driver_style as ds
from motorsport_lab.endurance import lineup
from motorsport_lab.endurance.laps import annotate

DATA = Path(__file__).resolve().parents[1] / "data"


def test_profiles_are_relative_to_the_car() -> None:
    df = annotate(pd.read_csv(DATA / "n24/n24_2026_gt3_laps.csv", dtype={"car": str}))
    prof = ds.profiles(df)
    assert set(ds.TRAITS) <= set(prof.columns)
    assert prof["clean_laps"].min() >= 8
    assert abs(prof["pace_s"].median()) < 2  # gaps to the car's own median centre on zero


def test_feasibility_keeps_every_driver_able_to_reach_the_minimum() -> None:
    p = lineup.LineupParams(drivers=3, stints=6, min_stints=2)
    # after 2 stints of driver 0, 4 stints left and drivers 1 and 2 still need 2 each
    ok = lineup._feasible(p, np.array([2, 0, 0]), last=0, run=2, stint=2)
    assert ok.tolist() == [False, True, True]


def test_policies_respect_the_rules_and_the_oracle_is_the_upper_bound() -> None:
    p = lineup.LineupParams()
    table = lineup.compare(p, n=200).set_index("policy")
    assert table.loc["rotation", "time_gained_s"] == 0
    assert table.loc["oracle", "time_gained_s"] >= table["time_gained_s"].drop("oracle").max()
    assert table.loc["learn_greedy", "time_gained_s"] > 0


def test_simulated_reliability_matches_the_measured_one() -> None:
    assert 0.3 < lineup.LineupParams().reliability() < 0.45  # measured 0.28-0.54 on 4 races
