from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from motorsport_lab.i18n import TEXT
from motorsport_lab.rallycross import analysis as rxa
from motorsport_lab.rallycross import simulator as rxs


def toy_heat() -> pd.DataFrame:
    rows = []
    for driver, times, joker_lap in (
        ("A", [36.0, 34.0, 34.2, 37.5], 4),
        ("B", [39.0, 34.5, 34.4, 34.6], 1),
    ):
        for lap, t in enumerate(times, start=1):
            rows.append(
                {
                    "event": "Test",
                    "date": "2026-01-01",
                    "category": "Supercar",
                    "kind": "heat",
                    "race": "MQ1",
                    "driver": driver,
                    "position": 1 if driver == "A" else 2,
                    "lap": lap,
                    "lap_time_s": t,
                    "is_joker": lap == joker_lap,
                    "heat_time_s": sum(times) + 4,
                }
            )
    return pd.DataFrame(rows)


def test_per_race_costs() -> None:
    races = rxa.per_race(toy_heat()).set_index("driver")
    assert races.loc["A", "joker_lap"] == 4
    normal = np.median([34.0, 34.2])  # laps 2..N without the joker
    assert races.loc["A", "joker_cost_s"] == pytest.approx(37.5 - normal)
    assert races.loc["A", "joker_when"] == "last lap"
    assert races.loc["B", "joker_when"] == "lap 1"
    assert races.loc["A", "lap1_cost_s"] == pytest.approx(36.0 - normal)


def test_simulator_positions_are_valid() -> None:
    pos = rxs.simulate(rxs.RXParams(n_cars=6, laps=6), grid_slot=3, joker_lap=2, n=500, seed=1)
    assert pos.min() >= 1 and pos.max() <= 6


def test_pole_beats_back_of_grid() -> None:
    p = rxs.RXParams()
    front = rxs.simulate(p, 1, 6, n=2000, seed=2).mean()
    back = rxs.simulate(p, 6, 6, n=2000, seed=2).mean()
    assert front < back


def test_free_joker_makes_timing_irrelevant_without_traffic() -> None:
    p = replace(rxs.RXParams(), joker_cost_s=0.0, pass_advantage_s=-5.0, lap1_incident_rate=0.0)
    early = rxs.simulate(p, 3, 1, n=3000, seed=4).mean()
    late = rxs.simulate(p, 3, 6, n=3000, seed=4).mean()
    assert abs(early - late) < 0.15


def test_every_text_has_both_languages() -> None:
    assert all({"en", "fr"} <= set(v) for v in TEXT.values())
