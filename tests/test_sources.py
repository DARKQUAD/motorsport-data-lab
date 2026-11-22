import math
from pathlib import Path

import pandas as pd
import pytest

from motorsport_lab.sources import parse_laptime
from motorsport_lab.sources.alkamel import gt3_only, read_analysis_csv
from motorsport_lab.sources.its_rallycross import _session_kind
from motorsport_lab.sources.n24_wige import _split_entrant_model, is_gt3

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("8:32.481", 512.481), ("34.112", 34.112), ("1:02:03.500", 3723.5), ("00:02:25.129", 145.129)],
)
def test_parse_laptime(text: str, seconds: float) -> None:
    assert parse_laptime(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "-", "pit", None])
def test_parse_laptime_missing(text: object) -> None:
    assert math.isnan(parse_laptime(text))


def test_alkamel_csv_padded_headers_and_gt3_filter() -> None:
    raw = (
        b"NUMBER; DRIVER_NUMBER; LAP_NUMBER; LAP_TIME; CROSSING_FINISH_LINE_IN_PIT; CLASS; TEAM; "
        b"MANUFACTURER; DRIVER_NAME\n"
        b"92;1;1;3:58.100;;LMGT3;Team A;Porsche;A. Driver\n"
        b"92;1;2;3:56.900;B;LMGT3;Team A;Porsche;A. Driver\n"
        b"7;1;1;3:30.000;;HYPERCAR;Team B;Toyota;B. Driver\n"
    )
    laps = read_analysis_csv(raw)
    assert list(laps["lap_time_s"].round(1)) == [210.0, 238.1, 236.9]
    gt3 = gt3_only(laps)
    assert set(gt3["car"]) == {"92"}
    assert gt3["in_pit"].tolist() == [False, True]


def test_alkamel_rejects_other_files() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        read_analysis_csv(b"A;B\n1;2\n")


def test_entrant_model_split_handles_glued_licence() -> None:
    assert _split_entrant_model("E: Winward Racing, E6091-I Mercedes-AMG GT3") == (
        "Winward Racing",
        "Mercedes-AMG GT3",
    )
    assert (
        _split_entrant_model("E: Black Falcon, A052Porsche 911 GT3 R (992)")[1]
        == "Porsche 911 GT3 R (992)"
    )


def test_gt3_detection_excludes_cup_cars() -> None:
    assert is_gt3("SP 9", "Porsche 911 GT3 R (992) Evo26")
    assert is_gt3("AT1", "Audi R8 LMS GT3 evo II")
    assert not is_gt3("SP-PRO", "Porsche 911 GT3 CUP MR")
    assert not is_gt3("SP 10", "BMW M4 GT4 EVO (G82)")


def test_session_kind() -> None:
    assert _session_kind("Manche Qualificative 2", "Classement général") == "heat"
    assert _session_kind("Demi-Finales", "Classement Demi A") == "semi"
    assert _session_kind("Finale", "Classement finale") == "final"
    assert _session_kind("Finale", "Classement championnat") is None


# --- checks on the committed real datasets -------------------------------------------


@pytest.fixture(scope="module")
def n24() -> pd.DataFrame:
    return pd.read_csv(DATA / "n24/n24_2026_gt3_laps.csv", dtype={"car": str})


def test_n24_parsed_laps_match_official_classification(n24: pd.DataFrame) -> None:
    classified = n24[n24["status"] == "classified"]
    per_car = classified.groupby("car").agg(
        parsed=("lap", "max"), official=("laps_completed", "first")
    )
    assert (per_car["parsed"] == per_car["official"]).mean() >= 0.95
    assert n24["car"].nunique() >= 40


def test_n24_lap_times_are_plausible(n24: pd.DataFrame) -> None:
    racing = n24[~n24["in_pit"]]
    assert racing["lap_time_s"].quantile(0.05) > 470  # no GT3 laps the 25.4 km track in < 7:50
    assert racing["lap_time_s"].median() < 600


@pytest.fixture(scope="module")
def rx() -> pd.DataFrame:
    return pd.read_csv(DATA / "rx_france/rx_france_2026_laps.csv", dtype={"car_no": str})


def test_rx_heat_time_equals_laps_plus_start(rx: pd.DataFrame) -> None:
    full = rx.groupby(["event", "category", "race", "driver"]).filter(
        lambda g: g["lap_time_s"].notna().all()
    )
    per = full.groupby(["event", "category", "race", "driver"]).agg(
        heat=("heat_time_s", "first"), laps=("lap_time_s", "sum")
    )
    start = per["heat"] - per["laps"]
    assert start.between(0, 15).mean() > 0.9  # heat time = laps + standing-start segment


def test_rx_joker_laps_are_slower(rx: pd.DataFrame) -> None:
    per_race = rx.groupby(["event", "category", "race", "driver"])
    joker = rx[rx["is_joker"]].set_index(["event", "category", "race", "driver"])["lap_time_s"]
    normal = (
        rx[(rx["lap"] > 1) & ~rx["is_joker"]]
        .groupby(["event", "category", "race", "driver"])["lap_time_s"]
        .median()
    )
    diff = (joker - normal).dropna()
    assert len(per_race) > 500
    assert (diff > 0).mean() > 0.9


@pytest.mark.parametrize("year", [2017, 2018, 2020, 2022, 2023, 2026])
def test_every_n24_edition_matches_its_official_lap_count(year: int) -> None:
    laps = pd.read_csv(DATA / f"n24/n24_{year}_gt3_laps.csv", dtype={"car": str})
    per_car = laps.groupby("car").agg(last=("lap", "max"), official=("laps_completed", "first"))
    assert len(per_car) >= 25  # GT3 field
    assert (per_car["last"] == per_car["official"]).mean() >= 0.95
    assert laps["cls"].str.replace(" ", "").eq("SP9").mean() > 0.8
