"""Empirical joker-lap analysis on real heats (one row per driver, race and lap).

* joker cost: how much slower the joker lap is than the same driver's normal laps
  in the same race (normal = laps 2..N without joker, median);
* lap-1 cost: the standing start plus the first-corner fight, measured the same way;
* joker timing: on which lap drivers take the joker, and how that relates to results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RACE_KEYS = ["event", "category", "race", "driver"]


def per_race(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver and race with joker lap, joker cost and lap-1 cost."""
    df = laps.dropna(subset=["lap_time_s"]).copy()
    rows = []
    for key, g in df.groupby(RACE_KEYS, sort=False):
        normal = g[(g["lap"] > 1) & ~g["is_joker"]]["lap_time_s"]
        if normal.empty:
            continue
        base = float(normal.median())
        joker = g[g["is_joker"]]
        lap1 = g[(g["lap"] == 1) & ~g["is_joker"]]["lap_time_s"]
        rows.append(
            {
                **dict(zip(RACE_KEYS, key, strict=True)),
                "date": g["date"].iat[0],
                "kind": g["kind"].iat[0],
                "position": g["position"].iat[0],
                "laps": int(g["lap"].max()),
                "normal_lap_s": base,
                "best_lap_s": float(g["lap_time_s"].min()),
                "joker_lap": int(joker["lap"].to_numpy()[0]) if not joker.empty else None,
                "joker_cost_s": float(joker["lap_time_s"].to_numpy()[0]) - base
                if not joker.empty
                else np.nan,
                "lap1_cost_s": float(lap1.to_numpy()[0]) - base if not lap1.empty else np.nan,
                "heat_time_s": g["heat_time_s"].iat[0],
            }
        )
    out = pd.DataFrame(rows)
    out["joker_when"] = np.select(
        [out["joker_lap"].eq(1), out["joker_lap"].eq(out["laps"]), out["joker_lap"].notna()],
        ["lap 1", "last lap", "middle"],
        default="none",
    )
    return out


def track_summary(races: pd.DataFrame) -> pd.DataFrame:
    """Per event and category: typical lap, joker cost and lap-1 cost (robust medians)."""
    valid = races[races["joker_cost_s"].between(0, 15)]
    out = valid.groupby(["event", "category"]).agg(
        date=("date", "first"),
        normal_lap_s=("normal_lap_s", "median"),
        joker_cost_s=("joker_cost_s", "median"),
        joker_cost_iqr=("joker_cost_s", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        lap1_cost_s=("lap1_cost_s", "median"),
        races=("race", "count"),
    )
    return out.reset_index().sort_values(["category", "date"])


def joker_timing(races: pd.DataFrame) -> pd.DataFrame:
    """Share of jokers taken on lap 1 / middle / last lap, per category and race type."""
    taken = races[races["joker_when"] != "none"]
    table = pd.crosstab([taken["category"], taken["kind"]], taken["joker_when"], normalize="index")
    return table.reset_index()


def pace_vs_position(races: pd.DataFrame) -> pd.DataFrame:
    """In finals: where did drivers who jokered on lap 1 vs later finish, at equal pace?

    Drivers are ranked by their own best lap inside the race (a pace proxy); ``gain`` is
    pace rank minus finishing position (positive = finished ahead of their pace).
    """
    finals = races[races["kind"] == "final"].copy()
    finals["pace_rank"] = finals.groupby(["event", "category", "race"])["best_lap_s"].rank()
    finals["gain"] = finals["pace_rank"] - finals["position"]
    return finals.groupby(["category", "joker_when"])["gain"].agg(["mean", "count"]).reset_index()
