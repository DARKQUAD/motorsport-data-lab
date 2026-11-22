"""Driver "style" from timing data only, and whether it carries over to the next race.

Without telemetry (throttle, brake, steering) a driver's style can only be seen through lap
times. Each clean lap is divided by the field-wide track index, then compared with the
median of the same car, so the car and the conditions cancel out. Per driver and race:

* ``pace_s``: median gap to the car (< 0 = faster than the car's usual pace);
* ``consistency_s``: spread of those gaps (inter-quartile range, lower = steadier);
* ``fade_s_per_lap``: how the gap grows along a stint (tyre and fuel management);
* ``night_delta_s``: pace at night minus pace in daylight (race hours 6-14 approximate the
  night for a mid-afternoon start);
* ``warmup_s``: first flying lap of a stint minus the driver's usual pace (cold tyres).

:func:`stability` then asks the question that matters before using any of this in a
strategy: does a driver's value in one race predict it in the next one?
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

NIGHT_HOURS = (6.0, 14.0)
TRAITS = ("pace_s", "consistency_s", "fade_s_per_lap", "night_delta_s", "warmup_s")


def relative_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Clean laps with ``rel_s`` = condition-adjusted lap minus the car's median."""
    clean = df[df["clean"] & df["driver"].notna()].copy()
    clean["adj_s"] = clean["lap_time_s"] / clean["track_index"]
    clean["rel_s"] = clean["adj_s"] - clean.groupby("car")["adj_s"].transform("median")
    clean["night"] = clean["race_hour"].between(*NIGHT_HOURS)
    return clean


def _slope(g: pd.DataFrame) -> float:
    if g["lap_in_stint"].nunique() < 3:
        return float("nan")
    x = g["lap_in_stint"].to_numpy(float)
    y = g["rel_s"].to_numpy(float)
    return float(np.polyfit(x - x.mean(), y, 1)[0])


def profiles(df: pd.DataFrame, min_laps: int = 8) -> pd.DataFrame:
    """One row per driver (and car) with the five traits and the number of clean laps."""
    rel = relative_laps(df)
    rows = []
    for (driver, car), g in rel.groupby(["driver", "car"]):
        if len(g) < min_laps:
            continue
        pace = float(g["rel_s"].median())
        day, night = g[~g["night"]]["rel_s"], g[g["night"]]["rel_s"]
        warm = g[g["lap_in_stint"] == 2]["rel_s"]
        rows.append(
            {
                "driver": driver,
                "car": car,
                "car_model": g["car_model"].iat[0],
                "clean_laps": len(g),
                "pace_s": pace,
                "consistency_s": float(g["rel_s"].quantile(0.75) - g["rel_s"].quantile(0.25)),
                "fade_s_per_lap": _slope(g),
                "night_delta_s": float(night.median() - day.median())
                if len(night) >= 3 and len(day) >= 3
                else np.nan,
                "warmup_s": float(warm.median() - pace) if len(warm) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stability(laps: pd.DataFrame, min_laps: int = 8) -> pd.DataFrame:
    """Spearman correlation of each trait between consecutive races (drivers in both).

    ``laps`` holds several races with a ``season`` column. A trait with a correlation near
    zero is noise for one race, and a strategy should not be tailored to it.
    """
    seasons = [
        int(s)
        for s in sorted(laps["season"].unique())
        if laps.loc[laps["season"] == s, "driver"].notna().mean() > 0.5
    ]
    per = {
        s: profiles(laps[laps["season"] == s], min_laps).groupby("driver")[list(TRAITS)].mean()
        for s in seasons
    }
    rows = []
    for a, b in pairwise(seasons):
        both = per[a].join(per[b], lsuffix="_a", rsuffix="_b", how="inner")
        for trait in TRAITS:
            pair = both[[f"{trait}_a", f"{trait}_b"]].dropna()
            rows.append(
                {
                    "from_season": a,
                    "to_season": b,
                    "trait": trait,
                    "drivers": len(pair),
                    "spearman": float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
                    if len(pair) >= 5
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def trait_spread(laps: pd.DataFrame, min_laps: int = 8) -> pd.Series:
    """Between-driver standard deviation of each trait (all races pooled)."""
    frames = [profiles(g, min_laps) for _, g in laps.groupby("season")]
    return pd.concat(frames)[list(TRAITS)].std()


def within_race_reliability(laps: pd.DataFrame, split_hour: float = 12.0) -> pd.DataFrame:
    """Does a driver's pace in the first half of a race predict it in the second half?

    This is the assumption behind learning a driver's pace *during* the race
    (:mod:`lineup`). Spearman correlation across drivers with 4+ clean laps in each half.
    """
    rows = []
    for season, g in laps.groupby("season"):
        rel = relative_laps(g)
        if rel.empty:
            continue
        half = np.where(rel["race_hour"] < split_hour, "first", "second")
        per = rel.groupby([rel["driver"], half])["rel_s"].agg(["median", "count"]).unstack()
        per = per[(per[("count", "first")] >= 4) & (per[("count", "second")] >= 4)]
        rows.append(
            {
                "season": int(str(season)),
                "drivers": len(per),
                "spearman": float(
                    per[("median", "first")].corr(per[("median", "second")], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)
