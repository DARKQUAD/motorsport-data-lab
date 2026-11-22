"""Stints, pit stops and clean-lap detection.

Vocabulary (see the README glossary):

* in-lap: the lap that ends in the pit lane (``in_pit``);
* out-lap: the next lap, which includes the time stationary in the pits;
* stint: the laps between two pit stops;
* track index: field-wide slowdown at a given moment (1.00 = normal, 1.20 = 20 % slower),
  which captures Code 60 / full-course yellow, rain and fog at the Nürburgring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUCKET_S = 600  # 10-minute buckets for the field-wide track index
NEUTRAL_THRESHOLD = 1.06  # field 6 % slower than normal = neutralised or wet


def annotate(laps: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed time, stint numbers, out-laps, driver stints and the track index."""
    df = laps.sort_values(["car", "lap"]).reset_index(drop=True).copy()
    df["car"] = df["car"].astype(str)
    by_car = df.groupby("car", sort=False)
    df["elapsed_s"] = by_car["lap_time_s"].cumsum()
    df["race_hour"] = df["elapsed_s"] / 3600
    prev_in_pit = by_car["in_pit"].shift(1, fill_value=False).astype(bool)
    df["out_lap"] = prev_in_pit & (df["lap"] > 1)
    df["stint"] = prev_in_pit.astype(int).groupby(df["car"]).cumsum() + 1
    df["lap_in_stint"] = df.groupby(["car", "stint"]).cumcount() + 1
    driver_change = by_car["driver_idx"].diff().fillna(0).ne(0)
    df["driver_stint"] = driver_change.astype(int).groupby(df["car"]).cumsum() + 1

    racing = ~(df["in_pit"] | df["out_lap"]) & (df["lap"] > 1)
    ref = df[racing].groupby("car")["lap_time_s"].quantile(0.05)
    df["car_ref_s"] = df["car"].map(ref)
    df["pace_ratio"] = df["lap_time_s"] / df["car_ref_s"]
    df["bucket"] = (df["elapsed_s"] // BUCKET_S).astype(int)
    raw_index = df[racing].groupby("bucket")["pace_ratio"].median()
    # 1.00 = the field's usual green-flag level (20th percentile of all 10-minute buckets)
    index = raw_index / raw_index.quantile(0.2)
    df["track_index"] = df["bucket"].map(index).ffill().bfill()
    df["neutralised"] = df["track_index"] > NEUTRAL_THRESHOLD
    df["clean"] = racing & ~df["neutralised"] & (df["pace_ratio"] / df["track_index"] < 1.06)
    return df


def stint_evolution(df: pd.DataFrame, min_laps: int = 5) -> pd.DataFrame:
    """Within-stint lap-time slope (s/lap) per car model, on clean laps only.

    Each stint is centred on its own mean (car x stint fixed effects), so the slope only
    compares laps of the same car, same driver, same fuel load cycle. A negative value means
    the car speeds up as fuel burns faster than the tyres degrade. Stints of the field's
    neutralised periods are excluded. 95 % intervals come from a bootstrap over stints.
    """
    clean = df[df["clean"]].copy()
    clean["adj_s"] = clean["lap_time_s"] / clean["track_index"]
    counts = clean.groupby(["car", "stint"])["lap"].transform("count")
    clean = clean[(counts >= min_laps) & (clean["lap_in_stint"] > 1)]
    grp = clean.groupby(["car", "stint"])
    clean["y"] = clean["adj_s"] - grp["adj_s"].transform("mean")
    clean["x"] = clean["lap_in_stint"] - grp["lap_in_stint"].transform("mean")
    rng = np.random.default_rng(0)
    rows = []
    for model, part in clean.groupby("car_model"):
        keys = part.groupby(["car", "stint"]).ngroup()
        n = int(keys.max()) + 1 if len(part) else 0
        if n < 3:
            continue
        sxy = part["x"].mul(part["y"]).groupby(keys).sum().to_numpy()
        sxx = part["x"].pow(2).groupby(keys).sum().to_numpy()
        boots = []
        for _ in range(500):
            pick = rng.integers(0, n, n)
            boots.append(float(sxy[pick].sum()) / max(float(sxx[pick].sum()), 1e-9))
        rows.append(
            {
                "car_model": model,
                "slope_s_per_lap": float(sxy.sum()) / float(sxx.sum()),
                "ci_low": float(np.percentile(boots, 2.5)),
                "ci_high": float(np.percentile(boots, 97.5)),
                "stints": n,
                "laps": len(part),
            }
        )
    columns = ["car_model", "slope_s_per_lap", "ci_low", "ci_high", "stints", "laps"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns).sort_values(by="slope_s_per_lap").reset_index(drop=True)
    )


def pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stop: time lost vs. the car's clean pace, and whether drivers swapped."""
    clean_pace = df[df["clean"]].groupby(["car", "stint"])["lap_time_s"].median()
    car_pace = df[df["clean"]].groupby("car")["lap_time_s"].median()
    stops = []
    for car, group in df.groupby("car", sort=False):
        group = group.reset_index(drop=True)
        for i in group.index[group["in_pit"]]:
            if i + 1 >= len(group):
                continue
            in_lap, out_lap = group.loc[i], group.loc[i + 1]
            pace = clean_pace.get((car, in_lap["stint"]), car_pace.get(car, np.nan))
            stops.append(
                {
                    "car": car,
                    "car_model": in_lap.get("car_model"),
                    "lap": int(in_lap["lap"]),
                    "race_hour": float(in_lap["race_hour"]),
                    "stint": int(in_lap["stint"]),
                    "stint_laps": int(in_lap["lap_in_stint"]),
                    "loss_s": float(in_lap["lap_time_s"] + out_lap["lap_time_s"] - 2 * pace),
                    "driver_change": bool(in_lap["driver_idx"] != out_lap["driver_idx"]),
                    "under_neutralisation": bool(in_lap["neutralised"] or out_lap["neutralised"]),
                }
            )
    return pd.DataFrame(stops)


def stint_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stint summary: driver, length, clean-lap median and spread."""
    clean = df[df["clean"]]
    agg = df.groupby(["car", "stint"]).agg(
        car_model=("car_model", "first"),
        driver=("driver", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        start_hour=("race_hour", "min"),
        laps=("lap", "count"),
    )
    pace = clean.groupby(["car", "stint"])["lap_time_s"].agg(
        clean_laps="count", median_s="median", std_s="std"
    )
    return agg.join(pace).reset_index()


def driver_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Driver pace vs. team-mates in the same car, corrected for track conditions.

    Each clean lap is divided by the track index of its 10-minute bucket, so a driver who
    drove at night or in drizzle is not penalised for conditions the whole field had.
    """
    clean = df[df["clean"]].copy()
    clean["adj_s"] = clean["lap_time_s"] / clean["track_index"]
    per_driver = (
        clean.groupby(["car", "car_model", "driver"])["adj_s"]
        .agg(["median", "count"])
        .reset_index()
    )
    per_driver["car_median"] = per_driver.groupby("car")["median"].transform("median")
    per_driver["delta_s"] = per_driver["median"] - per_driver["car_median"]
    return per_driver.rename(columns={"median": "adj_median_s", "count": "clean_laps"})


def car_model_pace(df: pd.DataFrame, top_quantile: float = 0.2) -> pd.DataFrame:
    """Balance-of-performance lens: best-20 % condition-adjusted laps per car model."""
    clean = df[df["clean"]].copy()
    clean["adj_s"] = clean["lap_time_s"] / clean["track_index"]
    grouped = clean.groupby("car_model")
    out = pd.DataFrame(
        {
            "best_s": grouped["adj_s"].min(),
            "top_pace_s": grouped["adj_s"].apply(lambda s: s[s <= s.quantile(top_quantile)].mean()),
            "laps": grouped["adj_s"].count(),
            "cars": grouped["car"].nunique(),
        }
    )
    return out.sort_values(by="top_pace_s").reset_index()
