"""Monte-Carlo 24h pit-strategy simulator for a GT3 car.

Each simulated race draws random neutralisation periods (Code 60 / full-course yellow /
safety car / weather), then drives one car lap by lap. A strategy decides the stint length,
how often tyres are changed, and whether to stop early when the race is neutralised (a stop
under neutralisation costs less because the field is slow too).

The Nürburgring preset is calibrated from the real 2026 lap table (``calibrate``); the
Le Mans and Spa presets are documented approximations to be tuned by the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .laps import pit_stops


@dataclass(frozen=True)
class TrackParams:
    """Everything the simulator needs to know about a track and a GT3 car."""

    name: str
    green_lap_s: float
    lap_sd_s: float
    fuel_stint_laps: int
    stop_loss_s: float
    tyre_change_extra_s: float
    tyre_deg_s_per_lap: float
    neutral_share: float
    neutral_mean_min: float
    neutral_slowdown: float
    neutral_stop_saving: float = 0.4
    race_s: float = 24 * 3600


@dataclass(frozen=True)
class Strategy:
    """A pit-stop plan: stint length, tyre changes, reaction to neutralisations."""

    name: str
    stint_laps: int | None = None  # None = use the fuel limit
    tyres_every: int = 1  # change tyres every N stops
    pit_early_window: int = 0  # stop up to N laps early if the race is neutralised


PRESETS: dict[str, TrackParams] = {
    # Approximations (lap ~3:56, ~14-lap fuel stints in LMGT3). Tune before use.
    "le_mans": TrackParams(
        "Le Mans 24h (LMGT3, approx.)", 236.0, 1.2, 14, 75.0, 30.0, 0.05, 0.12, 12.0, 1.35
    ),
    # Approximations (lap ~2:20, ~27-lap stints). Tune before use.
    "spa_24h": TrackParams(
        "Spa 24h (GT3, approx.)", 140.0, 0.9, 27, 70.0, 20.0, 0.03, 0.10, 8.0, 1.30
    ),
}


def calibrate(df: pd.DataFrame, name: str = "Nürburgring 24h (GT3, calibrated)") -> TrackParams:
    """Derive Nürburgring parameters from an annotated lap table (see ``laps.annotate``)."""
    clean = df[df["clean"]]
    top_cars = clean.groupby("car")["lap_time_s"].median().nsmallest(10).index
    top = clean[clean["car"].isin(top_cars)]
    stops = pit_stops(df)
    routine = stops[(stops["stint_laps"] >= 5) & (stops["loss_s"] < stops["loss_s"].quantile(0.9))]
    neutral_share = float(df.drop_duplicates("bucket")["neutralised"].mean())
    stint_laps = int(stops["stint_laps"].mode().to_numpy()[0])
    return TrackParams(
        name=name,
        green_lap_s=float(top["lap_time_s"].median()),
        lap_sd_s=float(top.groupby("car")["lap_time_s"].std().median()),
        fuel_stint_laps=stint_laps,
        stop_loss_s=float(routine["loss_s"].quantile(0.25)),
        tyre_change_extra_s=float(routine["loss_s"].median() - routine["loss_s"].quantile(0.25)),
        tyre_deg_s_per_lap=0.08,  # not identifiable from timing data alone (see README)
        neutral_share=neutral_share,
        neutral_mean_min=30.0,
        neutral_slowdown=float(df.loc[df["neutralised"], "track_index"].median()),
    )


def _neutral_windows(
    p: TrackParams, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    mean_s = p.neutral_mean_min * 60
    rate = p.neutral_share / max(mean_s * (1 - p.neutral_share), 1e-9)
    k = max(int(rate * p.race_s * 3) + 5, 5)
    gaps = rng.exponential(1 / max(rate, 1e-12), size=(n, k))
    durations = rng.exponential(mean_s, size=(n, k))
    starts = np.cumsum(gaps + np.concatenate([np.zeros((n, 1)), durations[:, :-1]], axis=1), axis=1)
    return starts, starts + durations


def simulate(p: TrackParams, s: Strategy, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Return one row per simulated race: laps completed and number of stops."""
    rng = np.random.default_rng(seed)
    starts, ends = _neutral_windows(p, n, rng)
    stint_target = min(s.stint_laps or p.fuel_stint_laps, p.fuel_stint_laps)
    t = np.zeros(n)
    laps = np.zeros(n, dtype=int)
    stops = np.zeros(n, dtype=int)
    lap_in_stint = np.zeros(n, dtype=int)
    tyre_age = np.zeros(n, dtype=int)
    stops_since_tyres = np.zeros(n, dtype=int)
    while True:
        racing = t < p.race_s
        if not racing.any():
            break
        neutral = ((t[:, None] >= starts) & (t[:, None] < ends)).any(axis=1)
        lap = p.green_lap_s + p.tyre_deg_s_per_lap * tyre_age + rng.normal(0, p.lap_sd_s, n)
        lap = np.where(neutral, lap * p.neutral_slowdown, lap)
        lap_in_stint += 1
        tyre_age += 1
        must_stop = lap_in_stint >= stint_target
        early = (
            (s.pit_early_window > 0) & neutral & (lap_in_stint >= stint_target - s.pit_early_window)
        )
        stop = (must_stop | early) & racing
        tyres = stop & (stops_since_tyres + 1 >= s.tyres_every)
        loss = p.stop_loss_s + np.where(tyres, p.tyre_change_extra_s, 0.0)
        loss = np.where(neutral, loss * (1 - p.neutral_stop_saving), loss)
        lap = lap + np.where(stop, loss, 0.0)
        t = np.where(racing, t + lap, t)
        laps += racing
        stops += stop
        lap_in_stint = np.where(stop, 0, lap_in_stint)
        tyre_age = np.where(tyres, 0, tyre_age)
        stops_since_tyres = np.where(
            tyres, 0, np.where(stop, stops_since_tyres + 1, stops_since_tyres)
        )
    return pd.DataFrame({"laps": laps, "stops": stops})


def compare(
    p: TrackParams, strategies: list[Strategy], n: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Same random races (common random numbers) for every strategy, so differences are fair."""
    rows = []
    results = {s.name: simulate(p, s, n, seed) for s in strategies}
    best = np.max(np.stack([r["laps"].to_numpy() for r in results.values()]), axis=0)
    for name, r in results.items():
        rows.append(
            {
                "strategy": name,
                "mean_laps": r["laps"].mean(),
                "p10_laps": r["laps"].quantile(0.1),
                "p90_laps": r["laps"].quantile(0.9),
                "mean_stops": r["stops"].mean(),
                "share_best": float((r["laps"].to_numpy() == best).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_laps", ascending=False).reset_index(drop=True)


def default_strategies(p: TrackParams) -> list[Strategy]:
    """Five classic 24h plans built around the fuel-limited stint length."""
    full = p.fuel_stint_laps
    return [
        Strategy("Full stints, new tyres every stop", full, 1, 0),
        Strategy("Full stints, double-stint tyres", full, 2, 0),
        Strategy("Full stints, new tyres, pit early under neutralisation", full, 1, 2),
        Strategy("Double-stint tyres + pit early under neutralisation", full, 2, 2),
        Strategy("Short stints (one lap less), new tyres", full - 1, 1, 0),
    ]
