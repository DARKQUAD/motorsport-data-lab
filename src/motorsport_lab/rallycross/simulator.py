"""Monte-Carlo rallycross race to choose *when* to take the joker lap.

Model, kept deliberately simple and explicit:

* every car has a pace (s/lap) drawn around the field median, plus lap-to-lap noise;
* lap 1 is slower (standing start) and the main route has first-corner incidents;
* the joker lap costs ``joker_cost_s`` but runs on a separate route, so no traffic there;
* on the main route a faster car caught behind a slower one passes with a probability
  that grows with its pace advantage; otherwise it is stuck ``follow_gap_s`` behind.

``calibrate`` takes the track's real numbers (lap time, joker cost, lap-1 cost, lap
noise, spread between drivers) from the ITS Chrono heats; the traffic parameters are
assumptions exposed as sliders in the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RXParams:
    """Track and traffic parameters of one rallycross race."""

    laps: int = 6
    n_cars: int = 6
    normal_lap_s: float = 40.0
    pace_sd_s: float = 0.5
    lap_noise_sd_s: float = 0.35
    joker_cost_s: float = 3.0
    lap1_extra_s: float = 3.0
    start_spacing_s: float = 0.2
    follow_gap_s: float = 0.3
    pass_advantage_s: float = 0.8
    lap1_incident_rate: float = 0.2
    lap1_incident_cost_s: float = 2.0


def calibrate(
    races: pd.DataFrame, event: str, category: str, laps: int = 6, n_cars: int = 6
) -> RXParams:
    """Track parameters from ``analysis.per_race`` rows of one event and category."""
    r = races[(races["event"] == event) & (races["category"] == category)]
    joker = r["joker_cost_s"][r["joker_cost_s"].between(0, 15)]
    lap1 = r["lap1_cost_s"][r["lap1_cost_s"].between(-5, 20)]
    spread = r.groupby("race")["normal_lap_s"].std().median()
    return RXParams(
        laps=laps,
        n_cars=n_cars,
        normal_lap_s=float(r["normal_lap_s"].median()),
        pace_sd_s=float(np.nan_to_num(spread, nan=0.5)),
        joker_cost_s=float(joker.median()) if not joker.empty else 3.0,
        lap1_extra_s=float(lap1.median()) if not lap1.empty else 3.0,
    )


def _sample_jokers(
    policy: dict[str, float], laps: int, size: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    keys = ["lap 1", "middle", "last lap"]
    probs = np.array([policy.get(k, 0.0) for k in keys], dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(3) / 3
    choice = rng.choice(3, size=size, p=probs)
    middle = rng.integers(2, max(laps, 3), size=size)
    return np.where(choice == 0, 1, np.where(choice == 1, middle, laps))


def simulate(
    p: RXParams,
    grid_slot: int,
    joker_lap: int,
    rival_policy: dict[str, float] | None = None,
    n: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """Finishing positions (1 = win) of a median-pace car starting from ``grid_slot`` (1-based)."""
    rng = np.random.default_rng(seed)
    policy = rival_policy or {"lap 1": 0.4, "middle": 0.3, "last lap": 0.3}
    n_cars = p.n_cars
    focal = grid_slot - 1
    rows = np.arange(n)
    pace = rng.normal(0, p.pace_sd_s, size=(n, n_cars))
    pace[:, focal] = 0.0
    jokers = _sample_jokers(policy, p.laps, (n, n_cars), rng)
    jokers[:, focal] = joker_lap
    times = np.tile(np.arange(n_cars) * p.start_spacing_s, (n, 1)).astype(float)
    for lap in range(1, p.laps + 1):
        on_joker = jokers == lap
        d = p.normal_lap_s + pace + rng.normal(0, p.lap_noise_sd_s, size=(n, n_cars))
        d = d + np.where(on_joker, p.joker_cost_s, 0.0)
        if lap == 1:
            incident = (
                (rng.random((n, n_cars)) < p.lap1_incident_rate)
                & ~on_joker
                & (np.arange(n_cars) > 0)
            )
            d = (
                d
                + p.lap1_extra_s
                + np.where(incident, rng.exponential(p.lap1_incident_cost_s, (n, n_cars)), 0.0)
            )
        order = np.argsort(times, axis=1)
        next_times = times.copy()
        ahead_fin = np.full(n, -np.inf)
        ahead_d = np.full(n, np.inf)
        for r in range(n_cars):
            idx = order[:, r]
            d_r = d[rows, idx]
            cand = times[rows, idx] + d_r
            main = ~on_joker[rows, idx]
            blocked = main & (cand < ahead_fin + p.follow_gap_s)
            advantage = ahead_d - d_r
            p_pass = 1 / (1 + np.exp(-(advantage - p.pass_advantage_s) / 0.25))
            passed = rng.random(n) < p_pass
            fin = np.where(blocked & ~passed, ahead_fin + p.follow_gap_s, cand)
            next_times[rows, idx] = fin
            take = main & (fin > ahead_fin)
            ahead_fin = np.where(take, fin, ahead_fin)
            ahead_d = np.where(take, d_r, ahead_d)
        times = next_times
    ranks = np.argsort(np.argsort(times, axis=1), axis=1) + 1
    return np.asarray(ranks[:, focal])


def joker_heatmap(
    p: RXParams, rival_policy: dict[str, float] | None = None, n: int = 4000, seed: int = 0
) -> pd.DataFrame:
    """Expected finishing position for every (grid slot, joker lap) pair."""
    rows = []
    for slot in range(1, p.n_cars + 1):
        for joker in range(1, p.laps + 1):
            pos = simulate(p, slot, joker, rival_policy, n, seed)
            rows.append(
                {
                    "grid_slot": slot,
                    "joker_lap": joker,
                    "mean_position": float(pos.mean()),
                    "p_win": float((pos == 1).mean()),
                    "p_podium": float((pos <= 3).mean()),
                }
            )
    return pd.DataFrame(rows)


def best_joker_by_slot(heatmap: pd.DataFrame) -> pd.DataFrame:
    """For each grid slot, the joker lap with the best expected finishing position."""
    idx = heatmap.groupby("grid_slot")["mean_position"].idxmin()
    return heatmap.loc[idx].reset_index(drop=True)
