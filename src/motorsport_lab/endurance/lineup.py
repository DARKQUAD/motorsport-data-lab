"""Who drives the next stint? Learning each driver's pace during the race.

At every pit stop the team picks the next driver. Each driver has an unknown pace and an
unknown night delta (see :mod:`driver_style`). The rules force every driver to drive a
minimum number of stints and cap consecutive stints. This is a constrained multi-armed
bandit, the simplest form of reinforcement learning: each stint is an action whose reward
(the stint time) also teaches the team about that driver.

Policies compared on the same simulated races (common random numbers):

* ``rotation``: A, B, C, A, B, C... (what most line-ups look like on paper);
* ``last_race``: greedy on last race's profile; its link to the truth is set by
  ``prior_corr`` (measured at 0.1-0.4 on real data, see :func:`driver_style.stability`);
* ``learn_greedy``: Bayesian update of every driver's pace and night delta after each stint,
  then the fastest feasible driver for the next stint;
* ``thompson``: same posterior, but the choice uses one random draw from it (it explores
  drivers it is unsure about, the classic bandit / RL answer);
* ``history_learn``: starts from last race's profile, weighted by how reliable it is, then
  learns during the race (what a team should do);
* ``oracle``: knows the true values (the upper bound no team reaches).

Neutralisations, pit-stop losses and weather affect every policy the same way, so they
are left out: only driver differences matter here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .driver_style import NIGHT_HOURS


@dataclass(frozen=True)
class LineupParams:
    """One car, one 24h race. Defaults come from the Nürburgring data (see README)."""

    drivers: int = 3
    stints: int = 19  # about 156 laps / 8-lap stints
    laps_per_stint: int = 8
    lap_s: float = 500.0
    pace_sd_s: float = 4.0  # between-driver spread of pace (measured)
    night_sd_s: float = 4.3  # between-driver spread of the night delta (measured)
    lap_noise_s: float = 6.6  # lap-to-lap noise once car and conditions are known (measured)
    # stint-to-stint form of a driver (traffic, tyres, fatigue). 4.5 s makes the simulated
    # first-half / second-half reliability of a driver's pace equal the measured ~0.38
    stint_sd_s: float = 4.5
    min_stints: int = 4  # regulation-style minimum per driver
    max_consecutive: int = 2  # double stint allowed, triple not
    prior_corr: float = 0.2  # how well last race's profile predicts this race (measured)

    def stint_var(self) -> float:
        """Variance of one stint's mean around the driver's true pace."""
        return self.stint_sd_s**2 + self.lap_noise_s**2 / self.laps_per_stint

    def reliability(self) -> float:
        """Correlation between two independent stints of the same drivers (split-half)."""
        return self.pace_sd_s**2 / (self.pace_sd_s**2 + self.stint_var())

    def night(self) -> np.ndarray:
        """True for stints whose middle falls in the night window."""
        hours = (np.arange(self.stints) + 0.5) * 24 / self.stints
        return (hours >= NIGHT_HOURS[0]) & (hours <= NIGHT_HOURS[1])


def _feasible(p: LineupParams, counts: np.ndarray, last: int, run: int, stint: int) -> np.ndarray:
    """Drivers allowed for ``stint`` given counts so far, the last driver and his run."""
    left_after = p.stints - stint - 1
    ok = np.ones(p.drivers, bool)
    if last >= 0 and run >= p.max_consecutive:
        ok[last] = False
    for d in range(p.drivers):
        need = np.maximum(p.min_stints - counts, 0)
        need[d] = max(p.min_stints - counts[d] - 1, 0)
        if need.sum() > left_after:
            ok[d] = False
    if not ok.any():  # cannot happen with sane parameters; never block the car
        ok[:] = True
    return ok


class _Posterior:
    """Bayesian linear model per driver: stint mean = pace + night_delta * is_night."""

    def __init__(self, p: LineupParams, prior_mean: np.ndarray, prior_corr: float = 0.0) -> None:
        self.var = p.stint_var()
        shrink = max(1 - prior_corr**2, 1e-3)  # what last race's profile leaves unexplained
        prior_prec = np.diag([1 / (p.pace_sd_s**2 * shrink), 1 / (p.night_sd_s**2 * shrink)])
        self.prec = np.repeat(prior_prec[None], p.drivers, axis=0)
        self.b = np.einsum("ij,dj->di", prior_prec, prior_mean)

    def update(self, d: int, night: bool, y: float) -> None:
        """Add one stint of driver ``d`` (mean relative lap time ``y``)."""
        x = np.array([1.0, float(night)])
        self.prec[d] += np.outer(x, x) / self.var
        self.b[d] += x * y / self.var

    def mean(self) -> np.ndarray:
        """Posterior mean of (pace, night delta) for every driver."""
        return np.linalg.solve(self.prec, self.b[..., None])[..., 0]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of (pace, night delta) per driver (Thompson sampling)."""
        cov = np.linalg.inv(self.prec)
        return np.array(
            [rng.multivariate_normal(m, c) for m, c in zip(self.mean(), cov, strict=True)]
        )


def _race(
    p: LineupParams,
    truth: np.ndarray,
    noise: np.ndarray,
    policy: str,
    prior: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Total driver-dependent time (s) of one race for one policy."""
    night = p.night()
    uses_history = policy in ("last_race", "history_learn")
    post = (
        _Posterior(p, prior, p.prior_corr) if uses_history else _Posterior(p, np.zeros_like(prior))
    )
    counts = np.zeros(p.drivers, int)
    last, run, total = -1, 0, 0.0
    for s in range(p.stints):
        ok = _feasible(p, counts, last, run, s)
        if policy == "rotation":
            order = [(last + k) % p.drivers for k in range(1, p.drivers + 1)]
            d = next(x for x in order if ok[x])
        else:
            if policy == "oracle":
                est = truth
            elif policy == "thompson":
                est = post.sample(rng)
            else:
                est = post.mean()
            cost = est[:, 0] + est[:, 1] * night[s]
            d = int(np.argmin(np.where(ok, cost, np.inf)))
        y = truth[d, 0] + truth[d, 1] * night[s] + noise[s, d]
        total += y * p.laps_per_stint
        if policy in ("learn_greedy", "thompson", "history_learn"):
            post.update(d, bool(night[s]), y)
        counts[d] += 1
        run = run + 1 if d == last else 1
        last = d
    return total


POLICIES = ("rotation", "last_race", "learn_greedy", "thompson", "history_learn", "oracle")


def compare(p: LineupParams, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean time and laps gained vs ``rotation`` for every policy, on ``n`` simulated races."""
    rng = np.random.default_rng(seed)
    results = {k: np.empty(n) for k in POLICIES}
    for i in range(n):
        truth = np.column_stack(
            [rng.normal(0, p.pace_sd_s, p.drivers), rng.normal(0, p.night_sd_s, p.drivers)]
        )
        # last race's estimate correlates with the truth at prior_corr
        r = p.prior_corr
        sd = np.array([p.pace_sd_s, p.night_sd_s])
        prior = r * truth + np.sqrt(1 - r**2) * rng.normal(0, 1, truth.shape) * sd
        prior = r * prior  # shrink: the best linear predictor of the truth
        noise = rng.normal(0, np.sqrt(p.stint_var()), (p.stints, p.drivers))
        for k in POLICIES:
            results[k][i] = _race(p, truth, noise, k, prior, np.random.default_rng(i))
    base = results["rotation"]
    rows = []
    for k in POLICIES:
        gain = base - results[k]
        rows.append(
            {
                "policy": k,
                "time_gained_s": float(gain.mean()),
                "ci_low_s": float(np.percentile(gain, 2.5)),
                "ci_high_s": float(np.percentile(gain, 97.5)),
                "laps_gained": float(gain.mean() / p.lap_s),
                "share_of_races_better": float((gain > 0).mean()),
                "share_of_oracle": float(gain.mean() / (base - results["oracle"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def value_of_history(
    p: LineupParams, corrs: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 1.0), n: int = 1000
) -> pd.DataFrame:
    """How much the ``last_race`` and ``thompson`` policies gain as history gets more reliable."""
    rows = []
    for c in corrs:
        table = compare(LineupParams(**{**p.__dict__, "prior_corr": c}), n=n).set_index("policy")
        rows.append(
            {
                "prior_corr": c,
                "last_race_s": table.loc["last_race", "time_gained_s"],
                "learn_greedy_s": table.loc["learn_greedy", "time_gained_s"],
                "thompson_s": table.loc["thompson", "time_gained_s"],
                "history_learn_s": table.loc["history_learn", "time_gained_s"],
                "oracle_s": table.loc["oracle", "time_gained_s"],
            }
        )
    return pd.DataFrame(rows)
