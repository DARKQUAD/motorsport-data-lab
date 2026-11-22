"""Driver ratings with a linear mixed-effects model (partial pooling).

The raw "delta to team-mates" of :func:`motorsport_lab.endurance.laps.driver_deltas` is
noisy for a driver with 6 clean laps and cannot compare drivers of different cars. Here
every clean lap (divided by the field-wide track index) is modelled as::

    adjusted lap = car effect (fixed) + a * lap in stint + b * race hour
                   + driver effect (random, shared across cars) + noise

The driver effect is shrunk towards zero when a driver has few laps (empirical Bayes),
and drivers who raced two cars link the cars together. Fitted with statsmodels' REML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z90 = 1.645


def driver_ratings(df: pd.DataFrame, min_laps: int = 3) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return (one row per driver, model summary).

    ``effect_s`` < 0 means faster than the model's expectation for that car, in seconds
    per Nürburgring lap; ``low_s``/``high_s`` is a 90 % interval from the conditional
    variance of the random effect.
    """
    import statsmodels.formula.api as smf

    clean = df[df["clean"]].dropna(subset=["driver"]).copy()
    clean["adj_s"] = clean["lap_time_s"] / clean["track_index"]
    for key in ("driver", "car"):  # a car or driver with 1-2 laps carries no information
        clean = clean[clean.groupby(key)["adj_s"].transform("count") >= min_laps]
    clean["race_hour"] = clean["race_hour"] - 12.0  # centred: better-conditioned fit
    model = smf.mixedlm("adj_s ~ C(car) + lap_in_stint + race_hour", clean, groups=clean["driver"])
    # Powell (derivative-free): L-BFGS hits a singular gradient step on this design
    fit = model.fit(reml=True, method="powell")
    effects = {name: float(v.iloc[0]) for name, v in fit.random_effects.items()}
    cond_sd = {
        name: float(np.sqrt(np.asarray(c)[0, 0])) for name, c in fit.random_effects_cov.items()
    }
    info = clean.groupby("driver").agg(
        car=("car", lambda s: "/".join(sorted(s.astype(str).unique()))),
        car_model=("car_model", "first"),
        clean_laps=("adj_s", "count"),
        raw_median_s=("adj_s", "median"),
    )
    out = info.assign(
        effect_s=pd.Series(effects),
        sd_s=pd.Series(cond_sd),
    )
    out["low_s"] = out["effect_s"] - Z90 * out["sd_s"]
    out["high_s"] = out["effect_s"] + Z90 * out["sd_s"]
    out = out.reset_index().sort_values("effect_s").reset_index(drop=True)
    driver_var = float(np.asarray(fit.cov_re)[0, 0])
    summary = {
        "driver_sd_s": float(np.sqrt(driver_var)),
        "residual_sd_s": float(np.sqrt(fit.scale)),
        "icc": driver_var / (driver_var + float(fit.scale)),
        "lap_in_stint_s": float(fit.fe_params["lap_in_stint"]),
        "race_hour_s": float(fit.fe_params["race_hour"]),
        "drivers": float(len(out)),
        "laps": float(len(clean)),
    }
    return out, summary
