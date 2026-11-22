"""Build the CSV tables and PNG figures shown in the README and the app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .endurance import laps as lp
from .endurance import strategy as strat
from .rallycross import analysis as rxa
from .rallycross import simulator as rxs

INK, MUTED, ACCENT, ACCENT2, GRID = "#1f2933", "#7b8794", "#d64545", "#2f6fdb", "#e4e7eb"


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _figure(width: float, height: float) -> tuple[Figure, Axes]:
    """A pyplot-free figure (no global state, no GUI backend needed)."""
    fig = Figure(figsize=(width, height))
    return fig, fig.add_subplot()


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def n24_report(laps_csv: Path, out: Path) -> dict[str, pd.DataFrame]:
    """Nürburgring GT3 tables (CSV) and figures (PNG) into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    df = lp.annotate(pd.read_csv(laps_csv, dtype={"car": str}))
    tables = {
        "pit_stops": lp.pit_stops(df),
        "stints": lp.stint_table(df),
        "driver_deltas": lp.driver_deltas(df),
        "car_model_pace": lp.car_model_pace(df),
        "stint_evolution": lp.stint_evolution(df),
    }
    params = strat.calibrate(df)
    tables["strategy"] = strat.compare(params, strat.default_strategies(params), n=3000)
    tables["calibration"] = pd.DataFrame([params.__dict__])
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)

    bucket = df.groupby("bucket").agg(hour=("race_hour", "median"), index=("track_index", "first"))
    fig, ax = _figure(9, 3.4)
    ax.plot(bucket["hour"], bucket["index"], color=ACCENT2, linewidth=1.6)
    ax.axhline(lp.NEUTRAL_THRESHOLD, color=ACCENT, linestyle="--", linewidth=1)
    ax.text(
        0.2,
        lp.NEUTRAL_THRESHOLD + 0.01,
        "neutralised / wet above this line",
        color=ACCENT,
        fontsize=8,
    )
    _style(
        ax,
        "Track index: how slow the whole GT3 field was",
        "race hour",
        "field pace (1.00 = normal)",
    )
    _save(fig, out / "track_index.png")

    top = (
        df[df["status"].eq("classified")]
        .groupby("car")["class_position"]
        .first()
        .nsmallest(5)
        .index
    )
    fig, ax = _figure(9, 3.8)
    for car in top:
        c = df[(df["car"] == car) & df["clean"]]
        ax.scatter(
            c["race_hour"], c["lap_time_s"] / 60, s=8, label=f"#{car} {c['car_model'].iat[0]}"
        )
    _style(ax, "Clean laps of the top-5 GT3 finishers", "race hour", "lap time (min)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _save(fig, out / "top5_clean_laps.png")

    stops = tables["pit_stops"]
    routine = stops[stops["loss_s"] < 600]
    fig, ax = _figure(6, 3.4)
    for flag, color in ((False, MUTED), (True, ACCENT)):
        ax.hist(
            routine[routine["driver_change"] == flag]["loss_s"],
            bins=30,
            alpha=0.7,
            color=color,
            label="driver change" if flag else "same driver",
        )
    _style(ax, "Time lost per pit stop", "seconds lost (in-lap + out-lap)", "stops")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "pit_loss.png")

    pace = tables["car_model_pace"]
    fig, ax = _figure(7, 4)
    ax.barh(
        pace["car_model"][::-1], pace["top_pace_s"][::-1] - pace["top_pace_s"].min(), color=ACCENT2
    )
    _style(ax, "Car models: gap to the fastest (best 20 % of clean laps)", "seconds per lap", "")
    _save(fig, out / "car_model_pace.png")

    strat_table = tables["strategy"]
    fig, ax = _figure(8, 3.6)
    ax.barh(strat_table["strategy"][::-1], strat_table["mean_laps"][::-1], color=ACCENT2)
    ax.set_xlim(strat_table["mean_laps"].min() - 3, strat_table["mean_laps"].max() + 1)
    _style(ax, "Monte-Carlo: average laps in 24 h per strategy", "laps", "")
    _save(fig, out / "strategy.png")
    return tables


def rx_report(laps_csv: Path, out: Path, category: str = "Supercar") -> dict[str, pd.DataFrame]:
    """French rallycross tables (CSV) and figures (PNG) into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    laps = pd.read_csv(laps_csv, dtype={"car_no": str})
    races = rxa.per_race(laps)
    tables = {
        "races": races,
        "tracks": rxa.track_summary(races),
        "joker_timing": rxa.joker_timing(races),
        "pace_vs_position": rxa.pace_vs_position(races),
    }
    finals = races[(races["kind"] == "final") & (races["category"] == category)]
    event = races[races["category"] == category].groupby("event")["race"].count().idxmax()
    event = str(event)
    final_laps = int(finals["laps"].mode().to_numpy()[0]) if not finals.empty else 6
    cars = (
        int(finals.groupby(["event", "race"])["driver"].count().median()) if not finals.empty else 6
    )
    params = rxs.calibrate(races, event, category, laps=final_laps, n_cars=cars)
    policy = rxa.joker_timing(races[races["category"] == category])
    policy_row = policy[policy["kind"] == "final"] if "final" in set(policy["kind"]) else policy
    rival_policy = {
        k: float(policy_row[k].mean()) for k in ("lap 1", "middle", "last lap") if k in policy_row
    }
    heat = rxs.joker_heatmap(params, rival_policy, n=3000)
    tables["joker_heatmap"] = heat.assign(event=event, category=category)
    tables["sim_params"] = pd.DataFrame([{**params.__dict__, "event": event, "category": category}])
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)

    tr = tables["tracks"][tables["tracks"]["category"] == category]
    fig, ax = _figure(7, 3.6)
    ax.bar(tr["event"], tr["joker_cost_s"], color=ACCENT, label="joker lap")
    ax.bar(tr["event"], tr["lap1_cost_s"], color=MUTED, alpha=0.5, label="lap 1 (standing start)")
    ax.tick_params(axis="x", rotation=30)
    _style(ax, f"{category}: extra time vs a normal lap", "", "seconds")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "joker_cost_by_track.png")

    pivot = heat.pivot(index="grid_slot", columns="joker_lap", values="mean_position")
    fig, ax = _figure(6, 4)
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [str(i) for i in pivot.index])
    values = pivot.to_numpy()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(im, ax=ax, label="expected finishing position")
    ax.set_title(
        f"When to joker? {event} {category} final (simulated)",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("joker lap", color=MUTED)
    ax.set_ylabel("grid slot", color=MUTED)
    _save(fig, out / "joker_heatmap.png")
    return tables


def ml_figure(scores: pd.DataFrame, path: Path, title: str) -> None:
    """Bar chart of MAE per model (grey = baselines, blue = ML models)."""
    fig, ax = _figure(7, 3.2)
    colors = [
        ACCENT2
        if "baseline" not in m and "last" not in m and "median" not in m and "previous" not in m
        else MUTED
        for m in scores["model"]
    ]
    ax.barh(scores["model"][::-1], scores["mae_all_s"][::-1], color=colors[::-1])
    _style(ax, title, "mean absolute error (s) — lower is better", "")
    _save(fig, path)
