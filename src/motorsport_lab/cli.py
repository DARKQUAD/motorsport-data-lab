"""Command line: ``mlab data ...``, ``mlab report ...``, ``mlab ml ...``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
CACHE = Path(os.environ.get("MLAB_CACHE", ROOT / ".cache"))


def _data(args: argparse.Namespace) -> None:
    if args.source == "n24":
        from .sources.n24_wige import load_edition

        df = load_edition(args.year, CACHE / "n24")
        path = ROOT / "data" / "n24" / f"n24_{args.year}_gt3_laps.csv"
    else:
        from .sources.its_rallycross import DEFAULT_CATEGORIES, load_season

        other = ("Division 3", "Division 4", "Division 5", "Féminines")
        categories = other if args.other_categories else DEFAULT_CATEGORIES
        df = load_season(str(args.year), categories, cache_dir=CACHE / "its")
        suffix = "_other_categories" if args.other_categories else ""
        path = ROOT / "data" / "rx_france" / f"rx_france_{args.year}{suffix}_laps.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"{len(df)} rows -> {path}")


def _report(args: argparse.Namespace) -> None:
    from . import reports

    if args.what == "n24":
        reports.n24_report(ROOT / "data/n24/n24_2026_gt3_laps.csv", ROOT / "reports/n24")
    else:
        reports.rx_report(
            ROOT / "data/rx_france/rx_france_2026_laps.csv", ROOT / "reports/rx_france"
        )
    print(f"reports written to {ROOT / 'reports'}")


def _forecast(df: pd.DataFrame, out: Path, models: list[str], time_limit: int) -> pd.DataFrame:
    from . import reports
    from .ml import forecasting as fc

    path = out / "forecast_backtest.csv.gz"
    results = fc.backtest(fc.to_series(df), models=tuple(models), ag_time_limit=time_limit)
    if path.exists():  # keep the columns of models that are not re-run (Chronos takes long)
        keys = ["cutoff_hour", "item_id", "timestamp"]
        old = pd.read_csv(path, parse_dates=["timestamp"], dtype={"item_id": str})
        kept = [c for c in old.columns if c in fc.ALL_MODELS and c not in models]
        results = results.merge(old[keys + kept], on=keys, how="left")
    results.to_csv(path, index=False)
    scores = fc.score(results)
    scores.to_csv(out / "forecast_scores.csv", index=False)
    challengers = [m for m in scores["model"] if m not in ("last_clean_lap", "median_5_clean_laps")]
    sig = [
        {
            "model": m,
            "reference": "last_clean_lap",
            "laps": name,
            **fc.paired_bootstrap(results[mask], m, "last_clean_lap"),
        }
        for m in challengers
        for name, mask in (
            ("all", results["clean"] | ~results["clean"]),
            ("clean", results["clean"]),
        )
    ]
    pd.DataFrame(sig).to_csv(out / "forecast_significance.csv", index=False)
    reports.ml_figure(
        scores, out / "forecast_scores.png", "Next-stint forecast, N24 2026 GT3 (rolling backtest)"
    )
    return scores


def _seasons(out: Path, args: argparse.Namespace) -> pd.DataFrame:
    from . import reports
    from .ml import seasons

    laps = seasons.load_all(ROOT / "data" / "n24")
    table = seasons.season_table(laps)
    errors = seasons.next_lap(table, autogluon_time_limit=args.time_limit)
    scores = seasons.summarise(errors, reference=seasons.BASELINES[1])
    scores.to_csv(out / "seasons_next_lap.csv", index=False)
    seasons.conformal(table).to_csv(out / "seasons_conformal.csv", index=False)
    seasons.rating_stability(laps).to_csv(out / "seasons_rating_stability.csv", index=False)
    reports.seasons_figure(scores, out / "seasons_next_lap.png")
    if args.models:
        results = seasons.forecast(laps, args.models.split(","), ag_time_limit=args.time_limit)
        results.to_csv(out / "seasons_forecast_backtest.csv.gz", index=False)
        seasons.forecast_summary(results).to_csv(out / "seasons_forecast.csv", index=False)
    return scores


def _ml(args: argparse.Namespace) -> None:
    from . import reports
    from .endurance.laps import annotate

    out = ROOT / "reports" / "ml"
    out.mkdir(parents=True, exist_ok=True)
    df = annotate(pd.read_csv(ROOT / "data/n24/n24_2026_gt3_laps.csv", dtype={"car": str}))
    if args.task == "forecast":
        scores = _forecast(df, out, args.models.split(","), args.time_limit)
    elif args.task == "tabular":
        from .ml import tabular as tab

        scores, importance = tab.evaluate(df, time_limit=args.time_limit)
        scores.to_csv(out / "tabular_scores.csv", index=False)
        importance.to_csv(out / "tabular_importance.csv", index=False)
        reports.ml_figure(
            scores, out / "tabular_scores.png", "Next-lap model, trained h0-16, tested h16-24"
        )
    elif args.task == "classic":
        from .ml import classic

        scores, best = classic.evaluate(df, n_trials=args.trials)
        scores.to_csv(out / "classic_scores.csv", index=False)
        pd.Series(best, name="value").rename_axis("param").to_csv(
            out / "lightgbm_optuna_params.csv"
        )
        shap = classic.shap_importance(df)
        shap.to_csv(out / "classic_shap.csv", index=False)
        classic.conformal_intervals(df).to_csv(out / "conformal_intervals.csv", index=False)
        reports.ml_figure(
            scores, out / "classic_scores.png", "Next-lap model zoo, trained h0-16, tested h16-24"
        )
        reports.shap_figure(shap, out / "classic_shap.png")
    elif args.task == "seasons":
        scores = _seasons(out, args)
    elif args.task == "ratings":
        from .endurance.ratings import driver_ratings

        scores, summary = driver_ratings(df)
        scores.to_csv(out / "driver_ratings.csv", index=False)
        pd.Series(summary, name="value").rename_axis("quantity").to_csv(
            out / "driver_ratings_summary.csv"
        )
        reports.ratings_figure(scores, out / "driver_ratings.png")
    else:
        from .rallycross import models as rx
        from .rallycross.analysis import per_race

        rx_dir = ROOT / "data" / "rx_france"
        table = rx.win_table(per_race(pd.read_csv(rx_dir / "rx_france_2026_laps.csv")))
        rx.odds_ratios(table).to_csv(out / "rx_win_odds_ratios.csv", index=False)
        scores = rx.cross_validate(table)
        scores.to_csv(out / "rx_win_cv.csv", index=False)
        other = rx_dir / "rx_france_2026_other_categories_laps.csv"
        if other.exists():  # blind test on Division 3 / 4 / 5 and Féminines
            unseen = rx.win_table(per_race(pd.read_csv(other)))
            rx.holdout(table, unseen).to_csv(out / "rx_win_holdout.csv", index=False)
            rx.odds_ratios(unseen).to_csv(out / "rx_win_odds_ratios_holdout.csv", index=False)
    print(scores.round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    """Entry point of the ``mlab`` command."""
    parser = argparse.ArgumentParser(prog="mlab", description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("data", help="download and parse official timing documents")
    p.add_argument("source", choices=["n24", "rx"])
    p.add_argument("--year", type=int, default=2026)
    p.add_argument(
        "--other-categories",
        action="store_true",
        help="rx: Division 3/4/5 and Féminines (the blind test set) instead of Supercar + S1600",
    )
    p.set_defaults(func=_data)
    p = sub.add_parser("report", help="build CSV tables and figures")
    p.add_argument("what", choices=["n24", "rx"])
    p.set_defaults(func=_report)
    p = sub.add_parser("ml", help="models and statistics (needs the [ml] / [models] extras)")
    p.add_argument(
        "task", choices=["forecast", "tabular", "classic", "ratings", "rx-win", "seasons"]
    )
    p.add_argument("--time-limit", type=int, default=120, help="AutoGluon budget (s)")
    p.add_argument(
        "--models",
        default="last_clean_lap,median_5_clean_laps,auto_ets,auto_theta,auto_arima_x",
        help="forecast models to (re)run; others are kept from the previous backtest file",
    )
    p.add_argument("--trials", type=int, default=40, help="Optuna trials for LightGBM")
    p.set_defaults(func=_ml)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
