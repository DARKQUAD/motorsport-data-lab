"""Command line: ``mlab data ...``, ``mlab report ...``, ``mlab ml ...``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path.cwd()
CACHE = Path(os.environ.get("MLAB_CACHE", ROOT / ".cache"))


def _data(args: argparse.Namespace) -> None:
    if args.source == "n24":
        from .sources.n24_wige import load_edition

        df = load_edition(args.year, CACHE / "n24")
        path = ROOT / "data" / "n24" / f"n24_{args.year}_gt3_laps.csv"
    else:
        from .sources.its_rallycross import load_season

        df = load_season(str(args.year), cache_dir=CACHE / "its")
        path = ROOT / "data" / "rx_france" / f"rx_france_{args.year}_laps.csv"
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


def _ml(args: argparse.Namespace) -> None:
    import pandas as pd

    from . import reports
    from .endurance.laps import annotate

    out = ROOT / "reports" / "ml"
    out.mkdir(parents=True, exist_ok=True)
    df = annotate(pd.read_csv(ROOT / "data/n24/n24_2026_gt3_laps.csv", dtype={"car": str}))
    if args.task == "forecast":
        from .ml import forecasting as fc

        results = fc.backtest(fc.to_series(df), ag_time_limit=args.time_limit)
        results.to_csv(out / "forecast_backtest.csv.gz", index=False)
        scores = fc.score(results)
        scores.to_csv(out / "forecast_scores.csv", index=False)
        sig = [
            {
                "model": m,
                "reference": "last_clean_lap",
                "laps": name,
                **fc.paired_bootstrap(results[mask], m, "last_clean_lap"),
            }
            for m in ("chronos2", "autogluon")
            for name, mask in (
                ("all", results["clean"] | ~results["clean"]),
                ("clean", results["clean"]),
            )
        ]
        pd.DataFrame(sig).to_csv(out / "forecast_significance.csv", index=False)
        reports.ml_figure(
            scores,
            out / "forecast_scores.png",
            "Next-stint forecast, N24 2026 GT3 (rolling backtest)",
        )
    else:
        from .ml import tabular as tab

        scores, importance = tab.evaluate(df, time_limit=args.time_limit)
        scores.to_csv(out / "tabular_scores.csv", index=False)
        importance.to_csv(out / "tabular_importance.csv", index=False)
        reports.ml_figure(
            scores, out / "tabular_scores.png", "Next-lap model, trained h0-16, tested h16-24"
        )
    print(scores.round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    """Entry point of the ``mlab`` command."""
    parser = argparse.ArgumentParser(prog="mlab", description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("data", help="download and parse official timing documents")
    p.add_argument("source", choices=["n24", "rx"])
    p.add_argument("--year", type=int, default=2026)
    p.set_defaults(func=_data)
    p = sub.add_parser("report", help="build CSV tables and figures")
    p.add_argument("what", choices=["n24", "rx"])
    p.set_defaults(func=_report)
    p = sub.add_parser("ml", help="Chronos-2 / AutoGluon experiments (needs the [ml] extra)")
    p.add_argument("task", choices=["forecast", "tabular"])
    p.add_argument("--time-limit", type=int, default=120)
    p.set_defaults(func=_ml)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
