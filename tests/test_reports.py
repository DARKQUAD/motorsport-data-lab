from pathlib import Path

import pandas as pd

from motorsport_lab import reports

DATA = Path(__file__).resolve().parents[1] / "data"


def test_n24_report_writes_tables_and_figures(tmp_path: Path) -> None:
    tables = reports.n24_report(DATA / "n24/n24_2026_gt3_laps.csv", tmp_path)
    assert {"pit_stops", "strategy", "car_model_pace", "stint_evolution"} <= set(tables)
    assert (tmp_path / "track_index.png").stat().st_size > 10_000
    assert len(tables["strategy"]) == 5
    assert tables["pit_stops"]["loss_s"].median() > 100  # a Nürburgring stop costs minutes


def test_rx_report_writes_heatmap(tmp_path: Path) -> None:
    tables = reports.rx_report(DATA / "rx_france/rx_france_2026_laps.csv", tmp_path)
    heat = tables["joker_heatmap"]
    assert heat["mean_position"].between(1, heat["grid_slot"].max()).all()
    assert (tmp_path / "joker_heatmap.png").exists()
    tracks = pd.read_csv(tmp_path / "tracks.csv")
    assert tracks["joker_cost_s"].between(1, 8).all()


def test_ml_figure(tmp_path: Path) -> None:
    scores = pd.DataFrame({"model": ["last_clean_lap", "chronos2"], "mae_all_s": [46.0, 43.0]})
    reports.ml_figure(scores, tmp_path / "s.png", "t")
    assert (tmp_path / "s.png").exists()
