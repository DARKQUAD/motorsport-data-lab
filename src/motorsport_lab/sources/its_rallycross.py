"""Championnat de France de Rallycross (FFSA) results from ITS Chrono -> lap table.

The public results site https://www.its-results.com/rallycrossfr (linked from
rallycrossfrance.com) is backed by an anonymous JSON API and serves one PDF per
classification. Each heat PDF gives, per driver, every lap time and the two split
times of the joker lap (``JKR1 / JKR2``), which is exactly what joker-strategy
analysis needs.
"""

from __future__ import annotations

import io
import re
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pdfplumber

from . import parse_laptime
from ._http import fetch, fetch_json

API = "https://api.its-results.com/v1"
CHAMPIONSHIP = "rallycrossfr"
DEFAULT_CATEGORIES = ("Supercar", "Super 1600")
NUM_RE = re.compile(r"^\d{1,2}(?::\d{2})?\.\d{3}$")
HEAT_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}$")


@dataclass(frozen=True)
class ResultDoc:
    """One classification PDF published on its-results.com."""

    event: str
    event_date: str
    category: str
    session: str
    name: str
    result_id: str


def _session_kind(session: str, doc: str) -> str | None:
    """Keep race classifications only: heats, semi-finals and finals."""
    s, d = session.lower(), doc.lower()
    if s.startswith("manche qualificative") and d == "classement général":
        return "heat"
    if s.startswith("demi") and d.startswith("classement demi"):
        return "semi"
    if s == "finale" and d == "classement finale":
        return "final"
    return None


def list_result_docs(
    season: str, categories: tuple[str, ...], cache_dir: Path | None = None
) -> Iterator[ResultDoc]:
    """Walk championship -> season -> events -> category folders -> sessions -> result PDFs."""
    cs = fetch_json(f"{API}/Cs/getChampionship/{CHAMPIONSHIP}", cache_dir)
    seasons = fetch_json(f"{API}/Cs/getSeasons/{cs['id']}", cache_dir)
    season_ids = [s["id"] for s in seasons if s["name"] == season]
    if not season_ids:
        return
    season_id = season_ids[0]
    for event in fetch_json(f"{API}/Cs/getEvents/{season_id}", cache_dir):
        for folder in event["folders"]:
            category = next((c for c in categories if folder["name"].endswith(c)), None)
            if category is None:
                continue
            for session in fetch_json(f"{API}/Results/getSessions/{folder['id']}", cache_dir):
                for doc in fetch_json(f"{API}/Results/getResults/{session['id']}", cache_dir):
                    if doc.get("pdfUploaded") and _session_kind(session["name"], doc["name"]):
                        yield ResultDoc(
                            event["name"].title(),
                            event["beginDate"],
                            category,
                            session["name"],
                            doc["name"],
                            doc["id"],
                        )


def download_pdf(doc: ResultDoc, cache_dir: Path | None = None) -> bytes:
    """Download one classification PDF as the public site does (anonymous: empty token)."""
    query = urllib.parse.urlencode({"result": doc.result_id, "name": f"{doc.result_id}.pdf"})
    return fetch(f"{API}/download-pdf?{query}&token=", cache_dir)


Word = dict[str, Any]


def _center(word: Word) -> float:
    return (float(word["x0"]) + float(word["x1"])) / 2


@dataclass(frozen=True)
class _Layout:
    """Column geometry read from the page header (the PDFs come in several page sizes)."""

    header_y: float
    car_no_x: float
    name_x: float
    nat_x: float
    tours_x: float
    temps_x: float
    ecart_x: float
    pts_x: float
    laps: tuple[float, ...]

    @property
    def lap_pitch(self) -> float:
        """Horizontal distance between two lap columns."""
        return float(np.median(np.diff(self.laps))) if len(self.laps) > 1 else 60.0


def _layout(words: list[Word]) -> _Layout | None:
    tours = [
        (w, n)
        for w, n in pairwise(words)
        if w["text"] == "Tour" and str(n["text"]).isdigit() and abs(w["top"] - n["top"]) < 2
    ]
    if not tours:
        return None
    header_y = float(tours[0][0]["top"])
    head = [w for w in words if abs(float(w["top"]) - header_y) < 12]

    def x_of(pred: Callable[[str], bool], default: float) -> float:
        return next((float(w["x0"]) for w in head if pred(str(w["text"]))), default)

    name_x = x_of(lambda t: t == "Pilote", 105.0)
    return _Layout(
        header_y=header_y,
        car_no_x=x_of(
            lambda t: t.startswith("N") and len(t) <= 3 and t not in ("Nb.",), name_x - 20
        ),
        name_x=name_x,
        nat_x=x_of(lambda t: t == "Nat.", name_x + 120),
        tours_x=x_of(lambda t: t == "Tours", name_x + 160),
        temps_x=x_of(lambda t: t == "Temps", name_x + 200),
        ecart_x=x_of(lambda t: t == "Ecart", name_x + 290),
        pts_x=x_of(lambda t: t == "Pts.", 10_000.0),
        laps=tuple(sorted((float(w["x0"]) + float(n["x1"])) / 2 for w, n in tours)),
    )


def parse_heat_pdf(pdf: bytes | Path) -> pd.DataFrame:
    """Parse one ITS classification PDF into one row per driver and lap."""
    source = io.BytesIO(pdf) if isinstance(pdf, bytes) else pdf
    rows: list[dict[str, object]] = []
    with pdfplumber.open(source) as document:
        for page in document.pages:
            words: list[Word] = page.extract_words()
            lay = _layout(words)
            if lay is None:
                continue
            anchors = [
                w
                for w in words
                if lay.car_no_x - 8 <= _center(w) < lay.name_x - 1
                and str(w["text"]).isdigit()
                and float(w["top"]) > lay.header_y + 15
            ]
            tops = sorted(float(a["top"]) for a in anchors)
            pitch = float(np.median(np.diff(tops))) if len(tops) > 1 else 30.0
            for anchor in anchors:
                y = float(anchor["top"])
                block = [w for w in words if y - 0.38 * pitch <= float(w["top"]) <= y + 0.6 * pitch]
                split_y = y + 0.12 * pitch
                top_line = [w for w in block if float(w["top"]) < split_y]
                car_line = [w for w in block if float(w["top"]) >= split_y]

                def in_cols(line: list[Word], lo: float, hi: float) -> list[str]:
                    return [str(w["text"]) for w in line if lo <= _center(w) < hi]

                name = " ".join(in_cols(top_line, lay.name_x - 2, lay.nat_x - 3))
                car = " ".join(in_cols(car_line, lay.name_x - 2, lay.nat_x - 3))
                pos = next(iter(in_cols(top_line, 0, lay.car_no_x - 3)), "")
                nat = next(iter(in_cols(top_line, lay.nat_x - 3, lay.tours_x - 2)), "")
                heat = next(
                    (str(w["text"]) for w in top_line if HEAT_TIME_RE.match(str(w["text"]))), ""
                )
                gap = next(
                    iter(in_cols(top_line, lay.ecart_x - 12, lay.laps[0] - lay.lap_pitch / 2)), ""
                )
                pts = next(iter(in_cols(top_line, lay.pts_x - 6, lay.pts_x + 30)), "")
                penalty = " ".join(
                    t for t in in_cols(car_line, lay.temps_x - 5, lay.ecart_x - 12) if t != "-"
                )
                laps: dict[int, float] = {}
                jokers: dict[int, list[float]] = {}
                for w in block:
                    if not NUM_RE.match(str(w["text"])):
                        continue
                    distances = [abs(x - _center(w)) for x in lay.laps]
                    k = distances.index(min(distances))
                    if distances[k] > 0.5 * lay.lap_pitch:
                        continue
                    if float(w["top"]) < split_y:
                        laps[k + 1] = parse_laptime(w["text"])
                    else:
                        jokers.setdefault(k + 1, []).append(parse_laptime(w["text"]))
                tours = lay.laps
                for lap in range(1, len(tours) + 1):
                    jkr = jokers.get(lap, [])
                    rows.append(
                        {
                            "position": int(pos) if pos.isdigit() else None,
                            "car_no": str(anchor["text"]),
                            "driver": name,
                            "car_model": car,
                            "nat": nat,
                            "heat_time_s": parse_laptime(heat),
                            "gap": gap,
                            "points": int(pts) if pts.isdigit() else None,
                            "penalty": penalty,
                            "lap": lap,
                            "lap_time_s": laps.get(lap, float("nan")),
                            "is_joker": len(jkr) == 2 and all(v > 0 for v in jkr),
                            "jkr1_s": jkr[0] if len(jkr) == 2 else None,
                            "jkr2_s": jkr[1] if len(jkr) == 2 else None,
                        }
                    )
    return pd.DataFrame(rows)


def load_season(
    season: str = "2026",
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download (cached) and parse every heat, semi-final and final of a season."""
    frames = []
    for doc in list_result_docs(season, categories, cache_dir):
        table = parse_heat_pdf(download_pdf(doc, cache_dir))
        if table.empty:
            continue
        table.insert(0, "race", f"{doc.session} - {doc.name}")
        table.insert(0, "kind", _session_kind(doc.session, doc.name))
        table.insert(0, "category", doc.category)
        table.insert(0, "date", doc.event_date)
        table.insert(0, "event", doc.event)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # a "RECTIFICATIF" (corrected) classification replaces the original one
    keys = [
        (str(e), str(c), str(r))
        for e, c, r in zip(df["event"], df["category"], df["race"], strict=True)
    ]
    unique = set(keys)
    superseded = {
        k
        for k in unique
        if any(o[:2] == k[:2] and o[2] != k[2] and o[2].startswith(k[2]) for o in unique)
    }
    return df[[k not in superseded for k in keys]].reset_index(drop=True)
