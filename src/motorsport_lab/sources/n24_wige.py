"""ADAC 24h Nürburgring official PDFs (timing by wige SOLUTIONS) -> common lap table.

Two public documents are combined:

* ``*_24h_Race_Lap_by_Lap.pdf``: every lap of every car, e.g. ``8:32.481 1(1)`` (lap 1,
  driver 1) or ``8:29.070 pit(1)`` (a lap that ended in the pit lane);
* ``*_24h_Race_Result_Class_Final.pdf``: class (``SP 9`` = GT3), entrant and car model.

Files are published at https://www.24h-rennen.de/ergebnisse/ .
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pdfplumber

from . import parse_laptime
from ._http import fetch

BASE_URL = "https://www.24h-rennen.de/wp-content/uploads/ergebnisse"
EDITIONS = {
    2026: (
        "2026-ADAC-RAVENOL-24h-Nuerburgring",
        "41_24h_Race_Lap_by_Lap.pdf",
        "40_24h_Race_Result_Class_Final.pdf",
    ),
}
TIME_RE = re.compile(r"^\d{1,2}(?::\d{2}){1,2}\.\d{3}$")
LAP_RE = re.compile(r"^(\d+|pit)(?:\((\d)\))?$")
DRIVER_RE = re.compile(r"^(.*?)\((\d)\)$")
CLASS_ROW_RE = re.compile(
    r"^(?:(\d+|DNF|DNS|DSQ|NC)\s+)?(\d{1,3})\s+"
    r"(SP ?\d+T?|SP-PRO|SP-X|AT ?\d|Cup ?\d|TCR|V ?\d|BMW|H ?\d|M ?\d)\s+"
    r"(.+?)\s+(\d{1,3})\s+(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3})"
)
MAKE_RE = re.compile(
    r"(Porsche|BMW|Mercedes|Audi|Ferrari|Lamborghini|McLaren|Aston Martin|Ford|Toyota|Hyundai|"
    r"Cupra|VW|Volkswagen|KTM|HWA|Corvette|Chevrolet|Lexus|Opel|Renault|Subaru|Honda|Nissan|"
    r"Mini|Lotus|Alfa|Peugeot|Seat|Dacia|Bentley|Glickenhaus|Toyo)"
)


def _split_entrant_model(text: str) -> tuple[str, str]:
    """``"E: Winward Racing, E6091-I Mercedes-AMG GT3"`` -> (entrant, model)."""
    text = text.removeprefix("E:").strip()
    entrant, _, tail = text.rpartition(", ") if ", " in text else ("", "", text)
    make = MAKE_RE.search(tail)
    model = tail[make.start() :] if make else tail
    return entrant.strip(), model.strip()


ENTRANT_MAX_X = 230.0  # entrant column ends before the driver column on the lap-by-lap pages


@dataclass(frozen=True)
class _Word:
    x0: float
    top: float
    text: str


def _lines(page: pdfplumber.page.Page, tolerance: float = 3.0) -> list[list[_Word]]:
    words = [_Word(w["x0"], w["top"], w["text"]) for w in page.extract_words()]
    words.sort(key=lambda w: (round(w.top), w.x0))
    lines: list[list[_Word]] = []
    for word in words:
        if lines and abs(lines[-1][0].top - word.top) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def _is_header(line: list[_Word]) -> bool:
    has_time = any(TIME_RE.match(w.text) for w in line)
    return line[0].text.isdigit() and line[0].x0 < 50 and not has_time and len(line) > 2


def parse_lap_by_lap(pdf: bytes | Path) -> pd.DataFrame:
    """Parse the lap-by-lap PDF into one row per lap (car, lap, lap_time_s, driver, in_pit)."""
    source = io.BytesIO(pdf) if isinstance(pdf, bytes) else pdf
    rows: list[dict[str, object]] = []
    drivers: dict[str, dict[int, str]] = {}
    entrants: dict[str, str] = {}
    car: str | None = None
    lap_no = 0
    last_driver = 1
    with pdfplumber.open(source) as document:
        for page in document.pages:
            for line in _lines(page):
                if _is_header(line):
                    car, lap_no, last_driver = line[0].text, 0, 1
                    entrants[car] = " ".join(w.text for w in line[1:] if w.x0 < ENTRANT_MAX_X)
                    driver_text = " ".join(w.text for w in line[1:] if w.x0 >= ENTRANT_MAX_X)
                    drivers[car] = {}
                    for chunk in driver_text.split(" / "):
                        match = DRIVER_RE.match(chunk.strip())
                        if match:
                            drivers[car][int(match.group(2))] = match.group(1).strip()
                    continue
                if car is None:
                    continue
                texts = [w.text for w in line]
                for i in range(len(texts) - 1):
                    lap_match = LAP_RE.match(texts[i + 1])
                    if not (TIME_RE.match(texts[i]) and lap_match):
                        continue
                    in_pit = lap_match.group(1) == "pit"
                    lap_no = lap_no + 1 if in_pit else int(lap_match.group(1))
                    if lap_match.group(2):
                        last_driver = int(lap_match.group(2))
                    rows.append(
                        {
                            "car": car,
                            "lap": lap_no,
                            "lap_time_s": parse_laptime(texts[i]),
                            "driver_idx": last_driver,
                            "in_pit": in_pit,
                        }
                    )
    laps = pd.DataFrame(rows)
    laps["driver"] = [
        drivers[c].get(d, "") for c, d in zip(laps["car"], laps["driver_idx"], strict=True)
    ]
    laps["team"] = laps["car"].map(entrants)
    return laps


def parse_class_results(pdf: bytes | Path) -> pd.DataFrame:
    """Parse the final classification by class: car number -> class, entrant, model, status."""
    source = io.BytesIO(pdf) if isinstance(pdf, bytes) else pdf
    rows = []
    with pdfplumber.open(source) as document:
        for page in document.pages:
            for text_line in (page.extract_text() or "").splitlines():
                match = CLASS_ROW_RE.match(text_line.strip())
                if match:
                    pos, car, cls, middle, laps, _ = match.groups()
                    entrant, model = _split_entrant_model(middle)
                    rows.append(
                        {
                            "car": car,
                            "cls": cls.strip(),
                            "team": entrant.strip(),
                            "car_model": model.strip(),
                            "laps_completed": int(laps),
                            "status": "classified" if (pos or "").isdigit() else (pos or "NC"),
                            "class_position": int(pos) if (pos or "").isdigit() else None,
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates("car")


def is_gt3(cls: str, car_model: str) -> bool:
    """SP 9 is the GT3 class at the Nürburgring; AT1 also runs GT3 cars (renewable fuel)."""
    model = car_model.upper()
    return cls.replace(" ", "") == "SP9" or ("GT3" in model and "CUP" not in model)


def load_edition(year: int, cache_dir: Path, gt3: bool = True) -> pd.DataFrame:
    """Download (cached) and parse one edition into the common lap schema."""
    folder, lap_file, class_file = EDITIONS[year]
    laps = parse_lap_by_lap(fetch(f"{BASE_URL}/{folder}/{lap_file}", cache_dir))
    entries = parse_class_results(fetch(f"{BASE_URL}/{folder}/{class_file}", cache_dir))
    laps = laps.drop(columns="team").merge(entries, on="car", how="left")
    laps["event"] = f"n24_{year}"
    laps["flag"] = "GF"
    if gt3:
        mask = [
            is_gt3(str(c), str(m))
            for c, m in zip(laps["cls"].fillna(""), laps["car_model"].fillna(""), strict=True)
        ]
        laps = laps[mask]
    return laps.sort_values(["car", "lap"]).reset_index(drop=True)
