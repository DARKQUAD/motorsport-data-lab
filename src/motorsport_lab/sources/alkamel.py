"""Al Kamel Systems "Analysis" CSV (FIA WEC / 24h of Le Mans) -> common lap table.

The official timing site (fiawec.alkamelsystems.com) lists one ``23_Analysis_Race*.CSV``
per session but no longer serves it at a public URL, so download it from the site yourself
and pass the file here. Files are semicolon separated with padded headers (``" LAP_TIME"``).
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from . import parse_laptime

REQUIRED = ("NUMBER", "LAP_NUMBER", "LAP_TIME")
GT3_CLASSES = ("LMGT3", "GT3", "LMGTE", "GTE", "GTD", "GTD PRO")


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def read_analysis_csv(source: str | Path | bytes, event: str = "le_mans") -> pd.DataFrame:
    """Load an Al Kamel Analysis CSV into the common one-row-per-lap schema."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    df = pd.read_csv(io.StringIO(_decode(raw)), sep=";", dtype=str, keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Not an Al Kamel Analysis file, missing columns: {missing}")
    df = df.apply(lambda s: s.str.strip())

    def col(name: str, default: str = "") -> pd.Series:
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    out = pd.DataFrame(
        {
            "event": event,
            "car": col("NUMBER"),
            "cls": col("CLASS").str.upper(),
            "car_model": col("MANUFACTURER"),
            "team": col("TEAM"),
            "lap": pd.to_numeric(col("LAP_NUMBER"), errors="coerce"),
            "lap_time_s": col("LAP_TIME").map(parse_laptime),
            "driver_idx": pd.to_numeric(col("DRIVER_NUMBER", "1"), errors="coerce"),
            "driver": col("DRIVER_NAME"),
            "in_pit": col("CROSSING_FINISH_LINE_IN_PIT").eq("B"),
            "flag": col("FLAG_AT_FL", "GF").replace("", "GF").str.upper(),
        }
    )
    out = out.dropna(subset=["lap", "lap_time_s"])
    out["lap"] = out["lap"].astype(int)
    out["driver_idx"] = out["driver_idx"].fillna(1).astype(int)
    return out.sort_values(["car", "lap"]).reset_index(drop=True)


def gt3_only(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep GT3-type classes (LMGT3 at Le Mans since 2024)."""
    return laps[laps["cls"].isin(GT3_CLASSES)].reset_index(drop=True)
