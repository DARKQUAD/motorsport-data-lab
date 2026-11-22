"""Motorsport Data Lab — bilingual Streamlit app.  Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from motorsport_lab.endurance import laps as lp
from motorsport_lab.endurance import lineup
from motorsport_lab.endurance import strategy as strat
from motorsport_lab.i18n import t
from motorsport_lab.rallycross import analysis as rxa
from motorsport_lab.rallycross import simulator as rxs
from motorsport_lab.sources.alkamel import gt3_only, read_analysis_csv

ROOT = Path(__file__).resolve().parents[1]
N24 = ROOT / "data/n24/n24_2026_gt3_laps.csv"
RX = ROOT / "data/rx_france/rx_france_2026_laps.csv"
ML = ROOT / "reports/ml"

st.set_page_config(page_title="Motorsport Data Lab", page_icon="🏁", layout="wide")
lang = (
    "fr" if st.sidebar.radio("Language / Langue", ["Français", "English"]) == "Français" else "en"
)
pages = ["home", "n24", "lemans", "rx", "ml"]
page = st.sidebar.radio(t("nav", lang), pages, format_func=lambda k: t(k, lang), key="page")
st.sidebar.caption(t("sources", lang))


@st.cache_data
def n24_laps() -> pd.DataFrame:
    return lp.annotate(pd.read_csv(N24, dtype={"car": str}))


@st.cache_data
def rx_races() -> pd.DataFrame:
    return rxa.per_race(pd.read_csv(RX, dtype={"car_no": str}))


def endurance_dashboard(df: pd.DataFrame, calibrated: bool) -> None:
    bucket = (
        df.groupby("bucket")
        .agg(hour=("race_hour", "median"), index=("track_index", "first"))
        .reset_index()
    )
    st.subheader(t("track_index_title", lang))
    fig = px.line(bucket, x="hour", y="index", labels={"hour": "h", "index": ""})
    fig.add_hline(y=lp.NEUTRAL_THRESHOLD, line_dash="dash", line_color="#d64545")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(t("track_index_help", lang))

    order = df.groupby("car")["lap"].max().sort_values(ascending=False).index.tolist()
    labels = {c: f"#{c} {df[df.car == c]['car_model'].iat[0]}" for c in order}
    cars = st.multiselect(
        t("car_pick", lang), order, default=order[:3], format_func=lambda c: labels[c]
    )
    st.subheader(t("laps_title", lang))
    sel = df[df["car"].isin(cars)].assign(lap_min=lambda d: d["lap_time_s"].clip(upper=900) / 60)
    fig = px.scatter(
        sel,
        x="race_hour",
        y="lap_min",
        color="car",
        symbol="clean",
        hover_data=["lap", "driver", "in_pit", "out_lap"],
        labels={"race_hour": "h", "lap_min": "min"},
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("pit_title", lang))
        stops = lp.pit_stops(df)
        st.plotly_chart(
            px.histogram(stops[stops["loss_s"] < 600], x="loss_s", color="driver_change", nbins=40),
            use_container_width=True,
        )
    with c2:
        st.subheader(t("bop_title", lang))
        pace = lp.car_model_pace(df)
        pace["gap_s"] = pace["top_pace_s"] - pace["top_pace_s"].min()
        st.plotly_chart(
            px.bar(pace, x="gap_s", y="car_model", orientation="h"), use_container_width=True
        )

    st.subheader(t("evolution_title", lang))
    evo = lp.stint_evolution(df)
    if not evo.empty:
        fig = px.scatter(
            evo,
            x="slope_s_per_lap",
            y="car_model",
            error_x=evo["ci_high"] - evo["slope_s_per_lap"],
            error_x_minus=evo["slope_s_per_lap"] - evo["ci_low"],
        )
        fig.add_vline(x=0, line_color="#7b8794")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(t("evolution_help", lang))

    st.subheader(t("drivers_title", lang))
    st.dataframe(
        lp.driver_deltas(df).sort_values(["car", "delta_s"]).round(2),
        use_container_width=True,
        height=280,
    )

    if calibrated:
        st.subheader(t("strategy_title", lang))
        st.caption(t("strategy_help", lang))
        base = strat.calibrate(df)
        c1, c2, c3, c4 = st.columns(4)
        params = replace(
            base,
            tyre_deg_s_per_lap=c1.slider(t("deg", lang), 0.0, 0.5, base.tyre_deg_s_per_lap, 0.01),
            tyre_change_extra_s=c2.slider(
                t("tyre_extra", lang), 0.0, 120.0, float(round(base.tyre_change_extra_s)), 5.0
            ),
            neutral_share=c3.slider(
                t("neutral_share", lang), 0.0, 0.7, float(round(base.neutral_share, 2)), 0.05
            ),
        )
        n = c4.select_slider(t("sims", lang), [250, 500, 1000, 2000], 500)
        st.dataframe(
            strat.compare(params, strat.default_strategies(params), n=n).round(2),
            use_container_width=True,
        )


if page == "home":
    st.title(t("title", lang))
    st.markdown(t("subtitle", lang))
    st.markdown(f"### {t('glossary', lang)}")
    st.markdown(t("glossary_body", lang))

elif page == "n24":
    st.title(t("n24", lang))
    endurance_dashboard(n24_laps(), calibrated=True)

elif page == "lemans":
    st.title(t("lemans", lang))
    st.caption(t("upload_help", lang))
    file = st.file_uploader(t("upload", lang), type=["csv", "CSV"])
    if file is not None:
        laps = gt3_only(read_analysis_csv(file.getvalue()))
        if laps.empty:
            st.warning("No GT3 class found / Aucune classe GT3 trouvée.")
        else:
            endurance_dashboard(lp.annotate(laps), calibrated=False)

elif page == "rx":
    st.title(t("rx", lang))
    races = rx_races()
    category = st.radio(t("category", lang), sorted(races["category"].unique()), horizontal=True)
    cat = races[races["category"] == category]
    tracks = rxa.track_summary(cat)
    st.subheader(t("joker_cost_title", lang))
    st.plotly_chart(
        px.bar(tracks, x="event", y=["joker_cost_s", "lap1_cost_s"], barmode="group"),
        use_container_width=True,
    )
    st.subheader(t("joker_when_title", lang))
    timing = rxa.joker_timing(cat).melt(
        id_vars=["category", "kind"], var_name="when", value_name="share"
    )
    st.plotly_chart(px.bar(timing, x="kind", y="share", color="when"), use_container_width=True)

    st.subheader(t("heatmap_title", lang))
    event = st.selectbox(t("event", lang), tracks["event"].tolist())
    finals = cat[cat["kind"] == "final"]
    n_laps = int(finals["laps"].mode().iat[0]) if not finals.empty else 6
    params = rxs.calibrate(races, event, category, laps=n_laps, n_cars=8)
    params = replace(
        params, pass_advantage_s=st.slider(t("pass", lang), 0.2, 2.0, params.pass_advantage_s, 0.1)
    )
    policy_row = rxa.joker_timing(cat)
    policy_row = policy_row[policy_row["kind"] == "final"]
    policy = {
        k: float(policy_row[k].mean()) for k in ("lap 1", "middle", "last lap") if k in policy_row
    }
    heat = rxs.joker_heatmap(params, policy, n=1500)
    pivot = heat.pivot(index="grid_slot", columns="joker_lap", values="mean_position")
    st.plotly_chart(
        px.imshow(pivot, text_auto=".2f", color_continuous_scale="RdYlGn_r", aspect="auto"),
        use_container_width=True,
    )
    st.caption(t("heatmap_help", lang))

elif page == "ml":
    st.title(t("ml", lang))
    st.markdown(t("ml_intro", lang))
    if not (ML / "forecast_scores.csv").exists():
        st.info(t("no_ml", lang))
    sections = {
        "ml_forecast": ("forecast_scores", "forecast_significance"),
        "ml_next_lap": (
            "classic_scores",
            "tabular_scores",
            "classic_shap",
            "conformal_intervals",
        ),
        "ml_drivers": ("driver_ratings_summary", "driver_ratings"),
        "ml_rx": (
            "rx_win_odds_ratios",
            "rx_win_cv",
            "rx_win_holdout",
            "rx_win_odds_ratios_holdout",
        ),
        "ml_seasons": (
            "seasons_next_lap",
            "seasons_forecast",
            "seasons_conformal",
            "seasons_rating_stability",
        ),
    }
    for header, names in sections.items():
        st.header(t(header, lang))
        st.markdown(t(header + "_help", lang))
        for name in names:
            csv = ML / f"{name}.csv"
            if csv.exists():
                st.dataframe(pd.read_csv(csv).round(3), use_container_width=True)
            png = ML / f"{name}.png"
            if png.exists():
                st.image(str(png))

    st.header(t("ml_lineup", lang))
    st.markdown(t("ml_lineup_help", lang))
    for name in ("style_stability", "style_within_race"):
        csv = ML / f"{name}.csv"
        if csv.exists():
            st.dataframe(pd.read_csv(csv).round(3), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    n_drivers = c1.slider(t("lineup_drivers", lang), 2, 4, 3)
    min_stints = c2.slider(t("lineup_min", lang), 1, 19 // n_drivers, min(4, 19 // n_drivers))
    prior_corr = c3.slider(t("lineup_corr", lang), 0.0, 1.0, 0.2, 0.05)

    @st.cache_data
    def _lineup(drivers: int, minimum: int, corr: float) -> pd.DataFrame:
        params = lineup.LineupParams(drivers=drivers, min_stints=minimum, prior_corr=corr)
        return lineup.compare(params, n=300)

    st.dataframe(
        _lineup(n_drivers, min_stints, float(prior_corr)).round(2), use_container_width=True
    )
    png = ML / "lineup_value_of_history.png"
    if png.exists():
        st.image(str(png))
