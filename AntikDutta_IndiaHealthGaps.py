# pip install streamlit pandas numpy plotly openpyxl

"""
NFHS-5 (2019-21) State Health Dashboard
Compares Indian states/UTs on maternal health, vaccination and child nutrition.
Data: NFHS5_clean.xlsx (preferred) → NFHS5_values.csv → st.file_uploader
"""

from __future__ import annotations

import io
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION CONSTANTS  (no magic numbers elsewhere)
# ═══════════════════════════════════════════════════════════════════

BASE_DIR = pathlib.Path(__file__).parent

# ---------- Index specification ----------
INDEX_SPEC: dict[str, list[tuple[str, str]]] = {
    "Maternal Health": [
        ("anc_first_trimester_pct",         "higher"),
        ("anc_4plus_visits_pct",            "higher"),
        ("tetanus_protected_birth_pct",     "higher"),
        ("ifa_100days_pct",                 "higher"),
        ("ifa_180days_pct",                 "higher"),
        ("mcp_card_pct",                    "higher"),
        ("postnatal_care_mother_2days_pct", "higher"),
        ("institutional_births_pct",        "higher"),
        ("skilled_birth_attendance_pct",    "higher"),
        ("women_anaemic_pregnant_pct",      "lower"),
        ("women_bmi_below_normal_pct",      "lower"),
    ],
    "Vaccination": [
        ("full_immunisation_card_recall_pct", "higher"),
        ("bcg_pct",                           "higher"),
        ("polio3_pct",                        "higher"),
        ("penta_dpt3_pct",                    "higher"),
        ("mcv1_pct",                          "higher"),
        ("mcv2_24to35m_pct",                  "higher"),
        ("rotavirus3_pct",                    "higher"),
        ("vitamin_a_pct",                     "higher"),
    ],
    "Child Nutrition": [
        ("breastfed_1hour_pct",             "higher"),
        ("exclusive_breastfed_u6m_pct",     "higher"),
        ("adequate_diet_6to23m_pct",        "higher"),
        ("stunted_pct",                     "lower"),
        ("wasted_pct",                      "lower"),
        ("severely_wasted_pct",             "lower"),
        ("underweight_pct",                 "lower"),
        ("children_anaemic_6to59m_pct",     "lower"),
    ],
}

# ---------- Column groups ----------
NEUTRAL_COLS: list[str] = [
    "caesarean_overall_pct",
    "caesarean_private_pct",
    "caesarean_public_pct",
    "institutional_births_public_pct",
    "total_fertility_rate",
    "overweight_child_pct",
]

OUTCOME_COLS: list[str] = [
    "neonatal_mortality_rate",
    "infant_mortality_rate",
    "under5_mortality_rate",
]

CONTEXT_COLS: list[str] = [
    "women_literate_pct",
    "women_10plus_yrs_school_pct",
    "improved_sanitation_pct",
    "clean_cooking_fuel_pct",
    "improved_drinking_water_pct",
    "health_insurance_pct",
    "women_married_before18_pct",
    "oop_cost_public_delivery_rs",
    "women_anaemic_all_pct",
]

SMALL_UTS_DEFAULT: list[str] = [
    "Lakshadweep",
    "Ladakh",
    "Chandigarh",
    "Andaman & Nicobar Islands",
]

# ---------- KPI spec ----------
# key: (label, unit, direction, domain, is_headline)
KPI_SPEC: dict[str, tuple[str, str, str, str]] = {
    "anc_4plus_visits_pct":              ("ANC 4+ Visits",          "%",              "higher", "Maternal"),
    "institutional_births_pct":          ("Institutional Births",    "%",              "higher", "Maternal"),
    "postnatal_care_mother_2days_pct":   ("PNC Mother (2 days)",     "%",              "higher", "Maternal"),
    "women_anaemic_pregnant_pct":        ("Anaemia – Pregnant Women","%",              "lower",  "Maternal"),
    "full_immunisation_card_recall_pct": ("Full Immunisation",       "%",              "higher", "Vaccination"),
    "mcv2_24to35m_pct":                  ("MCV2 (24–35 m)",          "%",              "higher", "Vaccination"),
    "rotavirus3_pct":                    ("Rotavirus 3",             "%",              "higher", "Vaccination"),
    "stunted_pct":                       ("Stunting",                "%",              "lower",  "Nutrition"),
    "wasted_pct":                        ("Wasting",                 "%",              "lower",  "Nutrition"),
    "adequate_diet_6to23m_pct":          ("Adequate Diet (6-23m)",   "%",              "higher", "Nutrition"),
    "children_anaemic_6to59m_pct":       ("Anaemia – Children",      "%",              "lower",  "Nutrition"),
    "neonatal_mortality_rate":           ("Neonatal Mortality",      "/1k live births","lower",  "Outcomes"),
    "infant_mortality_rate":             ("Infant Mortality",        "/1k live births","lower",  "Outcomes"),
    "under5_mortality_rate":             ("Under-5 Mortality",       "/1k live births","lower",  "Outcomes"),
}

# ---------- SDG Benchmarks (empty = not set) ----------
BENCHMARKS: dict[str, Optional[float]] = {
    "neonatal_mortality_rate": 12.0,
    "under5_mortality_rate":   25.0,
}
BENCHMARK_LABEL = "SDG 3.2 target"

# ---------- Cascade definitions ----------
MATERNAL_CASCADE_COLS   = ["anc_first_trimester_pct", "anc_4plus_visits_pct",
                            "institutional_births_pct", "postnatal_care_mother_2days_pct"]
MATERNAL_CASCADE_LABELS = ["ANC 1st trimester", "ANC 4+ visits",
                            "Institutional birth", "PNC (mother, 2 days)"]
IMMUNISATION_CASCADE_COLS   = ["bcg_pct", "penta_dpt3_pct", "mcv1_pct", "mcv2_24to35m_pct"]
IMMUNISATION_CASCADE_LABELS = ["BCG", "Penta/DPT 3", "MCV1", "MCV2 (24–35 m)"]

# ---------- Overview KPIs ----------
OVERVIEW_KPI_COLS: list[str] = [
    "stunted_pct",
    "full_immunisation_card_recall_pct",
    "institutional_births_pct",
    "anc_4plus_visits_pct",
    "infant_mortality_rate",
]

# ---------- Plotly theme ----------
PLOTLY_TEMPLATE = "plotly_white"
SCORE_COLORSCALE = "RdYlGn"

# ---------- Domain score column name helpers ----------
DOMAIN_SCORE_COLS: dict[str, str] = {
    d: f"{d.lower().replace(' ', '_')}_score" for d in INDEX_SPEC
}

# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — TILE-GRID MAP COORDINATES
# One square per state/UT, no GeoJSON required.
# ═══════════════════════════════════════════════════════════════════

TILE_GRID: dict[str, tuple[int, int]] = {
    "Jammu & Kashmir":                          (1, 0),
    "Ladakh":                                   (2, 0),
    "Himachal Pradesh":                         (1, 1),
    "Punjab":                                   (0, 1),
    "Chandigarh":                               (0, 2),
    "Haryana":                                  (1, 2),
    "NCT of Delhi":                             (2, 2),
    "Uttarakhand":                              (3, 1),
    "Arunachal Pradesh":                        (8, 1),
    "Nagaland":                                 (8, 2),
    "Manipur":                                  (8, 3),
    "Mizoram":                                  (8, 4),
    "Tripura":                                  (7, 4),
    "Meghalaya":                                (7, 3),
    "Assam":                                    (7, 2),
    "Sikkim":                                   (7, 1),
    "Rajasthan":                                (0, 3),
    "Gujarat":                                  (0, 5),
    "Dadra and Nagar Haveli & Daman and Diu":   (0, 6),
    "Uttar Pradesh":                            (2, 3),
    "Bihar":                                    (4, 3),
    "Jharkhand":                                (4, 4),
    "West Bengal":                              (5, 4),
    "Madhya Pradesh":                           (2, 4),
    "Chhattisgarh":                             (3, 4),
    "Odisha":                                   (4, 5),
    "Maharashtra":                              (1, 5),
    "Telangana":                                (2, 6),
    "Andhra Pradesh":                           (3, 6),
    "Karnataka":                                (1, 6),
    "Goa":                                      (0, 7),
    "Tamil Nadu":                               (2, 7),
    "Kerala":                                   (1, 7),
    "Andaman & Nicobar Islands":                (6, 7),
    "Lakshadweep":                              (0, 8),
    "Puducherry":                               (3, 7),
}


def _assert_tile_grid() -> None:
    expected = {
        "Andaman & Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh",
        "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
        "Dadra and Nagar Haveli & Daman and Diu", "Goa", "Gujarat", "Haryana",
        "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka",
        "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra",
        "Manipur", "Meghalaya", "Mizoram", "NCT of Delhi", "Nagaland",
        "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
        "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
        "Uttarakhand", "West Bengal",
    }
    missing = expected - set(TILE_GRID)
    extra   = set(TILE_GRID) - expected
    assert not missing, f"Tile grid missing states: {missing}"
    assert not extra,   f"Tile grid has unexpected entries: {extra}"
    coords = list(TILE_GRID.values())
    assert len(coords) == len(set(coords)), "Duplicate coordinates in TILE_GRID"


_assert_tile_grid()

# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(
    uploaded_bytes: Optional[bytes] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (values_df, flags_df, dictionary_df). Cached."""
    xlsx_path = BASE_DIR / "NFHS5_clean.xlsx"
    csv_path  = BASE_DIR / "NFHS5_values.csv"

    if uploaded_bytes is not None:
        xf     = pd.ExcelFile(io.BytesIO(uploaded_bytes))
        values = pd.read_excel(xf, sheet_name="Values")
        flags  = pd.read_excel(xf, sheet_name="Low_sample_flags")
        dct    = pd.read_excel(xf, sheet_name="Dictionary")
    elif xlsx_path.exists():
        xf     = pd.ExcelFile(str(xlsx_path))
        values = pd.read_excel(xf, sheet_name="Values")
        flags  = pd.read_excel(xf, sheet_name="Low_sample_flags")
        dct    = pd.read_excel(xf, sheet_name="Dictionary")
    elif csv_path.exists():
        values = pd.read_csv(str(csv_path), encoding="utf-8-sig")
        flags  = pd.DataFrame(0, index=values.index, columns=values.columns)
        dct    = pd.DataFrame(
            columns=["short_name", "category", "original_column",
                     "suppressed_count", "low_sample_count"]
        )
    else:
        raise FileNotFoundError("No data file found.")

    for col in ("state_ut", "area", "geography_type"):
        for df in (values, flags):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    return values, flags, dct


def validate_data(df: pd.DataFrame) -> None:
    if len(df) != 111:
        st.warning(f"Expected 111 rows, found {len(df)}.")
    if df["state_ut"].nunique() != 37:
        st.warning(f"Expected 37 unique geographies, found {df['state_ut'].nunique()}.")


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — SCORING (pure functions, unit-testable)
# ═══════════════════════════════════════════════════════════════════

def score_minmax(series: pd.Series, direction: str) -> pd.Series:
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(50.0, index=series.index)
    scaled = (series - lo) / (hi - lo) * 100.0
    return (100.0 - scaled) if direction == "lower" else scaled


def score_zscore(series: pd.Series, direction: str) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    if pd.isna(mu) or pd.isna(sigma) or sigma == 0:
        return pd.Series(50.0, index=series.index)
    z = (series - mu) / sigma
    if direction == "lower":
        z = -z
    lo, hi = z.min(), z.max()
    if hi == lo:
        return pd.Series(50.0, index=series.index)
    return (z - lo) / (hi - lo) * 100.0


def score_percentile(series: pd.Series, direction: str) -> pd.Series:
    ranked = series.rank(pct=True, na_option="keep") * 100.0
    return (100.0 - ranked) if direction == "lower" else ranked


SCORERS: dict[str, object] = {
    "min-max":    score_minmax,
    "z-score":    score_zscore,
    "percentile": score_percentile,
}


def compute_domain_scores(
    df: pd.DataFrame,
    method: str = "min-max",
    min_present_frac: float = 0.75,
) -> pd.DataFrame:
    """Add domain score columns, composite_score, and tier to df."""
    scorer = SCORERS[method]
    out = df.copy()

    for domain, indicators in INDEX_SPEC.items():
        score_tmp: list[str] = []
        for col, direction in indicators:
            if col not in df.columns:
                continue
            valid = df[col].dropna()
            if len(valid) < 2:
                continue
            sc_col = f"__sc_{col}"
            out[sc_col] = scorer(df[col], direction)
            score_tmp.append(sc_col)

        dcol = DOMAIN_SCORE_COLS[domain]
        if not score_tmp:
            out[dcol] = np.nan
        else:
            n_total   = sum(1 for c, _ in indicators if c in df.columns)
            threshold = max(1, int(n_total * min_present_frac))
            n_present = out[score_tmp].notna().sum(axis=1)
            row_mean  = out[score_tmp].mean(axis=1)
            out[dcol] = np.where(n_present >= threshold, row_mean, np.nan)
            out.drop(columns=score_tmp, inplace=True)

    d_cols = list(DOMAIN_SCORE_COLS.values())
    all_ok = out[d_cols].notna().all(axis=1)
    out["composite_score"] = np.where(all_ok, out[d_cols].mean(axis=1), np.nan)
    out["tier"] = pd.cut(
        out["composite_score"],
        bins=[-0.001, 25.0, 50.0, 75.0, 100.001],
        labels=["Bottom", "Lower-mid", "Upper-mid", "Top"],
    )
    return out


TIER_COLORS: dict[str, str] = {
    "Top":        "#1a9641",
    "Upper-mid":  "#a6d96a",
    "Lower-mid":  "#fdae61",
    "Bottom":     "#d7191c",
    "":           "#cccccc",
}

# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════

def pretty(col: str) -> str:
    """snake_case → readable label; uses KPI_SPEC if available."""
    if col in KPI_SPEC:
        return KPI_SPEC[col][0]
    return (col.replace("_pct", " (%)")
              .replace("_rate", " rate")
              .replace("_rs", " (Rs)")
              .replace("_", " ")
              .title())


def col_direction(col: str) -> str:
    for indicators in INDEX_SPEC.values():
        for c, d in indicators:
            if c == col:
                return d
    if col in KPI_SPEC:
        return KPI_SPEC[col][2]
    return "higher"


def all_index_cols() -> list[str]:
    seen: set[str] = set()
    cols: list[str] = []
    for indicators in INDEX_SPEC.values():
        for c, _ in indicators:
            if c not in seen:
                seen.add(c)
                cols.append(c)
    return cols


def filter_states(
    df: pd.DataFrame,
    area: str,
    exclude_small: bool,
    small_uts: list[str],
) -> pd.DataFrame:
    mask = (df["geography_type"] == "State/UT") & (df["area"] == area)
    sub  = df[mask].copy()
    if exclude_small:
        sub = sub[~sub["state_ut"].isin(small_uts)]
    return sub.reset_index(drop=True)


def india_row(df: pd.DataFrame, area: str) -> pd.Series:
    mask = (df["geography_type"] == "National") & (df["area"] == area)
    rows = df[mask]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    combined = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(combined) < 4:
        return float("nan")
    return float(combined["x"].corr(combined["y"], method="spearman"))


def fmt_val(v: float | None, unit: str, low_sample: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    s = f"{v:.1f}{unit}" if "%" in unit else f"{v:.1f}"
    return s + ("†" if low_sample else "")


def delta_color(delta: float, direction: str) -> str:
    """Green if better, red if worse."""
    if np.isnan(delta):
        return "normal"
    is_better = (delta > 0 and direction == "higher") or (delta < 0 and direction == "lower")
    return "normal" if delta == 0 else ("inverse" if is_better else "off")


def fill_urban_rural_with_total(
    area_df: pd.DataFrame,
    all_values: pd.DataFrame,
    area: str,
) -> pd.DataFrame:
    """Return copy of area_df with blank numeric cells filled from Total rows."""
    if area == "Total":
        return area_df
    total_lookup = (
        all_values[all_values["area"] == "Total"]
        .set_index("state_ut")
    )
    out = area_df.copy()
    num_cols = out.select_dtypes(include="number").columns.tolist()
    for idx, row in out.iterrows():
        state = row["state_ut"]
        if state not in total_lookup.index:
            continue
        for col in num_cols:
            if pd.isna(row[col]) and col in total_lookup.columns:
                tv = total_lookup.loc[state, col]
                if pd.notna(tv):
                    out.at[idx, col] = tv
    return out


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — TILE-GRID MAP BUILDER
# ═══════════════════════════════════════════════════════════════════

def _rdylgn(t: float) -> str:
    """Interpolate RdYlGn: t=0 → red, t=0.5 → yellow, t=1 → green."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        u = t * 2.0
        r = int(215 + (253 - 215) * u)
        g = int(25  + (174 - 25)  * u)
        b = int(28  + (97  - 28)  * u)
    else:
        u = (t - 0.5) * 2.0
        r = int(253 + (26  - 253) * u)
        g = int(174 + (150 - 174) * u)
        b = int(97  + (65  - 97)  * u)
    return f"rgb({r},{g},{b})"


def build_tile_map(
    scored_df: pd.DataFrame,
    color_col: str,
    flags_df: pd.DataFrame,
    title: str,
    reverse_scale: bool = False,
) -> go.Figure:
    """Plotly tile-grid map. Grey = suppressed, † = low-sample."""

    # Build per-tile data
    rows = []
    for state, (cx, ry) in TILE_GRID.items():
        sdf = scored_df[scored_df["state_ut"] == state]
        val = float("nan")
        if not sdf.empty and color_col in sdf.columns:
            v = sdf[color_col].values[0]
            val = float(v) if pd.notna(v) else float("nan")

        low_sample = False
        fdf = flags_df[flags_df["state_ut"] == state] if not flags_df.empty else pd.DataFrame()
        if not fdf.empty and color_col in fdf.columns:
            fv = fdf[color_col].values[0]
            low_sample = bool(pd.notna(fv) and fv == 1)

        rows.append(dict(state=state, cx=cx, ry=ry, val=val, low_sample=low_sample))

    tdf = pd.DataFrame(rows)
    vmin = tdf["val"].min()
    vmax = tdf["val"].max()
    val_range = (vmax - vmin) if (vmax > vmin) else 1.0

    fig = go.Figure()

    for _, r in tdf.iterrows():
        x0, y0 = float(r["cx"]),  -float(r["ry"])
        x1, y1 = x0 + 0.88,       y0 - 0.88

        suppressed = np.isnan(r["val"])
        if suppressed:
            fill = "#cccccc"
        else:
            t = (r["val"] - vmin) / val_range
            if reverse_scale:
                t = 1.0 - t
            fill = _rdylgn(t)

        label    = str(r["state"])[:12]
        val_text = f"{r['val']:.1f}" if not suppressed else "–"
        suffix   = "†" if r["low_sample"] else ""
        hover_txt = (
            f"<b>{r['state']}</b><br>"
            f"{pretty(color_col)}: {val_text}"
            + (" (†low-sample)" if r["low_sample"] else "")
            + (" (suppressed)" if suppressed else "")
        )

        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=fill, line=dict(color="white", width=1.5),
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.05,
            text=f"<span style='font-size:7.5px'>{label}</span>",
            showarrow=False, font=dict(size=7, color="black"),
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2 - 0.18,
            text=f"<span style='font-size:8px'><b>{val_text}{suffix}</b></span>",
            showarrow=False, font=dict(size=8, color="black"),
        )
        # Invisible scatter for tooltip
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            mode="markers",
            marker=dict(size=35, opacity=0),
            hovertext=[hover_txt], hoverinfo="text",
            showlegend=False,
        ))

    # Colour-bar via dummy scatter
    if not tdf["val"].isna().all():
        cs = SCORE_COLORSCALE if not reverse_scale else SCORE_COLORSCALE + "_r"
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                colorscale=cs,
                cmin=vmin, cmax=vmax,
                colorbar=dict(title=pretty(color_col), thickness=12, len=0.7),
                showscale=True, size=0,
            ),
            showlegend=False, hoverinfo="skip",
        ))

    ncols = max(c for c, _ in TILE_GRID.values()) + 2
    nrows = max(r for _, r in TILE_GRID.values()) + 2
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-0.2, ncols], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-(nrows), 0.5], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        height=530, margin=dict(l=0, r=60, t=40, b=0),
        template=PLOTLY_TEMPLATE, plot_bgcolor="white",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — DERIVED KPI FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def gap_to_india(state_val: float, india_val: float, direction: str) -> float:
    """Signed gap so negative always means worse than India."""
    if np.isnan(state_val) or np.isnan(india_val):
        return float("nan")
    raw = state_val - india_val
    return raw if direction == "higher" else -raw


def gap_to_best(state_val: float, best_val: float, direction: str) -> float:
    if np.isnan(state_val) or np.isnan(best_val):
        return float("nan")
    raw = state_val - best_val
    return raw if direction == "higher" else -raw


def compute_derived_kpis(
    scored_df: pd.DataFrame,
    india_series: pd.Series,
    method: str,
) -> pd.DataFrame:
    """Add gap_to_india, n_below_india, n_bottom_quartile to scored_df."""
    out = scored_df.copy()
    ind_cols = all_index_cols()
    avail = [c for c in ind_cols if c in out.columns and c in india_series.index]

    # gap to india per indicator (sign-adjusted)
    for col in avail:
        direction = col_direction(col)
        india_val = float(india_series[col]) if pd.notna(india_series.get(col)) else float("nan")
        out[f"__gap_{col}"] = out[col].apply(
            lambda sv: gap_to_india(float(sv) if pd.notna(sv) else float("nan"),
                                    india_val, direction)
        )

    gap_tmp = [f"__gap_{c}" for c in avail if f"__gap_{c}" in out.columns]
    out["n_below_india"] = (out[gap_tmp] < 0).sum(axis=1)
    out.drop(columns=gap_tmp, inplace=True)

    # n_bottom_quartile: indicator score in bottom 25% across the scored set
    scorer = SCORERS[method]
    bq_tmp: list[str] = []
    for col, direction in [(c, col_direction(c)) for c in avail]:
        valid = out[col].dropna()
        if len(valid) < 2:
            continue
        sc = scorer(out[col], direction)
        bq_col = f"__bq_{col}"
        out[bq_col] = (sc <= 25).astype(int)
        bq_tmp.append(bq_col)
    out["n_bottom_quartile"] = out[bq_tmp].sum(axis=1) if bq_tmp else 0
    out.drop(columns=bq_tmp, inplace=True)

    return out


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — PAGE SETUP & DATA BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NFHS-5 State Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NFHS-5 State Health Dashboard")
st.caption(
    "National Family Health Survey 2019-21 · "
    "India + 36 States/UTs · Maternal Health, Vaccination, Child Nutrition"
)

# --- Data loading ---
_xlsx = BASE_DIR / "NFHS5_clean.xlsx"
_csv  = BASE_DIR / "NFHS5_values.csv"
_uploaded_bytes: Optional[bytes] = None

if not _xlsx.exists() and not _csv.exists():
    _uf = st.file_uploader("Upload NFHS5_clean.xlsx", type=["xlsx"])
    if _uf:
        _uploaded_bytes = _uf.read()
    else:
        st.info("Please upload NFHS5_clean.xlsx to start.")
        st.stop()

try:
    values_df, flags_df, dict_df = load_data(_uploaded_bytes)
except Exception as _e:
    st.error(f"Failed to load data: {_e}")
    st.stop()

validate_data(values_df)

# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Filters & Scoring")

    area_choice: str = st.selectbox("Area", ["Total", "Urban", "Rural"])

    scoring_method: str = st.selectbox(
        "Scoring method",
        ["min-max", "z-score", "percentile"],
        help="How each indicator is normalised before computing domain scores.",
    )

    exclude_small: bool = st.checkbox("Exclude small UTs from rankings", value=True)

    _all_states = sorted(
        values_df[values_df["geography_type"] == "State/UT"]["state_ut"].unique().tolist()
    )
    small_uts_choice: list[str] = st.multiselect(
        "Small UTs to exclude",
        options=_all_states,
        default=[u for u in SMALL_UTS_DEFAULT if u in _all_states],
        disabled=not exclude_small,
    )

    st.markdown("---")
    fill_with_total: bool = st.checkbox(
        "Fill blank Urban/Rural with state Total (Map & Rankings only)",
        value=False,
        help="Fills suppressed cells with the state's Total value for Map/Rankings. "
             "Filled cells are clearly noted.",
    )

    st.markdown("---")
    st.caption(
        "NFHS fact sheets provide no confidence intervals. "
        "Small score differences are not statistically meaningful. "
        "Tiers (quartile bands) are used instead of exact ranks."
    )

# ═══════════════════════════════════════════════════════════════════
# SECTION 10 — GLOBAL DERIVED DATA (shared across tabs)
# ═══════════════════════════════════════════════════════════════════

# Raw state rows for selected area
state_df    = filter_states(values_df, area_choice, exclude_small, small_uts_choice)
state_flags = filter_states(flags_df,  area_choice, exclude_small, small_uts_choice)
india_ser   = india_row(values_df, area_choice)

# Scored data (no fill)
scored_df   = compute_domain_scores(state_df, method=scoring_method)
scored_df   = compute_derived_kpis(scored_df, india_ser, scoring_method)

# Optionally filled data for map/rankings only
if fill_with_total and area_choice != "Total":
    _state_df_filled = fill_urban_rural_with_total(state_df, values_df, area_choice)
    scored_df_map    = compute_domain_scores(_state_df_filled, method=scoring_method)
else:
    _state_df_filled = state_df
    scored_df_map    = scored_df

# Total-area scored data for Drivers tab
_state_df_total = filter_states(values_df, "Total", exclude_small, small_uts_choice)
scored_total_df  = compute_domain_scores(_state_df_total, method=scoring_method)

# ═══════════════════════════════════════════════════════════════════
# SECTION 11 — TABS
# ═══════════════════════════════════════════════════════════════════

tab_labels = [
    "Overview",
    "Map & Rankings",
    "Gap Analysis",
    "Urban vs Rural",
    "Drivers",
    "State Profile",
    "Data & Method",
]
tabs = st.tabs(tab_labels)

# ─────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────
with tabs[0]:
    st.subheader("India at a Glance — National Totals")

    # KPI cards — India values
    kpi_card_cols = st.columns(len(OVERVIEW_KPI_COLS))
    for i, col in enumerate(OVERVIEW_KPI_COLS):
        spec = KPI_SPEC.get(col, (pretty(col), "", "higher", ""))
        label, unit, direction, _ = spec
        v = india_ser.get(col, np.nan)
        v_float = float(v) if pd.notna(v) else float("nan")
        with kpi_card_cols[i]:
            st.metric(label, fmt_val(v_float, unit), help=unit)

    st.divider()
    st.subheader("Best & Worst State per Domain")

    findings: list[str] = []
    bw_cols = st.columns(len(DOMAIN_SCORE_COLS))
    for i, (domain, dcol) in enumerate(DOMAIN_SCORE_COLS.items()):
        sub = scored_df[["state_ut", dcol]].dropna()
        with bw_cols[i]:
            st.markdown(f"**{domain}**")
            if sub.empty:
                st.write("No data")
                continue
            best  = sub.loc[sub[dcol].idxmax()]
            worst = sub.loc[sub[dcol].idxmin()]
            st.markdown(
                f"**Best:** {best['state_ut']} ({best[dcol]:.1f})\n\n"
                f"**Worst:** {worst['state_ut']} ({worst[dcol]:.1f})"
            )
            findings.append(
                f"In **{domain}**, {best['state_ut']} leads "
                f"(score {best[dcol]:.1f}) while {worst['state_ut']} "
                f"scores lowest ({worst[dcol]:.1f})."
            )

    st.divider()
    st.subheader("Key Findings")
    for f in findings:
        st.markdown(f"- {f}")

    # Composite ranking table
    st.divider()
    st.subheader("Composite Score Table")
    disp_cols = ["state_ut"] + list(DOMAIN_SCORE_COLS.values()) + \
                ["composite_score", "tier", "n_below_india", "n_bottom_quartile"]
    disp_cols_avail = [c for c in disp_cols if c in scored_df.columns]
    disp = (
        scored_df[disp_cols_avail]
        .dropna(subset=["composite_score"])
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    disp.index += 1
    rename_map = {
        "state_ut":                              "State/UT",
        "maternal_health_score":                 "Maternal",
        "vaccination_score":                     "Vaccination",
        "child_nutrition_score":                 "Child Nutrition",
        "composite_score":                       "Composite",
        "tier":                                  "Tier",
        "n_below_india":                         "# Below India",
        "n_bottom_quartile":                     "# Bottom-quartile",
    }
    num_rename = {v: v for k, v in rename_map.items() if k in
                  ["maternal_health_score", "vaccination_score",
                   "child_nutrition_score", "composite_score"]}
    fmt_map = {rename_map.get(c, c): "{:.1f}" for c in
               ["maternal_health_score", "vaccination_score",
                "child_nutrition_score", "composite_score"]}
    st.dataframe(
        disp.rename(columns=rename_map).style.format(fmt_map, na_rep="n/a"),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────
# TAB 2 · MAP & RANKINGS
# ─────────────────────────────────────────────────
with tabs[1]:
    map_left, map_right = st.columns([2, 1])

    with map_left:
        st.subheader("Tile-Grid Map of India")

        _score_options = {
            "Composite Score":       "composite_score",
            "Maternal Health Score": "maternal_health_score",
            "Vaccination Score":     "vaccination_score",
            "Child Nutrition Score": "child_nutrition_score",
        }
        _ind_options = {pretty(c): c for c in all_index_cols()
                        if c in state_df.columns}
        _all_map_options = list(_score_options.keys()) + list(_ind_options.keys())

        map_choice = st.selectbox("Colour by", _all_map_options, key="map_choice")

        if map_choice in _score_options:
            map_col     = _score_options[map_choice]
            map_reverse = False
        else:
            map_col     = _ind_options.get(map_choice, "composite_score")
            map_reverse = (col_direction(map_col) == "lower")

        # Ensure the column exists in scored_df_map (raw indicators do)
        _map_source = scored_df_map.copy()
        if map_col not in _map_source.columns and map_col in _state_df_filled.columns:
            _map_source = _map_source.merge(
                _state_df_filled[["state_ut", map_col]], on="state_ut", how="left"
            )

        fig_map = build_tile_map(
            _map_source, map_col, state_flags,
            title=f"{map_choice} · {area_choice}",
            reverse_scale=map_reverse,
        )
        st.plotly_chart(fig_map, use_container_width=True)
        if fill_with_total and area_choice != "Total":
            st.caption("Some blank cells were filled with the state's Total value.")
        st.caption("† = estimate based on 25–49 unweighted cases  ·  Grey = suppressed (*)")

    with map_right:
        st.subheader("Rankings by Tier")
        rank_data = (
            scored_df[["state_ut", "composite_score", "tier"]]
            .dropna(subset=["composite_score"])
            .sort_values("composite_score", ascending=True)
        )
        if not rank_data.empty:
            bar_colors = [TIER_COLORS.get(str(t), "#cccccc") for t in rank_data["tier"]]
            fig_rank = go.Figure(go.Bar(
                x=rank_data["composite_score"],
                y=rank_data["state_ut"],
                orientation="h",
                marker_color=bar_colors,
                text=rank_data["composite_score"].round(1).astype(str),
                textposition="outside",
                hovertemplate="%{y}<br>Composite: %{x:.1f}<extra></extra>",
            ))
            # SDG benchmark lines (only outcome cols have them, composite has none)
            fig_rank.update_layout(
                height=700,
                xaxis=dict(title="Composite Score (0–100)", range=[0, 105]),
                yaxis=dict(title=""),
                margin=dict(l=10, r=40, t=10, b=10),
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_rank, use_container_width=True)
            st.caption(
                "Tiers: **Top** (75–100)  ·  **Upper-mid** (50–75)  ·  "
                "**Lower-mid** (25–50)  ·  **Bottom** (0–25)"
            )
        else:
            st.info("Not enough data to rank states.")

# ─────────────────────────────────────────────────
# TAB 3 · GAP ANALYSIS
# ─────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Gap Analysis")

    # ── 3a: Gap to India / best state ────────────────────────────
    st.markdown("### (a) Gap to India and to Best State")

    _ind_avail = [c for c in all_index_cols() if c in state_df.columns]
    if _ind_avail:
        gap_sel_pretty = st.selectbox(
            "Indicator", [pretty(c) for c in _ind_avail], key="gap3a_ind"
        )
        gap_col = next((c for c in _ind_avail if pretty(c) == gap_sel_pretty), _ind_avail[0])
        direction_3a = col_direction(gap_col)
        india_v3a = float(india_ser.get(gap_col, np.nan)) if pd.notna(india_ser.get(gap_col)) else float("nan")

        sub3a = state_df[["state_ut", gap_col]].dropna().copy()
        if not sub3a.empty:
            if direction_3a == "higher":
                best_v3a = sub3a[gap_col].max()
                best_s3a = sub3a.loc[sub3a[gap_col].idxmax(), "state_ut"]
            else:
                best_v3a = sub3a[gap_col].min()
                best_s3a = sub3a.loc[sub3a[gap_col].idxmin(), "state_ut"]

            sub3a["gap_india"] = sub3a[gap_col].apply(
                lambda v: gap_to_india(float(v), india_v3a, direction_3a)
            )
            sub3a["gap_best"] = sub3a[gap_col].apply(
                lambda v: gap_to_best(float(v), float(best_v3a), direction_3a)
            )
            sub3a = sub3a.sort_values("gap_india", ascending=True)

            fig3a = go.Figure()
            fig3a.add_trace(go.Bar(
                name="vs India", y=sub3a["state_ut"], x=sub3a["gap_india"],
                orientation="h", marker_color="#4472c4",
            ))
            fig3a.add_trace(go.Bar(
                name="vs Best state", y=sub3a["state_ut"], x=sub3a["gap_best"],
                orientation="h", marker_color="#ed7d31",
            ))
            india_label = f"{india_v3a:.1f}" if not np.isnan(india_v3a) else "n/a"
            fig3a.update_layout(
                barmode="group", height=560,
                xaxis_title="Gap (sign-adjusted: negative = worse than reference)",
                title=f"{gap_sel_pretty}  |  India = {india_label}  |  Best: {best_s3a} = {best_v3a:.1f}",
                template=PLOTLY_TEMPLATE,
            )
            fig3a.add_vline(x=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig3a, use_container_width=True)

    # ── 3b: Heatmap ───────────────────────────────────────────────
    st.markdown("### (b) States × Indicators Score Heatmap")
    _heat_data: dict[str, list] = {}
    _heat_labels: list[str] = []
    _heat_states = scored_df["state_ut"].tolist()

    scorer_fn = SCORERS[scoring_method]
    for domain, indicators in INDEX_SPEC.items():
        for col, direction in indicators:
            if col not in state_df.columns:
                continue
            sc = scorer_fn(state_df.set_index("state_ut")[col], direction)
            _heat_data[f"[{domain[:3]}] {pretty(col)}"] = [
                sc.get(s, np.nan) for s in _heat_states
            ]
            _heat_labels.append(f"[{domain[:3]}] {pretty(col)}")

    if _heat_data:
        heat_z = np.array(list(_heat_data.values()), dtype=float)
        fig3b = px.imshow(
            heat_z,
            x=_heat_states,
            y=list(_heat_data.keys()),
            color_continuous_scale=SCORE_COLORSCALE,
            zmin=0, zmax=100,
            labels=dict(color="Score"),
            aspect="auto",
        )
        fig3b.update_layout(
            height=max(420, 16 * len(_heat_data)),
            template=PLOTLY_TEMPLATE,
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig3b, use_container_width=True)

    # ── 3c: Inequality ────────────────────────────────────────────
    st.markdown("### (c) Inequality Across States")
    _ineq_rows: list[dict] = []
    for col, direction in [(c, col_direction(c)) for c in all_index_cols()]:
        if col not in state_df.columns:
            continue
        s = state_df[col].dropna()
        if len(s) < 3:
            continue
        spread = float(s.max() - s.min())
        cv     = float(s.std() / s.mean() * 100) if s.mean() != 0 else float("nan")
        _ineq_rows.append({"Indicator": pretty(col), "Max–Min Spread": spread, "CV (%)": cv})

    if _ineq_rows:
        ineq_df = pd.DataFrame(_ineq_rows).sort_values("Max–Min Spread", ascending=False)
        fig3c = px.bar(
            ineq_df, x="Max–Min Spread", y="Indicator", orientation="h",
            color="CV (%)", color_continuous_scale="Blues",
            title="Spread of indicator values across states (higher = more unequal)",
            height=max(420, 20 * len(ineq_df)),
        )
        fig3c.update_layout(template=PLOTLY_TEMPLATE, yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig3c, use_container_width=True)

    # ── 3d: Cascades ──────────────────────────────────────────────
    st.markdown("### (d) Care Cascades")
    casc_l, casc_r = st.columns(2)

    for (cols_def, labels_def, title_def), container in [
        ((MATERNAL_CASCADE_COLS,     MATERNAL_CASCADE_LABELS,     "Maternal Care Cascade"),    casc_l),
        ((IMMUNISATION_CASCADE_COLS, IMMUNISATION_CASCADE_LABELS, "Immunisation Drop-off"), casc_r),
    ]:
        with container:
            avail_c = [c for c in cols_def if c in state_df.columns]
            if not avail_c:
                st.info("No data for cascade.")
                continue
            casc_states = ["India (National)"] + sorted(state_df["state_ut"].tolist())
            casc_sel = st.selectbox("State", casc_states, key=f"casc_{title_def}")
            if casc_sel == "India (National)":
                casc_row_data = india_ser
                casc_name = "India"
            else:
                cr = state_df[state_df["state_ut"] == casc_sel]
                if cr.empty:
                    st.info("No data.")
                    continue
                casc_row_data = cr.iloc[0]
                casc_name = casc_sel

            casc_vals = [
                float(casc_row_data[c]) if pd.notna(casc_row_data.get(c)) else float("nan")
                for c in avail_c
            ]
            fig3d = go.Figure(go.Bar(
                x=labels_def[:len(avail_c)],
                y=casc_vals,
                marker_color="#4472c4",
                text=[f"{v:.1f}%" if not np.isnan(v) else "n/a" for v in casc_vals],
                textposition="outside",
            ))
            fig3d.update_layout(
                title=f"{title_def} — {casc_name}",
                yaxis=dict(range=[0, 110], title="%"),
                template=PLOTLY_TEMPLATE, height=350,
            )
            st.plotly_chart(fig3d, use_container_width=True)

    # ── 3e: Domain vs domain scatter ──────────────────────────────
    st.markdown("### (e) Domain vs Domain Scatter")
    d_names = list(INDEX_SPEC.keys())
    scat_x_name = st.selectbox("X-axis domain", d_names, key="scat3e_x")
    scat_y_name = st.selectbox("Y-axis domain", d_names, index=min(1, len(d_names) - 1), key="scat3e_y")
    sx_col = DOMAIN_SCORE_COLS[scat_x_name]
    sy_col = DOMAIN_SCORE_COLS[scat_y_name]

    if sx_col == sy_col:
        scat3e = pd.DataFrame(columns=["state_ut", sx_col, sy_col, "tier"])
    else:
        scat3e = scored_df[["state_ut", sx_col, sy_col, "tier"]].dropna()
    if not scat3e.empty:
        fig3e = px.scatter(
            scat3e, x=sx_col, y=sy_col, text="state_ut", color="tier",
            color_discrete_map=TIER_COLORS,
            labels={sx_col: scat_x_name, sy_col: scat_y_name},
            title=f"{scat_x_name} vs {scat_y_name}",
            template=PLOTLY_TEMPLATE, height=500,
        )
        fig3e.update_traces(textposition="top center", textfont_size=9)
        fig3e.update_layout(xaxis_range=[0, 100], yaxis_range=[0, 100])
        st.plotly_chart(fig3e, use_container_width=True)
    else:
        st.info("Not enough scored data for scatter.")

# ─────────────────────────────────────────────────
# TAB 4 · URBAN vs RURAL
# ─────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Urban vs Rural Gaps")
    st.caption(
        "Gap = Rural value − Urban value. "
        "Computed from raw Urban/Rural rows only — never imputed. "
        "Cells where either side is blank are hidden. "
        "† = low-sample estimate."
    )

    _urban_df  = values_df[(values_df["geography_type"] == "State/UT") & (values_df["area"] == "Urban")].set_index("state_ut")
    _rural_df  = values_df[(values_df["geography_type"] == "State/UT") & (values_df["area"] == "Rural")].set_index("state_ut")
    _uf_flags  = flags_df[(flags_df["geography_type"] == "State/UT") & (flags_df["area"] == "Urban")].set_index("state_ut")
    _rf_flags  = flags_df[(flags_df["geography_type"] == "State/UT") & (flags_df["area"] == "Rural")].set_index("state_ut")

    if exclude_small:
        _urban_df = _urban_df[~_urban_df.index.isin(small_uts_choice)]
        _rural_df = _rural_df[~_rural_df.index.isin(small_uts_choice)]

    _ind4_avail = [c for c in all_index_cols() if c in _urban_df.columns and c in _rural_df.columns]
    if _ind4_avail:
        ur_sel_pretty = st.selectbox(
            "Select indicator", [pretty(c) for c in _ind4_avail], key="ur4_ind"
        )
        ur_col = next((c for c in _ind4_avail if pretty(c) == ur_sel_pretty), _ind4_avail[0])

        _common = _urban_df.index.intersection(_rural_df.index)
        _ur_rows: list[dict] = []
        for s in _common:
            u = _urban_df.loc[s, ur_col] if ur_col in _urban_df.columns else np.nan
            r = _rural_df.loc[s, ur_col] if ur_col in _rural_df.columns else np.nan
            if pd.isna(u) or pd.isna(r):
                continue
            uf = int(_uf_flags.loc[s, ur_col]) if (s in _uf_flags.index and ur_col in _uf_flags.columns and pd.notna(_uf_flags.loc[s, ur_col])) else 0
            rf = int(_rf_flags.loc[s, ur_col]) if (s in _rf_flags.index and ur_col in _rf_flags.columns and pd.notna(_rf_flags.loc[s, ur_col])) else 0
            _ur_rows.append({
                "state_ut": s,
                "urban": float(u), "rural": float(r),
                "gap": float(r) - float(u),
                "low_sample": bool(uf or rf),
            })

        if _ur_rows:
            ur_df4 = pd.DataFrame(_ur_rows).sort_values("gap", ascending=True)
            bar4_col = ["#d62728" if g < 0 else "#2ca02c" for g in ur_df4["gap"]]
            suffix4  = ["†" if ls else "" for ls in ur_df4["low_sample"]]
            fig4 = go.Figure(go.Bar(
                y=[f"{s}{sx}" for s, sx in zip(ur_df4["state_ut"], suffix4)],
                x=ur_df4["gap"],
                orientation="h",
                marker_color=bar4_col,
                hovertemplate="%{y}<br>Gap (Rural−Urban): %{x:.1f}<extra></extra>",
            ))
            fig4.add_vline(x=0, line_dash="dash", line_color="grey")
            fig4.update_layout(
                xaxis_title=f"Gap = Rural − Urban  ({ur_sel_pretty})",
                yaxis_title="", template=PLOTLY_TEMPLATE,
                height=max(420, 16 * len(ur_df4)),
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Green = rural higher than urban  ·  Red = urban higher than rural  ·  † = low-sample")

            st.markdown("#### States with widest gaps")
            _top_gap = (
                ur_df4.assign(abs_gap=ur_df4["gap"].abs())
                .sort_values("abs_gap", ascending=False)
                .head(10)
            )
            st.dataframe(
                _top_gap[["state_ut", "urban", "rural", "gap"]]
                .rename(columns={"state_ut": "State/UT", "urban": "Urban",
                                  "rural": "Rural", "gap": "Gap (R−U)"})
                .reset_index(drop=True)
                .style.format({"Urban": "{:.1f}", "Rural": "{:.1f}", "Gap (R−U)": "{:.1f}"}),
                use_container_width=True,
            )

            # Summary: widest-gap indicators
            st.markdown("#### Indicator spread: Urban vs Rural (all indicators)")
            _summary_rows: list[dict] = []
            for c in _ind4_avail:
                _sr: list[dict] = []
                for s in _common:
                    u2 = _urban_df.loc[s, c] if c in _urban_df.columns else np.nan
                    r2 = _rural_df.loc[s, c] if c in _rural_df.columns else np.nan
                    if pd.notna(u2) and pd.notna(r2):
                        _sr.append({"gap": float(r2) - float(u2)})
                if _sr:
                    gaps = [x["gap"] for x in _sr]
                    _summary_rows.append({
                        "Indicator": pretty(c),
                        "Mean Gap (R−U)": float(np.mean(gaps)),
                        "Std Dev": float(np.std(gaps)),
                        "Max |Gap|": float(np.max(np.abs(gaps))),
                    })
            if _summary_rows:
                summ_df = (
                    pd.DataFrame(_summary_rows)
                    .sort_values("Max |Gap|", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(summ_df.style.format("{:.2f}", subset=["Mean Gap (R−U)", "Std Dev", "Max |Gap|"]),
                             use_container_width=True)
        else:
            st.info("No paired Urban/Rural data for this indicator.")
    else:
        st.info("Urban/Rural data not available.")

# ─────────────────────────────────────────────────
# TAB 5 · DRIVERS
# ─────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Drivers: Context Variables vs Health Scores & Outcomes")
    st.info(
        "Spearman correlations computed via DataFrame.corr() across "
        f"~{len(scored_total_df)} states (Total rows). "
        "Cross-sectional ecological correlation — shows association, not causation. "
        "Small sample size means estimates are uncertain."
    )

    _ctx_avail   = [c for c in CONTEXT_COLS  if c in scored_total_df.columns]
    _tgt_avail   = [c for c in
                    list(DOMAIN_SCORE_COLS.values()) + ["composite_score"] + OUTCOME_COLS
                    if c in scored_total_df.columns]

    if _ctx_avail and _tgt_avail:
        _corr_z = np.full((len(_ctx_avail), len(_tgt_avail)), float("nan"))
        for i, cc in enumerate(_ctx_avail):
            for j, tc in enumerate(_tgt_avail):
                _corr_z[i, j] = spearman_corr(scored_total_df[cc], scored_total_df[tc])

        fig5a = px.imshow(
            _corr_z,
            x=[pretty(c) for c in _tgt_avail],
            y=[pretty(c) for c in _ctx_avail],
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            text_auto=".2f",
            labels=dict(color="Spearman r"),
            title="Spearman Correlation: Context → Domain Scores & Outcomes",
        )
        fig5a.update_layout(
            height=420, template=PLOTLY_TEMPLATE,
            xaxis=dict(tickangle=30, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig5a, use_container_width=True)

        st.markdown("#### Scatter: context × target with trend line")
        sc5_x = st.selectbox("Context variable", _ctx_avail, format_func=pretty, key="sc5_x")
        sc5_y = st.selectbox("Target variable",  _tgt_avail, format_func=pretty, key="sc5_y")

        sc5_data = scored_total_df[["state_ut", sc5_x, sc5_y]].dropna()
        if len(sc5_data) >= 4:
            x5 = sc5_data[sc5_x].values.astype(float)
            y5 = sc5_data[sc5_y].values.astype(float)
            coeffs5 = np.polyfit(x5, y5, 1)
            x5_line = np.linspace(x5.min(), x5.max(), 60)
            y5_line = np.polyval(coeffs5, x5_line)
            rho5 = spearman_corr(sc5_data[sc5_x], sc5_data[sc5_y])

            fig5b = go.Figure()
            fig5b.add_trace(go.Scatter(
                x=x5, y=y5, mode="markers+text",
                text=sc5_data["state_ut"], textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=8, color="#4472c4"),
                name="States",
            ))
            fig5b.add_trace(go.Scatter(
                x=x5_line, y=y5_line, mode="lines",
                line=dict(dash="dash", color="firebrick"),
                name="Trend (numpy.polyfit)",
            ))
            fig5b.update_layout(
                xaxis_title=pretty(sc5_x),
                yaxis_title=pretty(sc5_y),
                title=f"Spearman ρ = {rho5:.3f}  (n = {len(sc5_data)} states)",
                template=PLOTLY_TEMPLATE, height=500,
            )
            st.plotly_chart(fig5b, use_container_width=True)
            st.caption(
                "Trend line computed with numpy.polyfit (OLS). "
                "Ecological correlation across states — individual-level inference is not valid."
            )
        else:
            st.info("Not enough data points for scatter plot.")
    else:
        st.info("Context or target columns not available.")

# ─────────────────────────────────────────────────
# TAB 6 · STATE PROFILE
# ─────────────────────────────────────────────────
with tabs[5]:
    st.subheader("State Profile")

    _all_states6 = sorted(state_df["state_ut"].tolist())
    if not _all_states6:
        st.info("No state data for current filters.")
    else:
        _default6 = _all_states6[:min(2, len(_all_states6))]
        sel_states6 = st.multiselect(
            "Select 1 or 2 states", _all_states6,
            default=_default6, max_selections=2, key="profile6",
        )
        if not sel_states6:
            st.info("Please select at least one state.")
        else:
            # Score in context of state set + India so radar is comparable
            _sp_combined = pd.concat(
                [state_df,
                 values_df[(values_df["geography_type"] == "National") & (values_df["area"] == area_choice)]],
                ignore_index=True,
            )
            _sp_scored = compute_domain_scores(_sp_combined, method=scoring_method)

            # Radar
            d_labels6 = list(INDEX_SPEC.keys())
            d_score6  = [DOMAIN_SCORE_COLS[d] for d in d_labels6]
            theta6    = d_labels6 + [d_labels6[0]]
            colors6   = ["#888888", "#1f77b4", "#ff7f0e"]
            traces6   = []

            india6 = _sp_scored[_sp_scored["state_ut"] == "India"]
            if not india6.empty:
                iv = [float(india6[c].values[0]) if c in india6.columns and pd.notna(india6[c].values[0]) else 0.0
                      for c in d_score6]
                traces6.append(("India", iv + [iv[0]], colors6[0]))

            for si, sel6 in enumerate(sel_states6):
                r6 = _sp_scored[_sp_scored["state_ut"] == sel6]
                if r6.empty:
                    continue
                rv = [float(r6[c].values[0]) if c in r6.columns and pd.notna(r6[c].values[0]) else 0.0
                      for c in d_score6]
                traces6.append((sel6, rv + [rv[0]], colors6[si + 1]))

            if traces6:
                fig6a = go.Figure()
                for name6, r_vals6, color6 in traces6:
                    fig6a.add_trace(go.Scatterpolar(
                        r=r_vals6, theta=theta6,
                        fill="toself", name=name6,
                        line_color=color6, opacity=0.6,
                    ))
                fig6a.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100])),
                    template=PLOTLY_TEMPLATE,
                    title="Domain Scores (0 = worst, 100 = best)",
                    height=430,
                )
                st.plotly_chart(fig6a, use_container_width=True)

            # KPI cards per state vs India
            st.markdown("#### Headline KPIs vs India")
            _kpi6_cols = list(KPI_SPEC.keys())
            _kpi6_avail = [c for c in _kpi6_cols if c in state_df.columns]
            if _kpi6_avail and not india6.empty:
                for sel6 in sel_states6:
                    st.markdown(f"**{sel6}**")
                    _row6 = state_df[state_df["state_ut"] == sel6]
                    if _row6.empty:
                        continue
                    kcard_cols = st.columns(min(len(_kpi6_avail), 6))
                    for ki, kcol in enumerate(_kpi6_avail[:6]):
                        label6, unit6, dir6, domain6 = KPI_SPEC[kcol]
                        sv6 = float(_row6[kcol].values[0]) if pd.notna(_row6[kcol].values[0]) else float("nan")
                        iv6 = float(india_ser.get(kcol, np.nan)) if pd.notna(india_ser.get(kcol)) else float("nan")
                        is_low = bool(state_flags[state_flags["state_ut"] == sel6][kcol].values[0] == 1) \
                            if (not state_flags.empty and kcol in state_flags.columns
                                and not state_flags[state_flags["state_ut"] == sel6].empty) else False
                        val_str = fmt_val(sv6, unit6, is_low)
                        raw_delta = (sv6 - iv6) if not (np.isnan(sv6) or np.isnan(iv6)) else float("nan")
                        delta_str = f"{raw_delta:+.1f}" if not np.isnan(raw_delta) else ""
                        dc = delta_color(raw_delta, dir6) if not np.isnan(raw_delta) else "normal"
                        with kcard_cols[ki]:
                            st.metric(label6, val_str, delta=delta_str, delta_color=dc)

            # Weakest 5 indicators
            for sel6 in sel_states6:
                st.markdown(f"#### Weakest 5 indicators — {sel6}")
                _w_rows: list[dict] = []
                for col6, dir6 in [(c, col_direction(c)) for c in all_index_cols()]:
                    if col6 not in state_df.columns:
                        continue
                    _s6 = state_df.set_index("state_ut")[col6].dropna()
                    if len(_s6) < 2:
                        continue
                    _sc6 = SCORERS[scoring_method](_s6, dir6)
                    sv6 = _sc6.get(sel6, float("nan"))
                    rv6 = state_df[state_df["state_ut"] == sel6][col6].values
                    raw6 = float(rv6[0]) if len(rv6) > 0 and pd.notna(rv6[0]) else float("nan")
                    _w_rows.append({"Indicator": pretty(col6), "Score": sv6, "Raw value": raw6})
                _w_df = (
                    pd.DataFrame(_w_rows)
                    .dropna(subset=["Score"])
                    .sort_values("Score")
                    .head(5)
                    .reset_index(drop=True)
                )
                st.dataframe(_w_df.style.format({"Score": "{:.1f}", "Raw value": "{:.1f}"}),
                             use_container_width=True)

# ─────────────────────────────────────────────────
# TAB 7 · DATA & METHOD
# ─────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Data & Methodology")

    method_text = {
        "min-max": (
            "**Min-Max Normalisation:** Each indicator is rescaled so the lowest observed "
            "value = 0 and the highest = 100. For *lower-is-better* indicators (e.g. stunting) "
            "the scale is reversed, so **higher score always means better**. "
            "Domain score = unweighted mean of available indicator scores, "
            "requiring ≥75 % of indicators to be present. "
            "Composite = mean of all three domain scores (all three required)."
        ),
        "z-score": (
            "**Z-score:** Values are standardised (subtract mean, divide by SD). "
            "Z-scores are then min-max rescaled to 0–100 and reversed for "
            "*lower-is-better* indicators so **higher score always means better**."
        ),
        "percentile": (
            "**Percentile Rank:** Each state's percentile rank (0–100) is used directly. "
            "Reversed for *lower-is-better* indicators so **higher score always means better**."
        ),
    }
    st.markdown(f"### Scoring: {scoring_method}")
    st.markdown(method_text[scoring_method])

    st.markdown(
        "### Tiers\n"
        "States are grouped into quartile bands based on their **composite score**: "
        "**Top** (75–100) · **Upper-mid** (50–75) · **Lower-mid** (25–50) · **Bottom** (0–25). "
        "Tiers are used instead of exact ranks because NFHS fact sheets provide no "
        "confidence intervals — small score differences are not statistically meaningful."
    )

    st.markdown(
        "### Suppression & Low-Sample Notes\n"
        "- **Suppressed (`*`):** Too few cases to report; stored as blank (NaN). Never filled with 0.\n"
        "- **Low-sample (†):** Estimate based on 25–49 unweighted cases. "
        "Original Excel stored these as negatives; absolute value taken. "
        "`Low_sample_flags` sheet records 1 for each such cell.\n"
        "- Scores exclude suppressed cells; states with <75 % of indicators available "
        "in a domain receive NaN for that domain."
    )

    st.markdown(
        f"### SDG Benchmarks\n"
        f"Reference lines shown where applicable: "
        + ", ".join(
            f"**{pretty(k)}** = {v} ({BENCHMARK_LABEL})"
            for k, v in BENCHMARKS.items() if v is not None
        )
    )

    if not dict_df.empty:
        st.markdown("### Data Dictionary")
        st.dataframe(dict_df, use_container_width=True, height=350)

        st.markdown("### Suppressed & Low-Sample Counts per Column")
        _sum7 = (
            dict_df[dict_df["category"] != "identifier"]
            [["short_name", "category", "suppressed_count", "low_sample_count"]]
            .sort_values("suppressed_count", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(_sum7, use_container_width=True, height=300)
    else:
        st.info("Dictionary not available (CSV fallback mode).")

    st.markdown("### Download Scored & Filtered Table")
    _keep = (
        ["state_ut", "area", "geography_type"]
        + list(DOMAIN_SCORE_COLS.values())
        + ["composite_score", "tier", "n_below_india", "n_bottom_quartile"]
        + [c for c in scored_df.columns if c not in
           {"state_ut", "area", "geography_type", "composite_score", "tier",
            "n_below_india", "n_bottom_quartile"} | set(DOMAIN_SCORE_COLS.values())]
    )
    _dl_df = scored_df[[c for c in _keep if c in scored_df.columns]].copy()
    _csv_bytes = _dl_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="Download scored & filtered table (CSV)",
        data=_csv_bytes,
        file_name=f"nfhs5_scored_{area_choice.lower()}_{scoring_method}.csv",
        mime="text/csv",
    )
