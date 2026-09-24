"""
build_nfhs5_dataset.py
Extracts analysis-ready dataset from NFHS_5_Factsheets_Data.xls
sheet "India & States".

Output: NFHS5_clean.xlsx  (sheets: Notes, Dictionary, Values, Low_sample_flags)
        NFHS5_values.csv
"""

import math
import re
import sys
import warnings

import numpy as np
import pandas as pd
import xlrd

# ---------------------------------------------------------------------------
# 1.  COLUMN SELECTION
#     Each entry: (short_name, category, original_col_index)
#     Col indices are 0-based, verified against the header dump.
# ---------------------------------------------------------------------------

SELECTED_COLS = [
    # Identifiers
    ("state_ut",            "identifier",       0),
    ("area",                "identifier",       1),
    ("hh_surveyed",         "identifier",       2),
    ("women_interviewed",   "identifier",       3),

    # Maternal health
    ("anc_first_trimester_pct",     "maternal_health",  44),
    ("anc_4plus_visits_pct",        "maternal_health",  45),
    ("tetanus_protected_birth_pct", "maternal_health",  46),
    ("ifa_100days_pct",             "maternal_health",  47),
    ("ifa_180days_pct",             "maternal_health",  48),
    ("mcp_card_pct",                "maternal_health",  49),
    ("postnatal_care_mother_2days_pct", "maternal_health", 50),
    ("oop_cost_public_delivery_rs", "maternal_health",  51),
    ("institutional_births_pct",    "maternal_health",  54),
    ("institutional_births_public_pct", "maternal_health", 55),
    ("skilled_birth_attendance_pct","maternal_health",  57),
    ("caesarean_overall_pct",       "maternal_health",  58),
    ("caesarean_private_pct",       "maternal_health",  59),
    ("caesarean_public_pct",        "maternal_health",  60),

    # Vaccination
    ("full_immunisation_card_recall_pct", "vaccination",  61),
    ("bcg_pct",                     "vaccination",      63),
    ("polio3_pct",                  "vaccination",      64),
    ("penta_dpt3_pct",              "vaccination",      65),
    ("mcv1_pct",                    "vaccination",      66),
    ("mcv2_24to35m_pct",            "vaccination",      67),
    ("rotavirus3_pct",              "vaccination",      68),
    ("vitamin_a_pct",               "vaccination",      70),

    # Child nutrition
    ("breastfed_1hour_pct",         "child_nutrition",  79),
    ("exclusive_breastfed_u6m_pct", "child_nutrition",  80),
    ("adequate_diet_6to23m_pct",    "child_nutrition",  84),
    ("stunted_pct",                 "child_nutrition",  85),
    ("wasted_pct",                  "child_nutrition",  86),
    ("severely_wasted_pct",         "child_nutrition",  87),
    ("underweight_pct",             "child_nutrition",  88),
    ("overweight_child_pct",        "child_nutrition",  89),
    ("children_anaemic_6to59m_pct", "child_nutrition",  96),

    # Women's nutrition
    ("women_bmi_below_normal_pct",  "womens_nutrition", 90),
    ("women_anaemic_all_pct",       "womens_nutrition", 99),
    ("women_anaemic_pregnant_pct",  "womens_nutrition", 98),

    # Mortality / fertility
    ("neonatal_mortality_rate",     "mortality_fertility", 29),
    ("infant_mortality_rate",       "mortality_fertility", 30),
    ("under5_mortality_rate",       "mortality_fertility", 31),
    ("total_fertility_rate",        "mortality_fertility", 26),

    # Context
    ("women_married_before18_pct",  "context",          24),
    ("women_literate_pct",          "context",          18),
    ("women_10plus_yrs_school_pct", "context",          20),
    ("improved_drinking_water_pct", "context",          12),
    ("improved_sanitation_pct",     "context",          13),
    ("clean_cooking_fuel_pct",      "context",          14),
    ("health_insurance_pct",        "context",          16),
]

# Short names for percentage-type columns (used for 0-100 range check)
NON_PCT_COLS = {
    "state_ut", "area", "hh_surveyed", "women_interviewed",
    "oop_cost_public_delivery_rs",
    "neonatal_mortality_rate", "infant_mortality_rate",
    "under5_mortality_rate", "total_fertility_rate",
}

# ---------------------------------------------------------------------------
# 2.  LOAD RAW DATA
# ---------------------------------------------------------------------------

def load_raw(path: str) -> tuple[list[str], list[list]]:
    wb = xlrd.open_workbook(path)
    ws = wb.sheet_by_name("India & States")
    headers = [ws.cell_value(0, c) for c in range(ws.ncols)]
    rows = []
    for r in range(1, ws.nrows):
        row = []
        for c in range(ws.ncols):
            ct = ws.cell_type(r, c)
            cv = ws.cell_value(r, c)
            if ct == xlrd.XL_CELL_EMPTY:
                row.append(None)
            else:
                row.append(cv)
        rows.append(row)
    return headers, rows


# ---------------------------------------------------------------------------
# 3.  BUILD SELECTED DATAFRAME
# ---------------------------------------------------------------------------

def build_df(headers: list[str], rows: list[list]) -> tuple[pd.DataFrame, pd.DataFrame]:
    short_names = [t[0] for t in SELECTED_COLS]
    orig_indices = [t[2] for t in SELECTED_COLS]

    raw_data = {}
    for sn, idx in zip(short_names, orig_indices):
        col_vals = [row[idx] for row in rows]
        raw_data[sn] = col_vals

    df = pd.DataFrame(raw_data)

    # Build a parallel low-sample flags frame (same shape, 0/1)
    flag_data = {sn: [0] * len(rows) for sn in short_names}
    flag_df = pd.DataFrame(flag_data)
    # Identifier flags stay 0 — only numeric indicator cols get flagged
    id_cols = {"state_ut", "area", "hh_surveyed", "women_interviewed"}

    indicator_short = [t[0] for t in SELECTED_COLS if t[0] not in id_cols]
    indicator_idx   = [t[2] for t in SELECTED_COLS if t[0] not in id_cols]

    for sn, idx in zip(indicator_short, indicator_idx):
        for r, row in enumerate(rows):
            v = row[idx]
            if isinstance(v, (int, float)) and not math.isnan(v) and v < 0:
                flag_df.at[r, sn] = 1

    return df, flag_df


# ---------------------------------------------------------------------------
# 4.  CLEANING
# ---------------------------------------------------------------------------

def clean_df(df: pd.DataFrame, flag_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = {"state_ut", "area", "hh_surveyed", "women_interviewed"}
    indicator_cols = [c for c in df.columns if c not in id_cols]

    # 4a. Drop non-data rows (rows where state_ut is blank/NaN)
    df = df[df["state_ut"].notna() & (df["state_ut"] != "")].copy()
    flag_df = flag_df.loc[df.index].copy()
    df.reset_index(drop=True, inplace=True)
    flag_df.reset_index(drop=True, inplace=True)

    # 4b. Fix spelling
    df["state_ut"] = df["state_ut"].str.replace("Maharastra", "Maharashtra", regex=False)

    # 4c. Add geography_type
    df.insert(2, "geography_type", df["state_ut"].apply(
        lambda x: "National" if str(x).strip().lower() == "india" else "State/UT"
    ))
    flag_df.insert(2, "geography_type", 0)   # flag col, always 0

    # 4d. Convert indicator columns to numeric:
    #     - "*" -> NaN
    #     - negative -> abs (low-sample artefact), already flagged
    for col in indicator_cols:
        def coerce(v):
            if v is None:
                return np.nan
            if isinstance(v, str):
                v2 = v.strip()
                if v2 == "*" or v2 == "":
                    return np.nan
                try:
                    return abs(float(v2))
                except ValueError:
                    return np.nan
            if isinstance(v, (int, float)):
                if math.isnan(v):
                    return np.nan
                return abs(v)
            return np.nan

        df[col] = df[col].apply(coerce)

    # hh_surveyed and women_interviewed: force positive int-like floats
    for col in ("hh_surveyed", "women_interviewed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, flag_df


# ---------------------------------------------------------------------------
# 5.  BUILD DICTIONARY
# ---------------------------------------------------------------------------

def build_dictionary(
    headers: list[str],
    df: pd.DataFrame,
    flag_df: pd.DataFrame,
) -> pd.DataFrame:
    id_cols = {"state_ut", "area", "hh_surveyed", "women_interviewed",
               "geography_type"}

    rows_out = []
    for t in SELECTED_COLS:
        short_name, category, orig_idx = t
        orig_col = headers[orig_idx]
        # Count suppressed (NaN due to "*") — original had "*" -> NaN
        # We can reconstruct: any NaN in a non-id col could be suppressed or
        # genuinely missing.  We loaded original raws so we can count directly.
        # For the dictionary we report from the cleaned df.
        col_vals = df[short_name] if short_name in df.columns else pd.Series(dtype=float)
        flag_vals = flag_df[short_name] if short_name in flag_df.columns else pd.Series(dtype=int)

        suppressed_count = int(col_vals.isna().sum()) if short_name not in id_cols else 0
        low_sample_count = int(flag_vals.sum()) if short_name not in id_cols else 0

        rows_out.append({
            "short_name": short_name,
            "category": category,
            "original_column": orig_col.strip(),
            "suppressed_count": suppressed_count,
            "low_sample_count": low_sample_count,
        })

    # Add geography_type
    rows_out.insert(2, {
        "short_name": "geography_type",
        "category": "identifier",
        "original_column": "(derived)",
        "suppressed_count": 0,
        "low_sample_count": 0,
    })

    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 6.  VALIDATION
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, orig_headers, orig_rows) -> dict:
    results = {}

    results["total_rows"] = len(df)
    results["unique_geographies"] = df["state_ut"].nunique()

    # % columns 0-100 check
    pct_cols = [c for c in df.columns
                if c not in NON_PCT_COLS and c not in {"geography_type"}]
    out_of_range = {}
    for col in pct_cols:
        bad = df[col].dropna()
        bad = bad[(bad < 0) | (bad > 100)]
        if len(bad) > 0:
            out_of_range[col] = list(bad)
    results["pct_cols_out_of_range"] = out_of_range

    # Columns with suppressed cells
    id_set = {"state_ut", "area", "hh_surveyed", "women_interviewed", "geography_type"}
    suppressed_cols = [
        c for c in df.columns
        if c not in id_set and df[c].isna().any()
    ]
    results["suppressed_cols"] = suppressed_cols

    # India-Total row
    india_total = df[
        (df["state_ut"].str.strip().str.lower() == "india") &
        (df["area"].str.strip().str.lower() == "total")
    ]
    results["india_total"] = india_total

    return results


# ---------------------------------------------------------------------------
# 7.  NOTES
# ---------------------------------------------------------------------------

NOTES_TEXT = [
    ("Source", "NFHS-5 (2019-21) India & States Factsheet Data — original file: NFHS_5_Factsheets_Data.xls"),
    ("Sheet", "India & States (112 rows including header, 136 columns)"),
    ("Rows kept", "111 data rows (India + 36 States/UTs, each with Urban / Rural / Total)"),
    ("Geographies", "37 unique geographies (1 National + 36 States/UTs)"),
    ("Suppressed values",
     "'*' in original = suppressed due to too few cases. Stored as NaN/blank. "
     "NEVER treated as 0. See Dictionary.suppressed_count for counts per column."),
    ("Low-sample (negative) values",
     "Original fact-sheet values shown as '(xx.x)' for 25-49 unweighted cases were "
     "stored as negative numbers by Excel. Absolute value taken; Low_sample_flags "
     "sheet records a 1 for every such cell. See Dictionary.low_sample_count."),
    ("Spelling fix", "'Maharastra' corrected to 'Maharashtra' in state_ut column."),
    ("geography_type",
     "'National' for rows where state_ut = India; 'State/UT' for all other rows."),
    ("Column naming",
     "Short snake_case names used in Values and Low_sample_flags sheets. "
     "Original column text preserved in Dictionary.original_column."),
    ("Non-% columns",
     "oop_cost_public_delivery_rs (Rupees), neonatal/infant/under5_mortality_rate "
     "(per 1000 live births), total_fertility_rate (children per woman) are NOT "
     "percent columns and are not subject to the 0-100 validation rule."),
    ("Script", "Generated by build_nfhs5_dataset.py"),
]


# ---------------------------------------------------------------------------
# 8.  WRITE OUTPUT
# ---------------------------------------------------------------------------

def write_output(
    df: pd.DataFrame,
    flag_df: pd.DataFrame,
    dictionary: pd.DataFrame,
    out_xlsx: str,
    out_csv: str,
):
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # --- Notes sheet ---
        notes_df = pd.DataFrame(NOTES_TEXT, columns=["Key", "Value"])
        notes_df.to_excel(writer, sheet_name="Notes", index=False)

        # --- Dictionary sheet ---
        dictionary.to_excel(writer, sheet_name="Dictionary", index=False)

        # --- Values sheet ---
        df.to_excel(writer, sheet_name="Values", index=False)

        # --- Low_sample_flags sheet ---
        flag_df.to_excel(writer, sheet_name="Low_sample_flags", index=False)

    # --- CSV ---
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Wrote: {out_xlsx}")
    print(f"Wrote: {out_csv}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    src = "NFHS_5_Factsheets_Data.xls"
    out_xlsx = "NFHS5_clean.xlsx"
    out_csv  = "NFHS5_values.csv"

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Loading raw data ...")
    headers, rows = load_raw(src)

    print("Building selected columns ...")
    df_raw, flag_df_raw = build_df(headers, rows)

    print("Cleaning ...")
    df, flag_df = clean_df(df_raw, flag_df_raw)

    print("Building dictionary ...")
    dictionary = build_dictionary(headers, df, flag_df)

    print("Validating ...")
    v = validate(df, headers, rows)

    print("\n=== VALIDATION RESULTS ===")
    print(f"Total rows       : {v['total_rows']}  (expected 111)")
    print(f"Unique geographies: {v['unique_geographies']}  (expected 37)")

    if v["pct_cols_out_of_range"]:
        print(f"[WARN] % columns with out-of-range values: {list(v['pct_cols_out_of_range'].keys())}")
    else:
        print("[OK]  All % columns are in [0, 100]")

    print(f"\nColumns with suppressed (NaN) cells ({len(v['suppressed_cols'])}):")
    for c in v["suppressed_cols"]:
        n = int(df[c].isna().sum())
        print(f"   {c:<45s}  NaN={n}")

    print("\n=== INDIA-TOTAL ROW ===")
    it = v["india_total"]
    if len(it) == 0:
        print("  [WARN] India-Total row NOT found!")
    else:
        for col in df.columns:
            print(f"  {col:<45s}: {it[col].values[0]}")

    print("\nWriting output …")
    write_output(df, flag_df, dictionary, out_xlsx, out_csv)

    print("\nDone.")
