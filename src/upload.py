"""
  Upload & cleaning.
• Upload Excel → canonicalise column names → store in session_state.
• Exposes _clean() + add_rankings() for ranking.py
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────
CANONICAL_MAP = {
    # map original → canonical
    "YEAR":               "year",
    "PAYOR_NAME":         "payor_name",
    "PAYOR_STATE":        "payor_state",
    "BUSINESS_TYPE":      "business_type",
    "PREMIUM":            "premium",
    "POPULATION":         "population",
    "MARKET_SHARE":       "market_share",
    "LINE_OF_BUSINESS":   "line_of_business",
    "UNIQUE_MEMBER_COUNT":"unique_member_count",
    "UNIQUE_PROVIDER_COUNT":"unique_provider_count",
    "UNIQUE_CLAIM_COUNT": "unique_claim_count",
    "UNIQUE_CLAIMANT_COUNT":"unique_claimant_count",
    "MEDICAL_COST":       "medical_cost",
    "ADMINISTRATIVE_COST":"administrative_cost",
    # …add more if you need them later
}

def _canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to lower‑case snake_case so the rest of the code is agnostic
    to how Excel names things.
    """
    mapping = {c: CANONICAL_MAP.get(c, c.lower()) for c in df.columns}
    return df.rename(columns=mapping)

# ------------------------------------------------------------------
# Cleaning helper
# ------------------------------------------------------------------
def _clean(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    df = raw[raw["year"] == year].copy()

    # drop dups on key business columns
    df = df.drop_duplicates(subset=[
        "payor_name", "payor_state", "business_type", "premium", "population"
    ])

    num_cols = [
        "premium", "population",
        "unique_member_count", "unique_provider_count",
        "unique_claim_count", "unique_claimant_count",
    ]
    if "market_share" in df.columns:
        df["market_share"] = (
            df["market_share"].astype(str)
              .str.rstrip("%").str.replace(",", "").astype(float) / 100
        )
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True).astype(float)

    # binary flags
    df["type_L"]        = df["business_type"].str.contains("L", na=False, case=False).astype(int)
    df["type_H"]        = df["business_type"].str.contains("H", na=False, case=False).astype(int)
    df["is_commercial"] = df["line_of_business"].str.contains("commercial", na=False, case=False).astype(int)
    df["is_medicare"]   = df["line_of_business"].str.contains("medicare",   na=False, case=False).astype(int)
    df["is_medicaid"]   = df["line_of_business"].str.contains("medicaid",  na=False, case=False).astype(int)

    return df.dropna(how="all")   # keep rows with at least one non‑NA

# ------------------------------------------------------------------
# In‑cluster ranking reused by ranking.py
# ------------------------------------------------------------------
def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    # recognise the cost‑columns the file actually contains
    for mc_col, ac_col in [
        ("medical_cost", "administrative_cost"),          # ← lower‑case
        ("MEDICAL_COST", "ADMINISTRATIVE_COST"),          # ← upper‑case
        ("Medical Cost (MC)", "Administrative Cost (AC)"),  # ← legacy
        ("MC", "AC"),                                     # ← fallback
    ]:
        if {mc_col, ac_col}.issubset(df.columns):
            break
    else:
        return df                      # nothing to rank

    out = df.copy()
    for cid, idx in out.groupby("cluster").groups.items():
        sub = out.loc[idx]

        # ✅ pass the method **by name**
        out.loc[idx, "MC_rank"] = sub[mc_col].rank(method="max",
                                                   ascending=True).astype(int)
        out.loc[idx, "AC_rank"] = sub[ac_col].rank(method="min",
                                                   ascending=True).astype(int)
        out.loc[idx, "rank_score"] = out.loc[idx, "MC_rank"] * out.loc[idx, "AC_rank"]
        out.loc[idx, "rank"]       = out.loc[idx, "rank_score"].rank(method="min").astype(int)

    return out


# ------------------------------------------------------------------
# Streamlit page: upload
# ------------------------------------------------------------------
def page() -> None:
    # Add minimal CSS for clean upload experience
    st.markdown("""
    <style>
    /* Simple banner styling */
    .banner-container {
        background: linear-gradient(135deg, #4A148C 0%, #6A1B9A 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .banner-logo {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #E1BEE7;
    }
    
    .banner-tagline {
        font-size: 1rem;
        color: #F8BBD9;
    }
    
    /* Simple button styling */
    .stButton > button {
        border-radius: 8px;
        background: #6B7280;
        color: white;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: #4B5563;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    

    
    st.header("📁 Upload your dataset")
    
    # Simple file uploader
    file = st.file_uploader("Select Excel file (.xlsx / .xls)", type=["xlsx", "xls"])
    
    if file is None:
        return

    # Show processing
    with st.spinner("Processing your data..."):
        try:
            raw = pd.read_excel(file)
            raw = _canonicalise(raw)
        except Exception as e:
            st.error(f"❌ Could not read Excel – {e}")
            return

    if raw.empty:
        st.error("The uploaded file seems to have no rows.")
        return

    must_have = ["year", "payor_name", "payor_state", "business_type", "premium", "population"]
    missing = [c for c in must_have if c not in raw.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.write("Available columns:", list(raw.columns))
        return

    # Save for the rest of the app
    st.session_state.df_raw = raw
    
    # Simple success message
    st.success(f"✅ {len(raw):,} rows loaded successfully.")
    
    # Simple data preview
    st.subheader("Data Preview")
    st.dataframe(raw.head(), use_container_width=True)
    st.rerun()