import io
import re
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------
# 1) Robust CSV loader (encoding + delimiter)
# ---------------------------

COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
COMMON_DELIMS = [",", ";", "\t", "|"]


def sniff_delimiter(sample_text: str) -> str:
    """
    Try csv.Sniffer first, fallback to simple counts.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=COMMON_DELIMS)
        return dialect.delimiter
    except Exception:
        # fallback: choose delimiter with max count
        counts = {d: sample_text.count(d) for d in COMMON_DELIMS}
        best = max(counts, key=counts.get)
        # if all zero -> default comma
        return best if counts[best] > 0 else ","


def read_csv_robust(uploaded_file) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns (df, meta) where meta contains encoding/delimiter guesses.
    """
    raw = uploaded_file.getvalue()
    # Take a small sample for sniffing delimiter
    sample_bytes = raw[:65536]

    used_encoding = None
    last_err = None

    for enc in COMMON_ENCODINGS:
        try:
            sample_text = sample_bytes.decode(enc, errors="strict")
            used_encoding = enc
            break
        except Exception as e:
            last_err = e

    if used_encoding is None:
        # last resort: decode with utf-8 ignoring errors
        used_encoding = "utf-8 (errors=replace)"
        sample_text = sample_bytes.decode("utf-8", errors="replace")

    delim = sniff_delimiter(sample_text)

    # Now read full file with chosen encoding
    if used_encoding.endswith("(errors=replace)"):
        text = raw.decode("utf-8", errors="replace")
        buffer = io.StringIO(text)
        df = pd.read_csv(buffer, sep=delim)
    else:
        buffer = io.BytesIO(raw)
        df = pd.read_csv(buffer, sep=delim, encoding=used_encoding)

    meta = {
        "filename": uploaded_file.name,
        "encoding": used_encoding,
        "delimiter": delim,
        "bytes": len(raw),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }
    return df, meta


# ---------------------------
# 2) Column role detection + confirmation step
# ---------------------------

ROLE_OPTIONS = ["id", "measure", "categorical", "datetime", "geo", "ignore"]


GEO_KEYWORDS = [
    "country", "state", "region", "city", "county", "province", "zip", "postal",
    "lat", "latitude", "lon", "lng", "longitude"
]


def normalize_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", c.strip().lower()).strip()


def try_parse_datetime(series: pd.Series) -> float:
    """
    Returns parse success ratio (0..1).
    """
    if series.dtype.kind in "Mm":
        return 1.0
    s = series.dropna().astype(str).head(500)
    if len(s) == 0:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    return float(parsed.notna().mean())


def is_probably_id(series: pd.Series, colname: str) -> bool:
    """
    Heuristics: mostly unique OR monotonic integer-like, and not a good measure.
    """
    s = series.dropna()
    if len(s) == 0:
        return False

    n = len(s)
    nunique = s.nunique(dropna=True)
    uniq_ratio = nunique / max(n, 1)

    # If name suggests ID/order/code
    name = normalize_colname(colname)
    if any(k in name for k in ["id", "uuid", "order", "number", "no", "code", "key"]):
        # allow if very unique or integer-ish
        if uniq_ratio > 0.6:
            return True

    # Pure uniqueness
    if uniq_ratio > 0.95:
        return True

    # Monotonic integer-like
    if pd.api.types.is_numeric_dtype(s):
        ss = pd.to_numeric(s, errors="coerce").dropna()
        if len(ss) > 0:
            # check if almost all integers
            int_ratio = float((np.isclose(ss % 1, 0)).mean())
            if int_ratio > 0.98 and uniq_ratio > 0.6:
                return True

    return False


def detect_roles(df: pd.DataFrame) -> Dict[str, str]:
    roles = {}
    for col in df.columns:
        s = df[col]
        name = normalize_colname(col)

        # geo by name hint
        if any(k in name for k in GEO_KEYWORDS):
            roles[col] = "geo"
            continue

        # datetime by parse success
        dt_ratio = try_parse_datetime(s)
        if dt_ratio >= 0.8:
            roles[col] = "datetime"
            continue

        # numeric?
        if pd.api.types.is_numeric_dtype(s):
            # id?
            if is_probably_id(s, col):
                roles[col] = "id"
            else:
                roles[col] = "measure"
            continue

        # non-numeric: maybe id if very unique
        if is_probably_id(s.astype(str), col):
            roles[col] = "id"
            continue

        # otherwise categorical
        roles[col] = "categorical"

    return roles


# ---------------------------
# 3) Dataset type detection (simple)
# ---------------------------

def detect_dataset_type(roles: Dict[str, str]) -> str:
    has_dt = any(r == "datetime" for r in roles.values())
    has_geo = any(r == "geo" for r in roles.values())
    measures = [c for c, r in roles.items() if r == "measure"]
    cats = [c for c, r in roles.items() if r == "categorical"]
    ids = [c for c, r in roles.items() if r == "id"]

    if has_dt and (len(cats) + len(ids)) > 0 and len(measures) > 0:
        return "PANEL"   # time + entity + measure
    if has_dt and len(measures) > 0:
        return "TIME_SERIES"
    if has_geo and len(measures) > 0:
        return "GEO"
    if len(measures) >= 1:
        return "SNAPSHOT"
    return "REGISTRY"


# ---------------------------
# 4) Analysis focus (direction) -> chooses KPIs and charts
# ---------------------------

FOCUS_OPTIONS = {
    "Business (generic)": "business",
    "Sales / Revenue": "sales",
    "Finance (balances, risk)": "finance",
    "Operations / Process": "ops",
    "Time patterns": "time",
    "Geography": "geo",
}


def pick_measure_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r == "measure"]


def pick_datetime_column(roles: Dict[str, str]) -> Optional[str]:
    for c, r in roles.items():
        if r == "datetime":
            return c
    return None


def pick_categorical_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r == "categorical"]


def pick_geo_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r == "geo"]


def numeric_kpis(df: pd.DataFrame, measures: List[str]) -> pd.DataFrame:
    if not measures:
        return pd.DataFrame()
    return df[measures].describe().T  # count/mean/std/min/25/50/75/max


def make_hist(df: pd.DataFrame, col: str, title: str):
    fig = plt.figure()
    plt.hist(df[col].dropna(), bins=25)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)


def make_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    fig = plt.figure()
    plt.scatter(df[x], df[y], s=10)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot(fig)
    plt.close(fig)


def make_bar_top(df: pd.DataFrame, col: str, title: str, top_n: int = 10):
    vc = df[col].astype(str).value_counts().head(top_n)
    fig = plt.figure()
    plt.bar(vc.index, vc.values)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def make_time_line(df: pd.DataFrame, dt_col: str, measure: str, title: str):
    tmp = df[[dt_col, measure]].copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        st.info("No valid datetime rows to plot.")
        return
    tmp = tmp.set_index(dt_col).sort_index()
    # daily aggregation to keep it simple/robust
    daily = tmp[measure].resample("D").sum()

    fig = plt.figure()
    plt.plot(daily.index, daily.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(measure)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def make_segment_3x3(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """
    Quartile-based 3x3 segmentation labels.
    LOW: <= q25, MID: (q25..q75), HIGH: >= q75
    """
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    tmp = pd.DataFrame({a: x, b: y}).dropna()
    if tmp.empty:
        return pd.DataFrame()

    qx1, qx3 = tmp[a].quantile(0.25), tmp[a].quantile(0.75)
    qy1, qy3 = tmp[b].quantile(0.25), tmp[b].quantile(0.75)

    def bucket(v, q1, q3):
        if v <= q1:
            return "LOW"
        if v >= q3:
            return "HIGH"
        return "MID"

    tmp["seg_x"] = tmp[a].apply(lambda v: bucket(v, qx1, qx3))
    tmp["seg_y"] = tmp[b].apply(lambda v: bucket(v, qy1, qy3))
    tmp["segment"] = tmp["seg_x"] + "_" + tmp["seg_y"]

    out = tmp["segment"].value_counts().reset_index()
    out.columns = ["segment", "count"]
    out["pct"] = (out["count"] / out["count"].sum() * 100).round(2)
    return out.sort_values("count", ascending=False)


def choose_analysis(df: pd.DataFrame, roles: Dict[str, str], focus: str):
    measures = pick_measure_columns(roles)
    cats = pick_categorical_columns(roles)
    dt_col = pick_datetime_column(roles)
    geo_cols = pick_geo_columns(roles)

    # KPI table
    st.subheader("KPIs (numeric)")
    kpi = numeric_kpis(df, measures)
    if kpi.empty:
        st.info("No numeric measures detected.")
    else:
        st.dataframe(kpi)

    # Charts (always 5, best-effort)
    st.subheader("Charts (5)")
    charts_done = 0

    # 1) two hists of top measures
    for m in measures[:2]:
        make_hist(df, m, f"Distribution: {m}")
        charts_done += 1

    # 2) scatter of first two measures
    if charts_done < 5 and len(measures) >= 2:
        make_scatter(df, measures[0], measures[1], f"{measures[0]} vs {measures[1]}")
        charts_done += 1

    # 3) segmentation (3x3) if measures >= 2
    if charts_done < 5 and len(measures) >= 2:
        seg = make_segment_3x3(df, measures[0], measures[1])
        st.write("Segmentation (quartile 3x3):")
        if seg.empty:
            st.info("Segmentation not available (not enough numeric data).")
        else:
            st.dataframe(seg)
        charts_done += 1  # counts as one “chart block”

    # 4) categorical top
    if charts_done < 5 and len(cats) >= 1:
        make_bar_top(df, cats[0], f"Top categories: {cats[0]}")
        charts_done += 1

    # 5) time series line if datetime exists
    if charts_done < 5 and dt_col and len(measures) >= 1:
        make_time_line(df, dt_col, measures[0], f"Time trend (daily sum): {measures[0]}")
        charts_done += 1

    # Fill remaining slots with additional hists or category bars
    i = 2
    while charts_done < 5 and i < len(measures):
        make_hist(df, measures[i], f"Distribution: {measures[i]}")
        charts_done += 1
        i += 1

    j = 1
    while charts_done < 5 and j < len(cats):
        make_bar_top(df, cats[j], f"Top categories: {cats[j]}")
        charts_done += 1
        j += 1

    # If still not enough, just show info blocks
    while charts_done < 5:
        st.info("Not enough diverse columns to build more charts.")
        charts_done += 1

    # Conclusion EN (short)
    st.subheader("Conclusion (EN)")
    ds_type = detect_dataset_type(roles)
    st.write(
        f"This dataset looks like a **{ds_type}** table with "
        f"{df.shape[0]} rows and {df.shape[1]} columns. "
        f"The report summarizes numeric KPIs, key distributions, and the most visible patterns "
        f"based on the confirmed column roles and selected focus."
    )


# ---------------------------
# Streamlit UI: 2-step confirmation
# ---------------------------

st.set_page_config(page_title="Auto Data Report (MVP)", layout="wide")
st.title("Auto Data Report (MVP)")
st.caption("Upload a CSV → confirm detected column roles → choose analysis direction → see KPIs and charts")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded:
    st.info("Upload a CSV file to start.")
    st.stop()

# Step A: Load robustly
df, meta = read_csv_robust(uploaded)

st.subheader("Loaded file")
st.write(meta)

st.subheader("Preview")
st.dataframe(df.head(30))

# Step B: Auto-detect roles, then let user confirm/override
st.subheader("Step 1 — Detected column roles (confirm / edit)")

auto_roles = detect_roles(df)

col1, col2 = st.columns([2, 1])

with col2:
    focus_label = st.selectbox("Step 2 — What direction do you want?", list(FOCUS_OPTIONS.keys()))
    focus = FOCUS_OPTIONS[focus_label]

with col1:
    # editable roles table (simple UI: per-column selectbox)
    confirmed_roles = {}
    for col in df.columns:
        default = auto_roles.get(col, "categorical")
        confirmed_roles[col] = st.selectbox(
            f"{col}",
            ROLE_OPTIONS,
            index=ROLE_OPTIONS.index(default) if default in ROLE_OPTIONS else ROLE_OPTIONS.index("categorical"),
            key=f"role_{col}",
        )

# Apply ignore role (drop)
use_cols = [c for c in df.columns if confirmed_roles.get(c) != "ignore"]
df2 = df[use_cols].copy()
roles2 = {c: r for c, r in confirmed_roles.items() if c in use_cols}

st.subheader("Confirmed dataset type")
st.write(detect_dataset_type(roles2))

# Step C: Produce results
st.divider()
st.header("Report")
choose_analysis(df2, roles2, focus)
