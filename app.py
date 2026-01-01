import io
import csv
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Auto Data Report (MVP)", layout="wide")


# ---------------------------
# Robust CSV loading
# ---------------------------
COMMON_SEPS = [",", ";", "\t", "|"]
COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


def sniff_delimiter(sample_text: str) -> str:
    """Try csv.Sniffer; fallback to counting separators."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(COMMON_SEPS))
        if dialect.delimiter in COMMON_SEPS:
            return dialect.delimiter
    except Exception:
        pass

    # fallback: choose sep with max count in first lines
    lines = sample_text.splitlines()[:10]
    counts = {}
    for sep in COMMON_SEPS:
        counts[sep] = sum(line.count(sep) for line in lines)
    best = max(counts, key=counts.get)
    return best


def try_read_csv_from_bytes(raw: bytes):
    """
    Returns: (df, meta) or (None, error_message)
    meta includes chosen encoding, sep, etc.
    """
    last_error = None

    # small sample for sniffing
    raw_head = raw[:100_000]

    for enc in COMMON_ENCODINGS:
        try:
            sample_text = raw_head.decode(enc, errors="strict")
        except Exception as e:
            last_error = e
            continue

        sep = sniff_delimiter(sample_text)

        # Try a couple of parsing strategies
        for engine in ["python", "c"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep,
                    encoding=enc,
                    engine=engine,
                    on_bad_lines="skip",  # important for messy CSVs
                )
                # If parsing produced a single column with lots of separators → wrong sep
                if df.shape[1] == 1 and any(s in df.columns[0] for s in COMMON_SEPS):
                    raise ValueError("Likely wrong delimiter (collapsed into 1 column).")

                meta = {
                    "encoding": enc,
                    "sep": sep,
                    "engine": engine,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "columns": list(df.columns),
                }
                return df, meta
            except Exception as e:
                last_error = e
                continue

    return None, f"Could not parse CSV. Last error: {last_error}"


def coerce_numeric(df: pd.DataFrame):
    """
    Try to convert object columns to numeric if they look like numbers.
    Returns: (df2, numeric_cols, categorical_cols)
    """
    df2 = df.copy()

    # Strip whitespace in column names
    df2.columns = [str(c).strip() for c in df2.columns]

    # Basic cleanup for object columns
    for col in df2.columns:
        if df2[col].dtype == "object":
            s = df2[col].astype(str).str.strip()

            # Try numeric conversion (handles "1,234" vs "1.234" a bit)
            s2 = s.str.replace(" ", "", regex=False)

            # If many commas and few dots → may be decimal comma, but in CSV often thousands separators.
            # We'll try both: first remove thousands commas, then parse.
            cand = pd.to_numeric(s2.str.replace(",", "", regex=False), errors="coerce")

            ratio = cand.notna().mean()
            if ratio >= 0.90:  # enough to treat as numeric
                df2[col] = cand

    numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    categorical_cols = [c for c in df2.columns if c not in numeric_cols]

    return df2, numeric_cols, categorical_cols


def detect_id_like(df: pd.DataFrame, numeric_cols: list[str]) -> set[str]:
    """
    Heuristic: numeric col is ID-like if:
    - very high uniqueness (>= 0.95) OR
    - name contains id/order/number/no
    """
    id_like = set()
    n = len(df)
    for c in numeric_cols:
        name = c.lower()
        uniq_ratio = df[c].nunique(dropna=True) / max(n, 1)
        if uniq_ratio >= 0.95:
            id_like.add(c)
        if any(k in name for k in ["id", "order", "number", "num", "no", "code"]):
            # don't blindly mark if it's clearly a measure, but usually ok for MVP
            id_like.add(c)
    return id_like


def pick_top_categories(df: pd.DataFrame, cat_cols: list[str]) -> str | None:
    """Pick a categorical column that is useful (not too unique, not empty)."""
    n = len(df)
    best = None
    best_score = -1
    for c in cat_cols:
        nun = df[c].nunique(dropna=True)
        if nun <= 1:
            continue
        # avoid almost-unique text columns (addresses, phone, names)
        uniq_ratio = nun / max(n, 1)
        if uniq_ratio > 0.7:
            continue
        score = nun  # simple
        if score > best_score:
            best_score = score
            best = c
    return best


# ---------------------------
# UI
# ---------------------------
st.title("Auto Data Report (MVP)")
st.caption("Upload a CSV → app auto-detects encoding & delimiter → KPIs + charts")

uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])

if not uploaded:
    st.info("Upload a CSV file to start.")
    st.stop()

raw = uploaded.getvalue()
df, meta_or_err = try_read_csv_from_bytes(raw)

if df is None:
    st.error(meta_or_err)
    st.stop()

meta = meta_or_err

# Confirm parsing
with st.expander("✅ Parsing confirmation (what the app understood)", expanded=True):
    st.write(
        {
            "file_name": uploaded.name,
            "encoding": meta["encoding"],
            "delimiter": meta["sep"],
            "engine": meta["engine"],
            "rows": meta["rows"],
            "columns": meta["cols"],
        }
    )

st.subheader("Preview")
st.dataframe(df.head(25), use_container_width=True)

# Coerce types
df2, numeric_cols, cat_cols = coerce_numeric(df)

missing_cells = int(df2.isna().sum().sum())

st.subheader("Basic stats")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df2.shape[0])
c2.metric("Columns", df2.shape[1])
c3.metric("Missing cells", missing_cells)

with st.expander("Detected columns (after type inference)"):
    st.write({"numeric_columns": numeric_cols, "categorical_columns": cat_cols})

# Remove ID-like from “KPI focus”
id_like = detect_id_like(df2, numeric_cols)
measure_cols = [c for c in numeric_cols if c not in id_like]

if len(measure_cols) == 0 and len(numeric_cols) > 0:
    # fallback: if everything looks like ID, still show numeric cols
    measure_cols = numeric_cols[:]

# KPI table for numeric
if len(numeric_cols) > 0:
    st.subheader("KPIs (numeric)")
    desc = df2[numeric_cols].describe().T  # count/mean/std/min/25%/50%/75%/max
    st.dataframe(desc, use_container_width=True)
else:
    st.warning("No numeric columns detected after inference. Charts will be limited.")

# Simple segmentation (top 2 measures)
st.subheader("Segmentation (simple quartiles)")
if len(measure_cols) >= 2:
    xcol, ycol = measure_cols[0], measure_cols[1]
    xq1, xq3 = df2[xcol].quantile(0.25), df2[xcol].quantile(0.75)
    yq1, yq3 = df2[ycol].quantile(0.25), df2[ycol].quantile(0.75)

    def bucket(v, q1, q3):
        if pd.isna(v):
            return "NA"
        if v < q1:
            return "LOW"
        if v > q3:
            return "HIGH"
        return "MID"

    seg = df2[[xcol, ycol]].copy()
    seg["X_SEG"] = seg[xcol].apply(lambda v: bucket(v, xq1, xq3))
    seg["Y_SEG"] = seg[ycol].apply(lambda v: bucket(v, yq1, yq3))
    seg["SEGMENT"] = seg["X_SEG"] + "_" + seg["Y_SEG"]
    seg_counts = seg["SEGMENT"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    seg_counts["pct"] = (seg_counts["count"] / len(df2) * 100).round(2)

    st.write(f"Using: **{xcol}** x **{ycol}**")
    st.dataframe(seg_counts, use_container_width=True)
else:
    st.info("Need at least 2 numeric measure-like columns to run segmentation.")

# Charts
st.subheader("Charts")

charts = []

# 1-2 distributions for measures
for col in measure_cols[:2]:
    fig = plt.figure()
    plt.hist(df2[col].dropna(), bins=20)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    charts.append(fig)

# scatter for top 2 measures
if len(measure_cols) >= 2:
    xcol, ycol = measure_cols[0], measure_cols[1]
    fig = plt.figure()
    plt.scatter(df2[xcol], df2[ycol], s=10)
    plt.title(f"{xcol} vs {ycol}")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    charts.append(fig)

# bar for best categorical
best_cat = pick_top_categories(df2, cat_cols)
if best_cat is not None:
    vc = df2[best_cat].astype(str).value_counts().head(10)
    fig = plt.figure()
    plt.bar(vc.index, vc.values)
    plt.title(f"Top categories: {best_cat}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    charts.append(fig)

# show up to 5 charts
for i, fig in enumerate(charts[:5], start=1):
    st.pyplot(fig)

# Conclusion
st.subheader("Conclusion (EN)")
rows, cols = df2.shape
msg = f"This dataset contains {rows} rows and {cols} columns. "
if len(measure_cols) >= 2:
    msg += f"Key numeric measures include {measure_cols[0]} and {measure_cols[1]}. "
elif len(numeric_cols) >= 1:
    msg += f"At least one numeric column was detected ({numeric_cols[0]}). "
if best_cat:
    msg += f"A main categorical dimension is {best_cat}. "
msg += "The report highlights basic quality signals, descriptive KPIs, and simple visual patterns."
st.write(msg)
