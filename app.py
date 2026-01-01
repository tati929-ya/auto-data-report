import io
import re
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# ---------------------------
# CSV loader (encoding + delimiter)
# ---------------------------

COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
COMMON_DELIMS = [",", ";", "\t", "|"]


def sniff_delimiter(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=COMMON_DELIMS)
        return dialect.delimiter
    except Exception:
        counts = {d: sample_text.count(d) for d in COMMON_DELIMS}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def read_csv_robust(uploaded_file) -> Tuple[pd.DataFrame, Dict]:
    raw = uploaded_file.getvalue()
    sample_bytes = raw[:65536]

    used_encoding = None
    sample_text = None

    for enc in COMMON_ENCODINGS:
        try:
            sample_text = sample_bytes.decode(enc, errors="strict")
            used_encoding = enc
            break
        except Exception:
            continue

    if used_encoding is None:
        used_encoding = "utf-8 (errors=replace)"
        sample_text = sample_bytes.decode("utf-8", errors="replace")

    delim = sniff_delimiter(sample_text)

    if used_encoding.endswith("(errors=replace)"):
        text = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), sep=delim)
    else:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, encoding=used_encoding)

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
# Roles
# ---------------------------

ROLE_OPTIONS = ["id", "measure", "categorical", "datetime", "geo", "ignore"]

ROLE_HELP = {
    "measure": "Numeric value you want to analyze (avg, sum, min/max). Examples: sales, revenue, temperature, balance.",
    "categorical": "Groups/labels for comparison. Examples: gender, product_line, status, region, department.",
    "datetime": "Time column used for trends. Examples: date, timestamp, order_date.",
    "geo": "Geography fields. Examples: country, city, state, latitude/longitude.",
    "id": "Identifier (usually NOT analyzed). Examples: order_id, customer_id, transaction_id, code.",
    "ignore": "Drop this column from analysis (not useful / noise).",
}


GEO_KEYWORDS = [
    "country", "state", "region", "city", "county", "province",
    "zip", "postal", "postcode", "lat", "latitude", "lon", "lng", "longitude"
]


def normalize_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", c.strip().lower()).strip()


def try_parse_datetime(series: pd.Series) -> float:
    if series.dtype.kind in "Mm":
        return 1.0
    s = series.dropna().astype(str).head(500)
    if len(s) == 0:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    return float(parsed.notna().mean())


def is_probably_id(series: pd.Series, colname: str) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False

    n = len(s)
    nunique = s.nunique(dropna=True)
    uniq_ratio = nunique / max(n, 1)

    name = normalize_colname(colname)
    name_hint = any(k in name for k in ["id", "uuid", "order", "number", "no", "code", "key"])
    if name_hint and uniq_ratio > 0.4:
        return True

    if uniq_ratio > 0.95:
        return True

    if pd.api.types.is_numeric_dtype(s):
        ss = pd.to_numeric(s, errors="coerce").dropna()
        if len(ss) > 0:
            int_ratio = float((np.isclose(ss % 1, 0)).mean())
            if int_ratio > 0.98 and uniq_ratio > 0.6:
                return True

    return False


def detect_roles(df: pd.DataFrame) -> Dict[str, str]:
    roles = {}
    for col in df.columns:
        s = df[col]
        name = normalize_colname(col)

        if any(k in name for k in GEO_KEYWORDS):
            roles[col] = "geo"
            continue

        dt_ratio = try_parse_datetime(s)
        if dt_ratio >= 0.8:
            roles[col] = "datetime"
            continue

        if pd.api.types.is_numeric_dtype(s):
            roles[col] = "id" if is_probably_id(s, col) else "measure"
            continue

        if is_probably_id(s.astype(str), col):
            roles[col] = "id"
            continue

        roles[col] = "categorical"

    return roles


def pick_measure_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r == "measure"]


def pick_categorical_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r == "categorical"]


def pick_datetime_column(roles: Dict[str, str]) -> Optional[str]:
    for c, r in roles.items():
        if r == "datetime":
            return c
    return None


def pick_geo_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r == "geo"]


def detect_dataset_type(roles: Dict[str, str]) -> str:
    has_dt = any(r == "datetime" for r in roles.values())
    has_geo = any(r == "geo" for r in roles.values())
    measures = [c for c, r in roles.items() if r == "measure"]
    cats_ids = [c for c, r in roles.items() if r in ("categorical", "id")]

    if has_dt and measures and cats_ids:
        return "PANEL"
    if has_dt and measures:
        return "TIME_SERIES"
    if has_geo and measures:
        return "GEO"
    if measures:
        return "SNAPSHOT"
    return "REGISTRY"


# ---------------------------
# Analysis & charts (direction-dependent)
# ---------------------------

FOCUS_OPTIONS = {
    "Business (generic)": "business",
    "Sales / Revenue": "sales",
    "Finance (balances, risk)": "finance",
    "Operations / Process": "ops",
    "Time patterns": "time",
    "Geography": "geo",
}

def numeric_kpis(df: pd.DataFrame, measures: List[str]) -> pd.DataFrame:
    if not measures:
        return pd.DataFrame()
    return df[measures].describe().T


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def chart_hist(df: pd.DataFrame, col: str, title: str):
    fig = plt.figure()
    plt.hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=25)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


def chart_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    fig = plt.figure()
    xx = pd.to_numeric(df[x], errors="coerce")
    yy = pd.to_numeric(df[y], errors="coerce")
    m = xx.notna() & yy.notna()
    plt.scatter(xx[m], yy[m], s=10)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


def chart_bar_top(df: pd.DataFrame, col: str, title: str, top_n: int = 10):
    vc = df[col].astype(str).value_counts().head(top_n)
    fig = plt.figure()
    plt.bar(vc.index, vc.values)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


def chart_time_line(df: pd.DataFrame, dt_col: str, measure: str, title: str):
    tmp = df[[dt_col, measure]].copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp[measure] = pd.to_numeric(tmp[measure], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        st.info("No valid datetime rows to plot.")
        return None
    tmp = tmp.set_index(dt_col).sort_index()
    daily = tmp[measure].resample("D").sum()

    fig = plt.figure()
    plt.plot(daily.index, daily.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(measure)
    plt.tight_layout()
    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


def segmentation_3x3(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
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


def choose_primary_measure(measures: List[str], focus: str) -> Optional[str]:
    if not measures:
        return None

    name = [m.lower() for m in measures]
    # focus-based hints
    if focus in ("sales", "business"):
        for key in ["sales", "revenue", "amount", "income", "turnover", "profit", "total"]:
            for i, m in enumerate(name):
                if key in m:
                    return measures[i]
    if focus == "finance":
        for key in ["balance", "debt", "loan", "credit", "risk", "asset", "equity"]:
            for i, m in enumerate(name):
                if key in m:
                    return measures[i]
    # fallback: first measure
    return measures[0]


def build_report(df: pd.DataFrame, roles: Dict[str, str], focus: str):
    """
    Renders Streamlit report + returns PDF parts:
      - kpi_df
      - seg_df (optional)
    and list of chart PNG bytes (up to 5).
    """
    measures = pick_measure_columns(roles)
    cats = pick_categorical_columns(roles)
    dt_col = pick_datetime_column(roles)
    geo_cols = pick_geo_columns(roles)

    ds_type = detect_dataset_type(roles)

    st.subheader("Basic stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", int(df.isna().sum().sum()))

    st.write("Detected dataset type:", ds_type)

    st.subheader("KPIs (numeric)")
    kpi_df = numeric_kpis(df, measures)
    if kpi_df.empty:
        st.info("No numeric measures detected.")
    else:
        st.dataframe(kpi_df)

    # charts: direction-dependent priority
    st.subheader("Charts (5)")
    chart_images: List[bytes] = []
    charts_done = 0

    primary = choose_primary_measure(measures, focus)

    # 1) If time focus and datetime exists: time trend first
    if charts_done < 5 and focus in ("time", "sales", "business", "finance") and dt_col and primary:
        b = chart_time_line(df, dt_col, primary, f"Time trend (daily sum): {primary}")
        if b:
            chart_images.append(b)
            charts_done += 1

    # 2) Geography focus: totals by geo (pick first geo)
    if charts_done < 5 and focus == "geo" and geo_cols and primary:
        g = geo_cols[0]
        tmp = df[[g, primary]].copy()
        tmp[primary] = pd.to_numeric(tmp[primary], errors="coerce")
        tmp = tmp.dropna()
        if not tmp.empty:
            top = tmp.groupby(g)[primary].sum().sort_values(ascending=False).head(10)
            fig = plt.figure()
            plt.bar(top.index.astype(str), top.values)
            plt.title(f"Top {g} by total {primary}")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel(primary)
            plt.tight_layout()
            st.pyplot(fig)
            chart_images.append(fig_to_png_bytes(fig))
            plt.close(fig)
            charts_done += 1

    # 3) two histograms (most useful measures)
    for m in measures[:2]:
        if charts_done >= 5:
            break
        chart_images.append(chart_hist(df, m, f"Distribution: {m}"))
        charts_done += 1

    # 4) scatter of 1st vs 2nd measure
    if charts_done < 5 and len(measures) >= 2:
        chart_images.append(chart_scatter(df, measures[0], measures[1], f"{measures[0]} vs {measures[1]}"))
        charts_done += 1

    # 5) segmentation table (counts as one “visual block”)
    seg_df = pd.DataFrame()
    if charts_done < 5 and len(measures) >= 2 and focus in ("sales", "business", "finance", "ops"):
        st.write("Segmentation (quartile 3x3):")
        seg_df = segmentation_3x3(df, measures[0], measures[1])
        if seg_df.empty:
            st.info("Segmentation not available.")
        else:
            st.dataframe(seg_df)
        # no image, but still counts as “slot”
        charts_done += 1

    # 6) categorical bar
    if charts_done < 5 and cats:
        chart_images.append(chart_bar_top(df, cats[0], f"Top categories: {cats[0]}"))
        charts_done += 1

    # fill leftovers
    i = 2
    while charts_done < 5 and i < len(measures):
        chart_images.append(chart_hist(df, measures[i], f"Distribution: {measures[i]}"))
        charts_done += 1
        i += 1

    j = 1
    while charts_done < 5 and j < len(cats):
        chart_images.append(chart_bar_top(df, cats[j], f"Top categories: {cats[j]}"))
        charts_done += 1
        j += 1

    while charts_done < 5:
        st.info("Not enough diverse columns to build more charts.")
        charts_done += 1

    st.subheader("Conclusion (EN)")
    st.write(
        f"This dataset looks like a **{ds_type}** table with {df.shape[0]} rows and {df.shape[1]} columns. "
        f"The report summarizes numeric KPIs and the most visible patterns based on the confirmed column roles "
        f"and the selected focus (**{focus}**)."
    )

    return {
        "dataset_type": ds_type,
        "kpi_df": kpi_df,
        "seg_df": seg_df,
        "chart_images": chart_images,
        "conclusion": (
            f"This dataset looks like a {ds_type} table with {df.shape[0]} rows and {df.shape[1]} columns. "
            f"The report highlights numeric KPIs and visible patterns based on confirmed roles and focus: {focus}."
        ),
    }


# ---------------------------
# PDF generation
# ---------------------------

def pdf_from_report(meta: Dict, roles: Dict[str, str], focus_label: str, report: Dict) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    def header(title: str, y: float) -> float:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, y, title)
        return y - 0.8 * cm

    def text_line(line: str, y: float, size: int = 10) -> float:
        c.setFont("Helvetica", size)
        c.drawString(2 * cm, y, line[:140])
        return y - 0.5 * cm

    # Page 1: meta + roles + conclusion
    y = h - 2 * cm
    y = header("Auto Data Report (MVP)", y)

    y = text_line(f"File: {meta.get('filename')}", y)
    y = text_line(f"Encoding: {meta.get('encoding')} | Delimiter: {meta.get('delimiter')}", y)
    y = text_line(f"Rows: {meta.get('rows')} | Columns: {meta.get('cols')}", y)
    y = text_line(f"Focus: {focus_label}", y)
    y = text_line(f"Dataset type: {report.get('dataset_type')}", y)
    y -= 0.3 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Confirmed roles:")
    y -= 0.6 * cm
    c.setFont("Helvetica", 9)
    for col, r in roles.items():
        line = f"- {col}: {r}"
        c.drawString(2.2 * cm, y, line[:160])
        y -= 0.45 * cm
        if y < 2.2 * cm:
            c.showPage()
            y = h - 2 * cm
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2 * cm, y, "Confirmed roles (cont.):")
            y -= 0.8 * cm
            c.setFont("Helvetica", 9)

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Conclusion (EN):")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    for chunk in wrap_text(report.get("conclusion", ""), 95):
        c.drawString(2 * cm, y, chunk)
        y -= 0.55 * cm

    # Page 2+: KPIs table (best-effort) + charts
    c.showPage()
    y = h - 2 * cm
    y = header("KPIs (numeric)", y)

    kpi_df: pd.DataFrame = report.get("kpi_df", pd.DataFrame())
    if kpi_df is None or kpi_df.empty:
        y = text_line("No numeric KPIs available.", y)
    else:
        # small table rendering
        kpi_small = kpi_df.copy()
        kpi_small = kpi_small[["count", "mean", "std", "min", "50%", "max"]].round(3)
        rows = [("field",) + tuple(kpi_small.columns)]
        for idx, row in kpi_small.iterrows():
            rows.append((str(idx),) + tuple(str(v) for v in row.values))

        c.setFont("Helvetica", 8)
        col_x = [2 * cm, 7.5 * cm, 10.2 * cm, 12.6 * cm, 15.0 * cm, 17.2 * cm, 19.3 * cm]
        # header
        for i, val in enumerate(rows[0]):
            c.drawString(col_x[i], y, val[:18])
        y -= 0.5 * cm
        for r in rows[1:31]:  # limit to keep PDF readable
            for i, val in enumerate(r):
                c.drawString(col_x[i], y, str(val)[:18])
            y -= 0.42 * cm
            if y < 2.0 * cm:
                c.showPage()
                y = h - 2 * cm
                c.setFont("Helvetica-Bold", 12)
                c.drawString(2 * cm, y, "KPIs (cont.)")
                y -= 0.8 * cm
                c.setFont("Helvetica", 8)

    # Charts pages
    imgs: List[bytes] = report.get("chart_images", [])
    for img_bytes in imgs[:5]:
        c.showPage()
        y = h - 2 * cm
        y = header("Chart", y)
        img = ImageReader(io.BytesIO(img_bytes))
        # Fit image to page
        max_w = w - 4 * cm
        max_h = h - 5 * cm
        c.drawImage(img, 2 * cm, 2 * cm, width=max_w, height=max_h, preserveAspectRatio=True, anchor='c')

    c.save()
    buf.seek(0)
    return buf.getvalue()


def wrap_text(text: str, width: int) -> List[str]:
    words = text.split()
    lines = []
    cur = []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > width:
            cur.pop()
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


# ---------------------------
# Streamlit App (2 steps)
# ---------------------------

st.set_page_config(page_title="Auto Data Report (MVP)", layout="wide")
st.title("Auto Data Report (MVP)")
st.caption("Step 1: Upload + confirm roles → Step 2: Choose direction + report + PDF")

if "step" not in st.session_state:
    st.session_state.step = 1
if "df" not in st.session_state:
    st.session_state.df = None
if "meta" not in st.session_state:
    st.session_state.meta = None
if "roles" not in st.session_state:
    st.session_state.roles = None
if "focus" not in st.session_state:
    st.session_state.focus = "business"
if "focus_label" not in st.session_state:
    st.session_state.focus_label = "Business (generic)"


# STEP 1
if st.session_state.step == 1:
    st.header("Step 1 — Upload & confirm column roles")

    with st.expander("What do these roles mean? (help)", expanded=True):
        st.markdown(
            "**measure** — numeric values to analyze (avg/sum/min/max)\n\n"
            "**categorical** — groups/labels to compare\n\n"
            "**datetime** — date/time column for trends\n\n"
            "**geo** — geography columns (country/city/lat/lon)\n\n"
            "**id** — identifiers (usually not used in stats)\n\n"
            "**ignore** — remove from analysis\n"
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV file to continue.")
        st.stop()

    df, meta = read_csv_robust(uploaded)
    st.subheader("Loaded file")
    st.write(meta)

    st.subheader("Preview")
    st.dataframe(df.head(30))

    auto_roles = detect_roles(df)

    st.subheader("Detected roles (edit if needed)")
    edited_roles = {}
    for col in df.columns:
        default = auto_roles.get(col, "categorical")
        edited_roles[col] = st.selectbox(
            f"{col}",
            ROLE_OPTIONS,
            index=ROLE_OPTIONS.index(default),
            help=ROLE_HELP.get(default, ""),
            key=f"role_{col}",
        )

    # Confirm button
    if st.button("✅ OK, roles confirmed"):
        use_cols = [c for c in df.columns if edited_roles.get(c) != "ignore"]
        df2 = df[use_cols].copy()
        roles2 = {c: r for c, r in edited_roles.items() if c in use_cols}

        st.session_state.df = df2
        st.session_state.meta = meta
        st.session_state.roles = roles2
        st.session_state.step = 2
        st.rerun()

    st.caption("Until you click OK, the report will not be shown.")


# STEP 2
elif st.session_state.step == 2:
    st.header("Step 2 — Choose direction & view report")

    df = st.session_state.df
    meta = st.session_state.meta
    roles = st.session_state.roles

    topbar1, topbar2 = st.columns([1, 2])
    with topbar1:
        if st.button("⬅ Back to Step 1"):
            st.session_state.step = 1
            st.rerun()

    with topbar2:
        focus_label = st.selectbox("Direction / focus", list(FOCUS_OPTIONS.keys()), index=list(FOCUS_OPTIONS.keys()).index(st.session_state.focus_label))
        st.session_state.focus_label = focus_label
        st.session_state.focus = FOCUS_OPTIONS[focus_label]

    st.divider()
    report = build_report(df, roles, st.session_state.focus)

    # PDF button
    pdf_bytes = pdf_from_report(meta, roles, st.session_state.focus_label, report)
    st.download_button(
        label="⬇️ Download PDF report",
        data=pdf_bytes,
        file_name=f"Auto_Report_{meta.get('filename','data')}.pdf",
        mime="application/pdf",
    )
