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
    "measure": "Числовая метрика для анализа (avg/sum/min/max): sales, revenue, temperature, balance, salary.",
    "categorical": "Категории/группы для сравнения: gender, product_line, status, region, department.",
    "datetime": "Дата/время для трендов: date, timestamp, order_date.",
    "geo": "География: country, city, state, lat/lon.",
    "id": "Идентификатор (обычно НЕ KPI): order_id, customer_id, code.",
    "ignore": "Исключить колонку из анализа.",
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

        # try numeric coercion (sometimes numbers are strings)
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().mean() >= 0.9 and not is_probably_id(coerced.dropna(), col):
            roles[col] = "measure"
            continue

        if is_probably_id(s.astype(str), col):
            roles[col] = "id"
            continue

        roles[col] = "categorical"

    return roles


def pick_cols(roles: Dict[str, str], role: str) -> List[str]:
    return [c for c, r in roles.items() if r == role]


def pick_datetime_column(roles: Dict[str, str]) -> Optional[str]:
    for c, r in roles.items():
        if r == "datetime":
            return c
    return None


def detect_dataset_type(roles: Dict[str, str]) -> str:
    has_dt = any(r == "datetime" for r in roles.values())
    has_geo = any(r == "geo" for r in roles.values())
    measures = pick_cols(roles, "measure")
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
# Focus options
# ---------------------------

FOCUS_OPTIONS = {
    "Business (generic)": "business",
    "Sales / Revenue": "sales",
    "Finance": "finance",
    "HR": "hr",
    "Operations / Process": "ops",
    "Time patterns": "time",
    "Geography": "geo",
}


# ---------------------------
# Chart helpers
# ---------------------------

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def chart_hist(df: pd.DataFrame, col: str, title: str):
    fig = plt.figure()
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    plt.hist(x, bins=25)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
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


# ---------------------------
# “Smart KPI” engine (business-like)
# ---------------------------

def find_col_by_keywords(cols: List[str], keywords: List[str]) -> Optional[str]:
    lower = {c: c.lower() for c in cols}
    for kw in keywords:
        for c, lc in lower.items():
            if kw in lc:
                return c
    return None


def render_kpi_cards(df: pd.DataFrame, roles: Dict[str, str], focus: str):
    measures = pick_cols(roles, "measure")
    cats = pick_cols(roles, "categorical")
    ids = pick_cols(roles, "id")
    dt = pick_datetime_column(roles)
    ds_type = detect_dataset_type(roles)

    # data health (small)
    with st.expander("Data health (technical)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isna().sum().sum()))
        st.write("Detected dataset type:", ds_type)

    st.subheader("Key KPIs")

    # If we have measures: choose business KPIs
    if measures:
        sales_col = find_col_by_keywords(measures, ["sales", "revenue", "amount", "turnover", "income", "total"])
        qty_col = find_col_by_keywords(measures, ["quantity", "qty", "units"])
        age_col = find_col_by_keywords(measures, ["age"])
        salary_col = find_col_by_keywords(measures, ["salary", "wage", "pay"])
        balance_col = find_col_by_keywords(measures, ["balance", "debt", "loan", "credit"])

        order_id = find_col_by_keywords(ids + cats, ["ordernumber", "order id", "order", "invoice", "transaction"])

        # pick main measure fallback
        main = sales_col or balance_col or salary_col or age_col or measures[0]

        # Build KPI cards depending on focus
        c1, c2, c3, c4 = st.columns(4)

        def metric_safe(label, value, fmt=None):
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                value = "—"
            elif fmt:
                value = fmt(value)
            return label, value

        # compute basics
        x = pd.to_numeric(df[main], errors="coerce")
        total = float(x.sum(skipna=True))
        mean = float(x.mean(skipna=True))
        median = float(x.median(skipna=True))

        # avg basket if possible (sales + order id)
        avg_basket = None
        if sales_col and order_id and order_id in df.columns:
            tmp = df[[order_id, sales_col]].copy()
            tmp[sales_col] = pd.to_numeric(tmp[sales_col], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                avg_basket = float(tmp.groupby(order_id)[sales_col].sum().mean())

        # headcount-like (unique id)
        headcount = None
        if focus == "hr":
            emp_id = find_col_by_keywords(ids + cats, ["employee", "emp", "staff", "person", "id"])
            if emp_id and emp_id in df.columns:
                headcount = int(df[emp_id].nunique(dropna=True))
            else:
                headcount = int(df.shape[0])

        # category count
        top_cat = cats[0] if cats else None
        top_cat_value = None
        if top_cat:
            vc = df[top_cat].astype(str).value_counts()
            if len(vc) > 0:
                top_cat_value = str(vc.index[0])

        # Render 4 cards
        if focus in ("sales", "business"):
            l, v = metric_safe("Total (main)", total, lambda z: f"{z:,.2f}")
            c1.metric(l, v)
            l, v = metric_safe("Mean (main)", mean, lambda z: f"{z:,.2f}")
            c2.metric(l, v)
            l, v = metric_safe("Median (main)", median, lambda z: f"{z:,.2f}")
            c3.metric(l, v)
            if avg_basket is not None:
                l, v = metric_safe("Avg basket", avg_basket, lambda z: f"{z:,.2f}")
                c4.metric(l, v)
            else:
                c4.metric("Top category", top_cat_value or "—")

        elif focus == "finance":
            main2 = balance_col or main
            xb = pd.to_numeric(df[main2], errors="coerce")
            neg_pct = float((xb < 0).mean()) * 100 if xb.notna().any() else None
            c1.metric("Avg balance", f"{float(xb.mean()):,.2f}" if xb.notna().any() else "—")
            c2.metric("Median balance", f"{float(xb.median()):,.2f}" if xb.notna().any() else "—")
            c3.metric("% negative", f"{neg_pct:.1f}%" if neg_pct is not None else "—")
            c4.metric("Top category", top_cat_value or "—")

        elif focus == "hr":
            if age_col:
                xa = pd.to_numeric(df[age_col], errors="coerce")
                c1.metric("Headcount", headcount if headcount is not None else df.shape[0])
                c2.metric("Avg age", f"{float(xa.mean()):.1f}" if xa.notna().any() else "—")
                c3.metric("Median age", f"{float(xa.median()):.1f}" if xa.notna().any() else "—")
                c4.metric("Top category", top_cat_value or "—")
            else:
                c1.metric("Headcount", headcount if headcount is not None else df.shape[0])
                c2.metric("Main mean", f"{mean:,.2f}")
                c3.metric("Main median", f"{median:,.2f}")
                c4.metric("Top category", top_cat_value or "—")

        else:
            # generic fallback
            c1.metric("Main total", f"{total:,.2f}")
            c2.metric("Main mean", f"{mean:,.2f}")
            c3.metric("Main median", f"{median:,.2f}")
            c4.metric("Top category", top_cat_value or "—")

        return {"dataset_type": ds_type, "main_measure": main, "sales_col": sales_col, "qty_col": qty_col, "top_cat": top_cat}

    # If NO measures: categorical KPIs
    st.warning("No numeric measures detected → showing categorical statistics.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Unique categories (total)", int(sum(df[c].nunique(dropna=True) for c in df.columns)))
    c4.metric("Missing cells", int(df.isna().sum().sum()))

    # Show top categories for up to 3 categorical columns
    cats_any = pick_cols(roles, "categorical") or list(df.columns[:3])
    for col in cats_any[:3]:
        st.write(f"Top values for: **{col}**")
        vc = df[col].astype(str).value_counts().head(10)
        st.dataframe(vc.rename("count").reset_index().rename(columns={"index": col}))

    return {"dataset_type": detect_dataset_type(roles), "main_measure": None, "sales_col": None, "qty_col": None, "top_cat": None}


# ---------------------------
# Report builder (direction-dependent charts)
# ---------------------------

def build_report(df: pd.DataFrame, roles: Dict[str, str], focus: str):
    measures = pick_cols(roles, "measure")
    cats = pick_cols(roles, "categorical")
    dt_col = pick_datetime_column(roles)
    ds_type = detect_dataset_type(roles)

    kpi_ctx = render_kpi_cards(df, roles, focus)

    st.subheader("Charts (5)")
    chart_images: List[bytes] = []
    charts_done = 0

    # If no measures → rely on categorical charts
    if not measures:
        for c in cats[:5]:
            if charts_done >= 5:
                break
            chart_images.append(chart_bar_top(df, c, f"Top categories: {c}"))
            charts_done += 1
        while charts_done < 5:
            st.info("Not enough columns to create more charts.")
            charts_done += 1

        conclusion = (
            f"This dataset looks like a {ds_type} table with mostly categorical fields. "
            f"The report focuses on distributions and top values by category."
        )
        st.subheader("Conclusion (EN)")
        st.write(conclusion)
        return {"dataset_type": ds_type, "chart_images": chart_images, "conclusion": conclusion, "kpi_df": pd.DataFrame()}

    # With measures → choose main metric
    main = kpi_ctx.get("main_measure") or measures[0]

    # 1) If time-related focus and datetime exists
    if charts_done < 5 and focus in ("time", "sales", "business", "finance") and dt_col:
        b = chart_time_line(df, dt_col, main, f"Time trend (daily sum): {main}")
        if b:
            chart_images.append(b)
            charts_done += 1

    # 2) 2 histograms
    for m in measures[:2]:
        if charts_done >= 5:
            break
        chart_images.append(chart_hist(df, m, f"Distribution: {m}"))
        charts_done += 1

    # 3) scatter if possible
    if charts_done < 5 and len(measures) >= 2:
        chart_images.append(chart_scatter(df, measures[0], measures[1], f"{measures[0]} vs {measures[1]}"))
        charts_done += 1

    # 4) category bar
    if charts_done < 5 and cats:
        chart_images.append(chart_bar_top(df, cats[0], f"Top categories: {cats[0]}"))
        charts_done += 1

    # fill
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

    conclusion = (
        f"This dataset looks like a {ds_type} table with {df.shape[0]} rows and {df.shape[1]} columns. "
        f"The report highlights key KPIs and visible patterns based on confirmed roles and focus: {focus}."
    )
    st.subheader("Conclusion (EN)")
    st.write(conclusion)

    return {"dataset_type": ds_type, "chart_images": chart_images, "conclusion": conclusion, "kpi_df": pd.DataFrame()}


# ---------------------------
# PDF generation
# ---------------------------

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
        c.drawString(2.2 * cm, y, f"- {col}: {r}"[:160])
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

    imgs: List[bytes] = report.get("chart_images", [])
    for img_bytes in imgs[:5]:
        c.showPage()
        y = h - 2 * cm
        y = header("Chart", y)
        img = ImageReader(io.BytesIO(img_bytes))
        max_w = w - 4 * cm
        max_h = h - 5 * cm
        c.drawImage(img, 2 * cm, 2 * cm, width=max_w, height=max_h, preserveAspectRatio=True, anchor="c")

    c.save()
    buf.seek(0)
    return buf.getvalue()


# ---------------------------
# Streamlit App (2 steps)
# ---------------------------

st.set_page_config(page_title="Auto Data Report (MVP)", layout="wide")
st.title("Auto Data Report (MVP)")
st.caption("Step 1: Upload + choose direction + confirm roles → Step 2: Report + PDF")

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
    st.header("Step 1 — Upload, choose direction, confirm roles")

    with st.expander("Памятка: что значит measure/categorical/id/datetime/geo?", expanded=True):
        st.markdown(
            "- **measure** — числовая метрика (сумма/среднее/медиана и т.д.)\n"
            "- **categorical** — категории/группы (для сравнения и долей)\n"
            "- **datetime** — дата/время (тренды, сезонность)\n"
            "- **geo** — география (страны/города/координаты)\n"
            "- **id** — идентификатор (обычно НЕ KPI)\n"
            "- **ignore** — исключить колонку\n"
        )

    # Focus selection MUST be here
    focus_label = st.selectbox("Direction / focus (выбери до отчёта)", list(FOCUS_OPTIONS.keys()))
    st.session_state.focus_label = focus_label
    st.session_state.focus = FOCUS_OPTIONS[focus_label]

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

    if st.button("✅ OK, generate report"):
        use_cols = [c for c in df.columns if edited_roles.get(c) != "ignore"]
        df2 = df[use_cols].copy()
        roles2 = {c: r for c, r in edited_roles.items() if c in use_cols}

        st.session_state.df = df2
        st.session_state.meta = meta
        st.session_state.roles = roles2
        st.session_state.step = 2
        st.rerun()

    st.caption("Пока ты не нажмёшь OK — результата не будет.")


# STEP 2
elif st.session_state.step == 2:
    st.header("Step 2 — Report")

    df = st.session_state.df
    meta = st.session_state.meta
    roles = st.session_state.roles

    top1, top2 = st.columns([1, 2])
    with top1:
        if st.button("⬅ Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
    with top2:
        st.info(f"Direction/focus locked from Step 1: **{st.session_state.focus_label}**")

    st.divider()
    report = build_report(df, roles, st.session_state.focus)

    pdf_bytes = pdf_from_report(meta, roles, st.session_state.focus_label, report)
    st.download_button(
        label="⬇️ Download PDF report",
        data=pdf_bytes,
        file_name=f"Auto_Report_{meta.get('filename','data')}.pdf",
        mime="application/pdf",
    )
