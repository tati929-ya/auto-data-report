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
# CSV loader
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

ROLE_HELP_EN = {
    "measure": "Numeric metric (sum/avg/median/min/max): sales, revenue, temperature, balance, salary.",
    "categorical": "Groups/labels to compare: gender, product line, status, region, department.",
    "datetime": "Date/time for trends: date, timestamp, order_date.",
    "geo": "Geography: country/city/state or latitude/longitude.",
    "id": "Identifier (usually NOT a KPI): order_id, customer_id, code.",
    "ignore": "Exclude this column from analysis.",
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

        # numeric dtype
        if pd.api.types.is_numeric_dtype(s):
            roles[col] = "id" if is_probably_id(s, col) else "measure"
            continue

        # numeric-like strings
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
# Detection helpers: gender + age
# ---------------------------

def find_col_by_keywords(cols: List[str], keywords: List[str]) -> Optional[str]:
    lower = {c: c.lower() for c in cols}
    for kw in keywords:
        for c, lc in lower.items():
            if kw in lc:
                return c
    return None


def detect_gender_column(cols: List[str]) -> Optional[str]:
    return find_col_by_keywords(cols, ["gender", "sex", "sexe", "genre"])


def detect_age_column(measure_cols: List[str], all_cols: List[str]) -> Optional[str]:
    # prefer real age column
    c = find_col_by_keywords(measure_cols, ["age"])
    if c:
        return c
    # sometimes Age is categorical (string)
    return find_col_by_keywords(all_cols, ["age"])


def detect_dob_column(all_cols: List[str]) -> Optional[str]:
    return find_col_by_keywords(all_cols, ["dob", "birth", "date of birth", "naissance", "birthday"])


def add_age_if_missing(df: pd.DataFrame, roles: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str], Optional[str], Optional[str]]:
    """
    Ensures we have:
      - gender_col if exists
      - age_col (existing) or computed from dob
    Returns updated df, roles, gender_col, age_col
    """
    all_cols = list(df.columns)
    gender_col = detect_gender_column(all_cols)

    measure_cols = pick_cols(roles, "measure")
    age_col = detect_age_column(measure_cols, all_cols)

    if age_col and age_col in df.columns:
        # make sure it's numeric
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
        # force role = measure
        roles[age_col] = "measure"
        return df, roles, gender_col, age_col

    # try DOB
    dob_col = detect_dob_column(all_cols)
    if dob_col and dob_col in df.columns:
        dob = pd.to_datetime(df[dob_col], errors="coerce", infer_datetime_format=True)
        today = pd.Timestamp.today().normalize()
        age = (today - dob).dt.days / 365.25
        df["Age (computed)"] = age.round(1)
        roles["Age (computed)"] = "measure"
        return df, roles, gender_col, "Age (computed)"

    return df, roles, gender_col, None


# ---------------------------
# Charts (with bar labels)
# ---------------------------

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        ax.annotate(
            f"{h:,.0f}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def chart_bar_counts(df: pd.DataFrame, col: str, title: str, top_n: int = 10):
    vc = df[col].astype(str).value_counts().head(top_n)
    fig, ax = plt.subplots()
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    annotate_bars(ax)

    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


def chart_bar_sum_by_category(df: pd.DataFrame, cat: str, measure: str, title: str, top_n: int = 10):
    tmp = df[[cat, measure]].copy()
    tmp[measure] = pd.to_numeric(tmp[measure], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        st.info("Not enough data for bar sum chart.")
        return None
    grp = tmp.groupby(cat)[measure].sum().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots()
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_title(title)
    ax.set_xlabel(cat)
    ax.set_ylabel(f"Sum({measure})")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    annotate_bars(ax)

    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


def chart_hist(df: pd.DataFrame, col: str, title: str):
    fig, ax = plt.subplots()
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    ax.hist(x, bins=25)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


def chart_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots()
    xx = pd.to_numeric(df[x], errors="coerce")
    yy = pd.to_numeric(df[y], errors="coerce")
    m = xx.notna() & yy.notna()
    ax.scatter(xx[m], yy[m], s=10)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


def chart_time_line(df: pd.DataFrame, dt_col: str, measure: str, title: str):
    tmp = df[[dt_col, measure]].copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce", infer_datetime_format=True)
    tmp[measure] = pd.to_numeric(tmp[measure], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        st.info("No valid datetime rows to plot.")
        return None
    tmp = tmp.set_index(dt_col).sort_index()
    daily = tmp[measure].resample("D").sum()

    fig, ax = plt.subplots()
    ax.plot(daily.index, daily.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(measure)
    plt.tight_layout()
    st.pyplot(fig)
    b = fig_to_png_bytes(fig)
    plt.close(fig)
    return b


# ---------------------------
# KPI blocks (focus-dependent)
# ---------------------------

def metric_value(x: Optional[float], fmt: str = "{:,.2f}") -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return fmt.format(x)
    except Exception:
        return str(x)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def build_kpis(df: pd.DataFrame, roles: Dict[str, str], focus: str, gender_col: Optional[str], age_col: Optional[str]) -> Dict:
    measures = pick_cols(roles, "measure")
    cats = pick_cols(roles, "categorical")
    ids = pick_cols(roles, "id")
    dt = pick_datetime_column(roles)

    # Candidate “main” measures by focus
    sales_col = find_col_by_keywords(measures, ["sales", "revenue", "amount", "turnover", "income", "total"])
    balance_col = find_col_by_keywords(measures, ["balance", "debt", "loan", "credit"])
    salary_col = find_col_by_keywords(measures, ["salary", "wage", "pay"])
    duration_col = find_col_by_keywords(measures, ["duration", "lead", "cycle", "time", "delay"])
    qty_col = find_col_by_keywords(measures, ["quantity", "qty", "units"])

    # order id guess (for AOV / basket)
    order_id = find_col_by_keywords(ids + cats, ["ordernumber", "order id", "order", "invoice", "transaction"])

    ds_type = detect_dataset_type(roles)

    # Gender split always if present
    gender_split = None
    if gender_col and gender_col in df.columns:
        vc = df[gender_col].astype(str).value_counts(dropna=True)
        total = float(vc.sum())
        gender_split = [(k, int(v), float(v) / total * 100 if total else 0.0) for k, v in vc.items()]

    # Age stats if available
    age_stats = None
    if age_col and age_col in df.columns:
        a = numeric_series(df, age_col)
        if a.notna().any():
            age_stats = {
                "avg_age": float(a.mean()),
                "median_age": float(a.median()),
                "min_age": float(a.min()),
                "max_age": float(a.max()),
            }

    kpis = []
    # Focus-driven KPI set
    if focus == "sales":
        main = sales_col or measures[0] if measures else None
        if main:
            x = numeric_series(df, main)
            total = float(x.sum()) if x.notna().any() else None
            avg = float(x.mean()) if x.notna().any() else None
            med = float(x.median()) if x.notna().any() else None
            kpis += [
                ("Total revenue (sum)", metric_value(total)),
                ("Average value (mean)", metric_value(avg)),
                ("Median value", metric_value(med)),
            ]

            # Avg basket (AOV) if order id present
            aov = None
            if order_id and order_id in df.columns:
                tmp = df[[order_id, main]].copy()
                tmp[main] = pd.to_numeric(tmp[main], errors="coerce")
                tmp = tmp.dropna()
                if not tmp.empty:
                    aov = float(tmp.groupby(order_id)[main].sum().mean())
            kpis.append(("Avg basket (AOV)", metric_value(aov)))

    elif focus == "finance":
        main = balance_col or sales_col or measures[0] if measures else None
        if main:
            x = numeric_series(df, main)
            avg = float(x.mean()) if x.notna().any() else None
            med = float(x.median()) if x.notna().any() else None
            neg_pct = float((x < 0).mean() * 100) if x.notna().any() else None
            kpis += [
                ("Average balance", metric_value(avg)),
                ("Median balance", metric_value(med)),
                ("% negative", metric_value(neg_pct, "{:,.1f}%")),
                ("Std deviation", metric_value(float(x.std()) if x.notna().any() else None)),
            ]

    elif focus == "hr":
        # headcount heuristic
        headcount = df.shape[0]
        kpis.append(("Headcount (rows)", str(headcount)))
        if age_stats:
            kpis += [
                ("Avg age", metric_value(age_stats["avg_age"], "{:,.1f}")),
                ("Median age", metric_value(age_stats["median_age"], "{:,.1f}")),
                ("Age range", f'{age_stats["min_age"]:.0f}–{age_stats["max_age"]:.0f}'),
            ]
        else:
            # fallback: top category diversity
            if cats:
                kpis.append((f"Unique {cats[0]}", str(df[cats[0]].nunique(dropna=True))))

        if gender_split:
            # show majority as a KPI
            majority = max(gender_split, key=lambda t: t[1])
            kpis.append(("Gender majority", f"{majority[0]} ({majority[2]:.1f}%)"))

    elif focus == "ops":
        main = duration_col or qty_col or sales_col or (measures[0] if measures else None)
        if main:
            x = numeric_series(df, main)
            kpis += [
                ("Average (mean)", metric_value(float(x.mean()) if x.notna().any() else None)),
                ("Median", metric_value(float(x.median()) if x.notna().any() else None)),
                ("P90 (quantile)", metric_value(float(x.quantile(0.9)) if x.notna().any() else None)),
                ("Max", metric_value(float(x.max()) if x.notna().any() else None)),
            ]

    elif focus == "time":
        main = sales_col or balance_col or (measures[0] if measures else None)
        kpis.append(("Has datetime column", "Yes" if dt else "No"))
        if main:
            x = numeric_series(df, main)
            kpis.append(("Total (sum)", metric_value(float(x.sum()) if x.notna().any() else None)))

    elif focus == "geo":
        main = sales_col or balance_col or (measures[0] if measures else None)
        geo_cols = pick_cols(roles, "geo")
        kpis.append(("Geo fields detected", ", ".join(geo_cols) if geo_cols else "None"))
        if main:
            x = numeric_series(df, main)
            kpis.append(("Total (sum)", metric_value(float(x.sum()) if x.notna().any() else None)))

    else:  # business generic
        main = sales_col or balance_col or salary_col or (measures[0] if measures else None)
        if main:
            x = numeric_series(df, main)
            kpis += [
                ("Total (sum)", metric_value(float(x.sum()) if x.notna().any() else None)),
                ("Mean", metric_value(float(x.mean()) if x.notna().any() else None)),
                ("Median", metric_value(float(x.median()) if x.notna().any() else None)),
                ("Std", metric_value(float(x.std()) if x.notna().any() else None)),
            ]

    return {
        "dataset_type": ds_type,
        "kpis": kpis,
        "gender_split": gender_split,
        "age_stats": age_stats,
        "sales_col": sales_col,
        "balance_col": balance_col,
        "qty_col": qty_col,
        "main_measure": None,  # will be chosen in chart plan
    }


# ---------------------------
# Charts plan (focus-dependent)
# ---------------------------

def choose_main_measure(measures: List[str], focus: str) -> Optional[str]:
    if not measures:
        return None
    if focus == "sales":
        c = find_col_by_keywords(measures, ["sales", "revenue", "amount", "turnover", "income"])
        return c or measures[0]
    if focus == "finance":
        c = find_col_by_keywords(measures, ["balance", "debt", "loan", "credit"])
        return c or measures[0]
    if focus == "hr":
        c = find_col_by_keywords(measures, ["salary", "wage", "pay", "age"])
        return c or measures[0]
    if focus == "ops":
        c = find_col_by_keywords(measures, ["duration", "lead", "cycle", "time", "delay", "qty", "quantity"])
        return c or measures[0]
    return measures[0]


def build_report(df: pd.DataFrame, roles: Dict[str, str], focus: str, focus_label: str):
    # Enforce gender+age logic
    df, roles, gender_col, age_col = add_age_if_missing(df, roles)

    measures = pick_cols(roles, "measure")
    cats = pick_cols(roles, "categorical")
    ids = pick_cols(roles, "id")
    dt_col = pick_datetime_column(roles)
    geo_cols = pick_cols(roles, "geo")
    ds_type = detect_dataset_type(roles)

    # Data health (small)
    with st.expander("Data health (technical)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isna().sum().sum()))
        st.write("Detected dataset type:", ds_type)

    # KPIs
    st.subheader(f"KPIs — {focus_label}")
    kpi_pack = build_kpis(df, roles, focus, gender_col, age_col)

    # KPI cards (up to 4)
    kpis = kpi_pack["kpis"]
    cols = st.columns(4)
    for i in range(4):
        if i < len(kpis):
            cols[i].metric(kpis[i][0], kpis[i][1])
        else:
            cols[i].metric("—", "—")

    # Gender + age blocks (always if present)
    if kpi_pack["gender_split"]:
        st.write("Gender split:")
        for g, n, pct in kpi_pack["gender_split"]:
            st.write(f"- {g}: {n} ({pct:.1f}%)")

    if kpi_pack["age_stats"]:
        s = kpi_pack["age_stats"]
        st.write(f"Age stats: avg={s['avg_age']:.1f}, median={s['median_age']:.1f}, range={s['min_age']:.0f}–{s['max_age']:.0f}")

    # Charts
    st.subheader("Charts (5)")
    chart_images: List[bytes] = []
    done = 0

    # If no measures: categorical-only report
    if not measures:
        st.warning("No numeric measures detected → building categorical report.")
        # prioritize gender then other cats
        priority = []
        if gender_col and gender_col in df.columns:
            priority.append(gender_col)
        for c in cats:
            if c not in priority:
                priority.append(c)
        for c in priority[:5]:
            if done >= 5:
                break
            chart_images.append(chart_bar_counts(df, c, f"Top categories: {c}"))
            done += 1

        while done < 5:
            st.info("Not enough columns to create more charts.")
            done += 1

        conclusion = (
            f"This dataset looks like a {ds_type} table dominated by categorical fields. "
            f"The report focuses on top values and distributions by category (focus: {focus})."
        )
        st.subheader("Conclusion (EN)")
        st.write(conclusion)
        return {
            "dataset_type": ds_type,
            "chart_images": chart_images,
            "conclusion": conclusion,
            "roles": roles,
        }

    main = choose_main_measure(measures, focus)

    # Focus-driven chart plan
    # 1) Time trend (if datetime exists and focus needs it)
    if done < 5 and dt_col and focus in ("time", "sales", "business", "finance", "ops") and main:
        b = chart_time_line(df, dt_col, main, f"Time trend (daily sum): {main}")
        if b:
            chart_images.append(b)
            done += 1

    # 2) Sales: bar sum by product/status/top category
    if done < 5 and focus == "sales" and main:
        cat = find_col_by_keywords(cats, ["product", "productline", "category", "status", "deal", "customer"]) or (cats[0] if cats else None)
        if cat:
            b = chart_bar_sum_by_category(df, cat, main, f"Total {main} by {cat} (Top 10)")
            if b:
                chart_images.append(b)
                done += 1

    # 3) Finance: distribution + top category by sum
    if done < 5 and focus == "finance" and main:
        chart_images.append(chart_hist(df, main, f"Distribution: {main}"))
        done += 1
        cat = cats[0] if cats else None
        if done < 5 and cat:
            b = chart_bar_sum_by_category(df, cat, main, f"Total {main} by {cat} (Top 10)")
            if b:
                chart_images.append(b)
                done += 1

    # 4) HR: age distribution + gender bar + optional salary hist
    if focus == "hr":
        if done < 5 and age_col and age_col in df.columns:
            chart_images.append(chart_hist(df, age_col, f"Age distribution: {age_col}"))
            done += 1
        if done < 5 and gender_col and gender_col in df.columns:
            chart_images.append(chart_bar_counts(df, gender_col, f"Gender split: {gender_col}"))
            done += 1
        sal = find_col_by_keywords(measures, ["salary", "wage", "pay"])
        if done < 5 and sal:
            chart_images.append(chart_hist(df, sal, f"Salary distribution: {sal}"))
            done += 1

    # 5) Ops: duration hist + P90-like distribution + category counts
    if focus == "ops":
        dur = find_col_by_keywords(measures, ["duration", "lead", "cycle", "time", "delay"])
        if done < 5 and dur:
            chart_images.append(chart_hist(df, dur, f"Duration distribution: {dur}"))
            done += 1

    # 6) Geo: map if lat/lon exists
    if focus == "geo" and done < 5:
        lat = find_col_by_keywords(df.columns.tolist(), ["lat", "latitude"])
        lon = find_col_by_keywords(df.columns.tolist(), ["lon", "lng", "longitude"])
        if lat and lon:
            st.write("Map (lat/lon):")
            tmp = df[[lat, lon]].copy()
            tmp[lat] = pd.to_numeric(tmp[lat], errors="coerce")
            tmp[lon] = pd.to_numeric(tmp[lon], errors="coerce")
            tmp = tmp.dropna().rename(columns={lat: "lat", lon: "lon"})
            if not tmp.empty:
                st.map(tmp)
            # карта не попадает в PDF как картинка — ок для MVP
        else:
            # fallback: bar counts for best geo column
            g = geo_cols[0] if geo_cols else None
            if g:
                chart_images.append(chart_bar_counts(df, g, f"Geo distribution: {g}"))
                done += 1

    # Fill remaining slots with generic plots
    # hist for up to 2 measures
    i = 0
    while done < 5 and i < len(measures):
        m = measures[i]
        if m != main:
            chart_images.append(chart_hist(df, m, f"Distribution: {m}"))
            done += 1
        i += 1

    # scatter if still space
    if done < 5 and len(measures) >= 2:
        chart_images.append(chart_scatter(df, measures[0], measures[1], f"{measures[0]} vs {measures[1]}"))
        done += 1

    # categorical bars
    j = 0
    while done < 5 and j < len(cats):
        chart_images.append(chart_bar_counts(df, cats[j], f"Top categories: {cats[j]}"))
        done += 1
        j += 1

    while done < 5:
        st.info("Not enough diverse columns to build more charts.")
        done += 1

    conclusion = (
        f"This dataset looks like a {ds_type} table with {df.shape[0]} rows and {df.shape[1]} columns. "
        f"The report is tuned for **{focus_label}** and uses confirmed roles, plus enforced age/gender logic when available."
    )
    st.subheader("Conclusion (EN)")
    st.write(conclusion)

    return {
        "dataset_type": ds_type,
        "chart_images": chart_images,
        "conclusion": conclusion,
        "roles": roles,
    }


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
# Streamlit app (2 steps, focus selected in Step 1)
# ---------------------------

st.set_page_config(page_title="Auto Data Report (MVP)", layout="wide")
st.title("Auto Data Report (MVP)")
st.caption("Step 1: Upload + choose focus + confirm roles → Step 2: Report + PDF")

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


if st.session_state.step == 1:
    st.header("Step 1 — Upload, choose direction, confirm roles")

    with st.expander("Role cheat-sheet (EN)", expanded=True):
        st.markdown(
            "- **measure** — numeric metric (avg/sum/median/min/max)\n"
            "- **categorical** — categories/groups (shares, comparisons)\n"
            "- **datetime** — date/time (trends, seasonality)\n"
            "- **geo** — geography (country/city or lat/lon)\n"
            "- **id** — identifier (usually not a KPI)\n"
            "- **ignore** — exclude\n"
        )

    focus_label = st.selectbox("Direction / focus (select BEFORE report)", list(FOCUS_OPTIONS.keys()))
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
            help=ROLE_HELP_EN.get(default, ""),
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

    st.caption("Until you click OK, no results are shown.")


else:
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
        st.info(f"Focus locked from Step 1: **{st.session_state.focus_label}**")

    st.divider()
    report = build_report(df, roles, st.session_state.focus, st.session_state.focus_label)

    pdf_bytes = pdf_from_report(meta, report["roles"], st.session_state.focus_label, report)
    st.download_button(
        label="⬇️ Download PDF report",
        data=pdf_bytes,
        file_name=f"Auto_Report_{meta.get('filename','data')}.pdf",
        mime="application/pdf",
    )
