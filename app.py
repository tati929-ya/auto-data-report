import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Auto Data Report", layout="wide")
st.title("Auto Data Report (MVP)")
st.caption("Upload a CSV → see quick KPIs and charts")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to start.")
    st.stop()

# Read CSV (simple MVP: comma delimiter)
df = pd.read_csv(uploaded)

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Basic stats")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing cells", int(df.isna().sum().sum()))

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

st.write("Numeric columns:", num_cols if num_cols else "—")
st.write("Other columns:", cat_cols if cat_cols else "—")

if num_cols:
    st.subheader("KPIs (numeric)")
    desc = df[num_cols].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    st.dataframe(desc, use_container_width=True)

    st.subheader("Charts (up to 5)")
    charts = 0

    # 1-2: distributions
    for c in num_cols[:2]:
        fig = plt.figure()
        plt.hist(df[c].dropna(), bins=30)
        plt.title(f"Distribution: {c}")
        st.pyplot(fig, clear_figure=True)
        charts += 1

    # 3: scatter of first two numeric
    if len(num_cols) >= 2 and charts < 5:
        x, y = num_cols[0], num_cols[1]
        fig = plt.figure()
        plt.scatter(df[x], df[y], s=10)
        plt.title(f"{y} vs {x}")
        plt.xlabel(x)
        plt.ylabel(y)
        st.pyplot(fig, clear_figure=True)
        charts += 1

    # 4: correlation heatmap (simple)
    if charts < 5 and len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig = plt.figure()
        plt.imshow(corr.values, aspect="auto")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation (numeric)")
        st.pyplot(fig, clear_figure=True)
        charts += 1

# simple conclusion
st.subheader("Conclusion (EN)")
st.write(
    f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
    "The report highlights basic quality signals, numeric KPIs, and simple visual patterns."
)
