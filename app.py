import io
import chardet
import streamlit as st
import pandas as pd
import plotly.express as px

from cleaning import clean_data
from eda import generate_eda
from insights import generate_insights
from nlp_query import run_query


st.set_page_config(page_title="AutoEDA Pro", layout="wide")

st.title("AutoEDA Pro - Automated Data Cleaning & EDA")


# session variables
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

if "report" not in st.session_state:
    st.session_state.report = None


file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])


if file:

    if file.name.endswith("csv"):
        raw = file.read()
        detected = chardet.detect(raw)
        encoding = detected["encoding"] or "utf-8"

        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)

        except (UnicodeDecodeError, pd.errors.ParserError):
            df = None
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue

            if df is None:
                st.error("Could not decode the CSV file. Try re-saving it as UTF-8.")
                st.stop()

    else:
        df = pd.read_excel(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.button("Run AutoEDA"):

        df_clean, report = clean_data(df)

        st.session_state.df_clean = df_clean
        st.session_state.report = report


# ------------------------------------
# Only run below if dataset processed
# ------------------------------------
if st.session_state.df_clean is not None:

    df_clean = st.session_state.df_clean
    report = st.session_state.report


    st.subheader("Cleaning Report")

    st.write("### Datatype Conversions")
    if report["datatype_conversions"]:
        st.write(report["datatype_conversions"])
    else:
        st.write("No datatype conversions needed")

    st.write("### Date Standardization")
    if report["date_standardization"]:
        st.write(report["date_standardization"])
    else:
        st.write("No date columns found or standardized")

    st.write("### Duplicates Removed")
    st.write(report["duplicates_removed"])

    st.write("### Dropped Columns")
    if report["dropped_columns"]:
        st.write(report["dropped_columns"])
    else:
        st.write("No columns dropped")

    st.write("### Rows Removed")
    st.write(report["rows_removed"])

    st.write("### Missing Values Filled")
    st.write(report["missing_filled"])

    st.write("### Outliers Detected")
    st.write(report["outliers_detected"])


    st.subheader("Clean Dataset Preview")
    st.dataframe(df_clean.head())

    # ✅ Download button placed right after cleaning report
    st.download_button(
        "Download Clean Dataset",
        df_clean.to_csv(index=False),
        "clean_data.csv"
    )

    st.divider()

    # automatic EDA
    generate_eda(df_clean)

    generate_insights(df_clean)


    # --------------------------------
    # Column Analyzer
    # --------------------------------
    st.subheader("Column Analyzer")

    col = st.selectbox("Select Column", df_clean.columns)

    st.write("Datatype:", df_clean[col].dtype)
    st.write("Unique:", df_clean[col].nunique())
    st.write("Missing:", df_clean[col].isnull().sum())

    if df_clean[col].dtype != "object":
        st.write("Mean:", df_clean[col].mean())
        st.write("Median:", df_clean[col].median())


    # --------------------------------
    # NLP Query
    # --------------------------------
    st.subheader("Ask Questions About Dataset")

    st.write("Available columns:")
    st.write(list(df_clean.columns))

    query = st.text_input("Example: average salary")

    if query:
        run_query(df_clean, query)


    # --------------------------------
    # Custom Chart Builder
    # --------------------------------
    st.subheader("Custom Chart Builder")

    x = st.selectbox("X axis", df_clean.columns, key="x")
    y = st.selectbox("Y axis", df_clean.columns, key="y")

    chart = st.selectbox(
        "Chart Type",
        [
            "scatter",
            "bar",
            "line",
            "histogram",
            "box",
            "violin",
            "pie"
        ],
        key="chart"
    )

    chart_help = {
        "scatter": "Shows relationship between two numeric variables.",
        "bar": "Compares values across categories.",
        "line": "Shows trends over time or ordered data.",
        "histogram": "Displays the distribution of a numeric column.",
        "box": "Shows distribution and detects outliers.",
        "violin": "Displays distribution density and variation.",
        "pie": "Shows proportion of categories."
    }

    st.info(chart_help[chart])

    if chart == "scatter":
        fig = px.scatter(df_clean, x=x, y=y)

    elif chart == "bar":
        fig = px.bar(df_clean, x=x, y=y)

    elif chart == "line":
        fig = px.line(df_clean, x=x, y=y)

    elif chart == "histogram":
        fig = px.histogram(df_clean, x=x)

    elif chart == "box":
        fig = px.box(df_clean, x=x, y=y)

    elif chart == "violin":
        fig = px.violin(df_clean, x=x, y=y)

    elif chart == "pie":
        fig = px.pie(df_clean, names=x)

    st.plotly_chart(fig, use_container_width=True)
