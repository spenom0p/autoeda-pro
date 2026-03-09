import streamlit as st
import pandas as pd
import re


def normalize_columns(df):

    mapping = {}

    for col in df.columns:

        key1 = col.lower()
        key2 = col.lower().replace("_"," ")

        mapping[key1] = col
        mapping[key2] = col

    return mapping


def detect_column(query, column_map):

    for key in column_map:

        if key in query:
            return column_map[key]

    return None


def detect_two_columns(query, column_map):

    found = []

    for key in column_map:

        if key in query:
            found.append(column_map[key])

    return list(set(found))


def run_query(df, query):

    query = query.lower().strip()

    column_map = normalize_columns(df)

    numeric_cols = df.select_dtypes(include="number").columns

    # -----------------------------
    # GROUP BY QUERIES
    # -----------------------------
    if " by " in query:

        words = query.split(" by ")

        metric_part = words[0]
        group_part = words[1]

        metric_col = detect_column(metric_part, column_map)
        group_col = detect_column(group_part, column_map)

        if metric_col and group_col:

            if "average" in metric_part or "mean" in metric_part:

                result = df.groupby(group_col)[metric_col].mean()

            elif "sum" in metric_part or "total" in metric_part:

                result = df.groupby(group_col)[metric_col].sum()

            elif "count" in metric_part:

                result = df.groupby(group_col).size()

            else:

                st.warning("Unsupported groupby operation")
                return

            st.write(result.sort_values(ascending=False))

            return

    # -----------------------------
    # AVERAGE / MEAN
    # -----------------------------
    if "average" in query or "mean" in query:

        col = detect_column(query, column_map)

        if col and col in numeric_cols:

            st.success(f"Average {col}: {round(df[col].mean(),2)}")
            return

    # -----------------------------
    # MEDIAN
    # -----------------------------
    if "median" in query:

        col = detect_column(query, column_map)

        if col and col in numeric_cols:

            st.success(f"Median {col}: {round(df[col].median(),2)}")
            return

    # -----------------------------
    # SUM
    # -----------------------------
    if "sum" in query or "total" in query:

        col = detect_column(query, column_map)

        if col and col in numeric_cols:

            st.success(f"Total {col}: {round(df[col].sum(),2)}")
            return

    # -----------------------------
    # MAX
    # -----------------------------
    if "max" in query or "highest" in query:

        col = detect_column(query, column_map)

        if col and col in numeric_cols:

            st.success(f"Max {col}: {df[col].max()}")
            return

    # -----------------------------
    # MIN
    # -----------------------------
    if "min" in query or "lowest" in query:

        col = detect_column(query, column_map)

        if col and col in numeric_cols:

            st.success(f"Min {col}: {df[col].min()}")
            return

    # -----------------------------
    # TOP VALUES
    # -----------------------------
    if "top" in query or "most" in query:

        col = detect_column(query, column_map)

        if col:

            if col in numeric_cols:

                st.write(df[[col]].sort_values(by=col, ascending=False).head())

            else:

                st.write(df[col].value_counts().head())

            return

    # -----------------------------
    # COUNT
    # -----------------------------
    if "count" in query:

        col = detect_column(query, column_map)

        if col:

            st.write(df[col].value_counts())
            return

    # -----------------------------
    # UNIQUE VALUES
    # -----------------------------
    if "unique" in query:

        col = detect_column(query, column_map)

        if col:

            st.write(df[col].unique())
            return

    # -----------------------------
    # CORRELATION
    # -----------------------------
    if "correlation" in query:

        cols = detect_two_columns(query, column_map)

        if len(cols) >= 2:

            c1, c2 = cols[0], cols[1]

            if c1 in numeric_cols and c2 in numeric_cols:

                corr = df[c1].corr(df[c2])

                st.success(f"Correlation between {c1} and {c2}: {round(corr,3)}")

                return

    # -----------------------------
    # DESCRIBE
    # -----------------------------
    if "describe" in query or "summary" in query:

        col = detect_column(query, column_map)

        if col:

            st.write(df[col].describe())
            return

    # -----------------------------
    # HELP MESSAGE
    # -----------------------------
    st.warning(
        """
Query not recognized.

Try examples:

average salary  
median price  
sum sales  
max salary  
min price  

top city  
top 5 cities  
most common city  

count gender  
unique city  

correlation price size  

average price by city  
sum sales by department  
count by city
"""
    )