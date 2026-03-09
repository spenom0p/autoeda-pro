import pandas as pd
import numpy as np


# ------------------------------
# Detect and Convert Data Types
# ------------------------------
def detect_and_convert_types(df):

    conversions = {}

    for col in df.columns:

        original_type = df[col].dtype

        # Try numeric conversion
        if df[col].dtype == "object":

            numeric_version = pd.to_numeric(df[col], errors="coerce")

            if numeric_version.notna().sum() > len(df) * 0.7:
                df[col] = numeric_version

        # Try datetime conversion
        if df[col].dtype == "object":

            datetime_version = pd.to_datetime(df[col], errors="coerce")

            if datetime_version.notna().sum() > len(df) * 0.7:
                df[col] = datetime_version

        if original_type != df[col].dtype:
            conversions[col] = str(df[col].dtype)

    return df, conversions


# ------------------------------
# Remove Duplicate Rows
# ------------------------------
def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    return df, removed


# ------------------------------
# Drop Columns with Too Many Missing Values
# ------------------------------
def drop_high_missing_columns(df):

    dropped_columns = []

    for col in df.columns:

        missing_ratio = df[col].isnull().mean()

        if missing_ratio > 0.6:
            dropped_columns.append(col)

    df = df.drop(columns=dropped_columns)

    return df, dropped_columns


# ------------------------------
# Drop Rows with Too Many Missing Values
# ------------------------------
def drop_bad_rows(df):

    threshold = int(len(df.columns) * 0.5)

    before = len(df)

    df = df.dropna(thresh=threshold)

    removed_rows = before - len(df)

    return df, removed_rows


# ------------------------------
# Fill Missing Values
# ------------------------------
def handle_missing_values(df):

    filled = {}

    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    for col in numeric_cols:

        missing = df[col].isnull().sum()

        if missing > 0:

            df[col] = df[col].fillna(df[col].median())

            filled[col] = f"{missing} filled with median"

    for col in cat_cols:

        missing = df[col].isnull().sum()

        if missing > 0:

            df[col] = df[col].fillna(df[col].mode()[0])

            filled[col] = f"{missing} filled with mode"

    return df, filled


# ------------------------------
# Outlier Detection (IQR)
# ------------------------------
def detect_outliers(df):

    outliers = {}

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = ((df[col] < lower) | (df[col] > upper)).sum()

        if count > 0:
            outliers[col] = int(count)

    return outliers


# ------------------------------
# Main Cleaning Function
# ------------------------------
def clean_data(df):

    report = {}

    # datatype conversion
    df, conversions = detect_and_convert_types(df)
    report["datatype_conversions"] = conversions

    # duplicates
    df, duplicates_removed = remove_duplicates(df)
    report["duplicates_removed"] = duplicates_removed

    # drop columns
    df, dropped_cols = drop_high_missing_columns(df)
    report["dropped_columns"] = dropped_cols

    # drop rows
    df, removed_rows = drop_bad_rows(df)
    report["rows_removed"] = removed_rows

    # missing values
    df, filled = handle_missing_values(df)
    report["missing_filled"] = filled

    # outlier detection
    outliers = detect_outliers(df)
    report["outliers_detected"] = outliers

    return df, report