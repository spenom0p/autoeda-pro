import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import re


# ──────────────────────────────────────────
# Type detection & conversion
# ──────────────────────────────────────────
def detect_and_convert_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Tries numeric → datetime → category conversions for object columns.
    Only converts when >70% of non-null values parse successfully.
    Returns updated df + a human-readable conversion log.
    """
    conversions = {}

    for col in df.columns:
        original_dtype = str(df[col].dtype)

        if df[col].dtype != object:
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            continue

        # 1. Try numeric
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().mean() > 0.7:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            conversions[col] = f"{original_dtype} → {df[col].dtype}"
            continue

        # 2. Try datetime (common formats only — avoid false positives)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dt = pd.to_datetime(non_null, errors="coerce")
            if dt.notna().mean() > 0.7:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                conversions[col] = f"{original_dtype} → datetime64"
                continue
        except Exception:
            pass

        # 3. Convert low-cardinality strings to category (memory + speed win)
        if non_null.nunique() / len(non_null) < 0.1:
            df[col] = df[col].astype("category")
            conversions[col] = f"{original_dtype} → category"

    return df, conversions


# ──────────────────────────────────────────
# Standardize date formats to DD-MM-YYYY
# ──────────────────────────────────────────
def standardize_date_formats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Converts all datetime columns to DD-MM-YYYY format string.
    Handles mixed date formats: DD-MM-YYYY, MM/DD/YYYY, M/DD/YYYY, MM-DD-YYYY, etc.
    Returns updated df + a log of formatted columns.
    """
    formatted = {}
    
    for col in df.columns:
        original_dtype = str(df[col].dtype)
        is_date_column = False
        
        # Check if column is datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%d-%m-%Y')
            formatted[col] = "Formatted to DD-MM-YYYY"
            is_date_column = True
        
        # Check if object column contains mixed date formats
        elif df[col].dtype == 'object':
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            
            try:
                # Try to parse with multiple format attempts
                successful_parse = 0
                total_valid = len(non_null)
                
                # First attempt: generic pandas datetime parsing
                parsed_dates = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
                successful_parse = parsed_dates.notna().sum()
                
                # If >70% successful, convert the entire column
                if successful_parse / total_valid > 0.7:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    # Format all valid dates to DD-MM-YYYY
                    df[col] = df[col].dt.strftime('%d-%m-%Y')
                    formatted[col] = f"Standardized {original_dtype} to DD-MM-YYYY ({successful_parse}/{total_valid} dates)"
                    is_date_column = True
                    
            except Exception as e:
                # Silent fail - not a date column
                pass
        
    return df, formatted


# ──────────────────────────────────────────
# Duplicates
# ──────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


# ──────────────────────────────────────────
# Column triage — drop near-empty columns
# ──────────────────────────────────────────
def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.6) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns where more than `threshold` fraction of values are missing."""
    to_drop = [col for col in df.columns if df[col].isnull().mean() > threshold]
    df = df.drop(columns=to_drop)
    return df, to_drop


# ──────────────────────────────────────────
# Row triage — drop rows with too few valid cells
# ──────────────────────────────────────────
def drop_bad_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows where more than half the columns are null.
    Uses at-least-half logic so we don't over-drop wide datasets.
    """
    threshold = max(1, int(len(df.columns) * 0.5))
    before = len(df)
    df = df.dropna(thresh=threshold).reset_index(drop=True)
    return df, before - len(df)


# ──────────────────────────────────────────
# Missing value imputation
# ──────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Numeric → median imputation (robust to outliers).
    Categorical/object → mode imputation.
    Reports count of cells filled per column.
    """
    filled = {}

    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        n = df[col].isnull().sum()
        if n > 0:
            df[col] = df[col].fillna(df[col].median())
            filled[col] = {"count": int(n), "method": "median", "value": float(df[col].median())}

    for col in cat_cols:
        n = df[col].isnull().sum()
        if n > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                filled[col] = {"count": int(n), "method": "mode", "value": str(mode_val[0])}

    return df, filled


# ──────────────────────────────────────────
# Outlier detection (IQR) — detect only, never silently drop
# ──────────────────────────────────────────
def detect_outliers(df: pd.DataFrame) -> dict:
    """
    IQR-based outlier detection.
    Returns per-column dict: count, pct, lower/upper fence.
    Outliers are flagged for the user — not automatically removed.
    """
    outliers = {}
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count > 0:
            outliers[col] = {
                "count": count,
                "pct": round(count / len(df) * 100, 2),
                "lower_fence": round(float(lower), 4),
                "upper_fence": round(float(upper), 4),
            }

    return outliers


# ──────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────
def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended describe() — adds skewness, kurtosis, and missing count.
    Only for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        return pd.DataFrame()

    stats = df[numeric_cols].describe().T
    stats["skewness"] = df[numeric_cols].skew()
    stats["kurtosis"] = df[numeric_cols].kurt()
    stats["missing"] = df[numeric_cols].isnull().sum()
    return stats.round(3)


# ──────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline. Returns (cleaned_df, report_dict).
    The report carries structured data — the UI layer decides how to render it.
    """
    report = {
        "original_shape": df.shape,
        "datatype_conversions": {},
        "date_standardization": {},
        "duplicates_removed": 0,
        "dropped_columns": [],
        "rows_removed": 0,
        "missing_filled": {},
        "outliers_detected": {},
        "summary_stats": None,
        "cleaned_shape": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    df, report["datatype_conversions"] = detect_and_convert_types(df)
    df, report["date_standardization"] = standardize_date_formats(df)
    df, report["duplicates_removed"] = remove_duplicates(df)
    df, report["dropped_columns"] = drop_high_missing_columns(df)
    df, report["rows_removed"] = drop_bad_rows(df)
    df, report["missing_filled"] = handle_missing_values(df)
    report["outliers_detected"] = detect_outliers(df)
    report["summary_stats"] = compute_summary_stats(df)
    report["cleaned_shape"] = df.shape

    return df, report
