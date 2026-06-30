import numpy as np
import pandas as pd

from utilities import get_logger

logger = get_logger(__name__)


def detect_datetime_columns(df):
    datetime_cols = []
    datetime_cols.extend(df.select_dtypes(include="datetime").columns.tolist())

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        success_ratio = parsed.notna().mean()
        if success_ratio > 0.8:
            datetime_cols.append(col)

    if datetime_cols:
        logger.info("Detected datetime columns: %s", datetime_cols)
    return datetime_cols


def detect_missing_values(df):
    datetime_cols = detect_datetime_columns(df)
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = [c for c in df.select_dtypes(include="object").columns if c not in datetime_cols]

    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    details = {}
    missing_numeric, missing_categorical, missing_datetime = [], [], []

    for col, count in missing_counts.items():
        if col in datetime_cols:
            col_type = "datetime"
            missing_datetime.append(col)
        elif col in numeric_cols:
            col_type = "numeric"
            missing_numeric.append(col)
        elif col in categorical_cols:
            col_type = "categorical"
            missing_categorical.append(col)

        details[col] = {
            "count": int(count),
            "pct": round((count / len(df)) * 100, 2),
            "type": col_type,
        }

    logger.info("Missing value detection: %d columns have missing values", len(details))
    return {
        "has_missing": len(missing_counts) > 0,
        "total_missing": int(missing_counts.sum()),
        "details": details,
        "numeric_cols": missing_numeric,
        "categorical_cols": missing_categorical,
        "datetime_cols": missing_datetime,
        "all_datetime_cols": datetime_cols,
    }


def treat_missing_values(df, missing_report, threshold=0.50, indicator_threshold=0.05):
    df_clean = df.copy()

    datetime_cols = missing_report["datetime_cols"]
    numeric_cols = missing_report["numeric_cols"]
    categorical_cols = missing_report["categorical_cols"]

    fill_values = {}
    dropped_cols = []
    filled_log = []
    indicator_cols = []

    for col, info in missing_report["details"].items():
        missing_count = info["count"]
        missing_ratio = missing_count / len(df_clean)

        if missing_ratio > threshold:
            df_clean = df_clean.drop(columns=[col])
            dropped_cols.append((col, round(missing_ratio * 100, 2)))
            logger.warning("Dropped column '%s' (%.2f%% missing)", col, missing_ratio * 100)
            continue

        if col in datetime_cols:
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
            value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(value)
            fill_values[col] = value
            filled_log.append((col, "median date", str(value.date()), missing_count))

        elif col in numeric_cols:
            value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(value)
            fill_values[col] = value
            filled_log.append((col, "median", round(value, 2), missing_count))

        elif col in categorical_cols:
            value = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(value)
            fill_values[col] = value
            filled_log.append((col, "mode", value, missing_count))

    report = {
        "dropped": dropped_cols,
        "filled": filled_log,
        "missing_after": int(df_clean.isnull().sum().sum()),
    }
    logger.info("Missing treatment done: %d filled, %d dropped, %d remaining",
                len(filled_log), len(dropped_cols), report["missing_after"])
    return df_clean, fill_values, report


def detect_duplicates(df):
    full_dup = int(df.duplicated().sum())
    id_like = [col for col in df.columns if df[col].nunique() == len(df)]
    logger.info("Duplicate detection: %d duplicates, ID-like columns: %s", full_dup, id_like)
    return {
        "full_duplicate_count": full_dup,
        "full_duplicate_pct": round((full_dup / len(df)) * 100, 2),
        "id_like_columns": id_like,
    }


def remove_duplicates(df):
    rows_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    rows_removed = rows_before - len(df_clean)
    logger.info("Removed %d duplicate rows", rows_removed)
    return df_clean, rows_removed


def detect_outliers(df, target=None, min_unique=10):
    numeric_cols = df.select_dtypes(include=np.number).columns
    report = {}

    for col in numeric_cols:
        if target is not None and col == target:
            continue
        if df[col].nunique() < min_unique:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        count = int(outlier_mask.sum())

        if count > 0:
            report[col] = {
                "outlier_count": count,
                "outlier_pct": round((count / len(df)) * 100, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
            }

    logger.info("Outlier detection: %d columns have outliers", len(report))
    return report


def treat_outlier(df, outlier_report):
    df_clean = df.copy()
    capped_log = {}
    for col, info in outlier_report.items():
        df_clean[col] = df_clean[col].clip(lower=info["lower_bound"], upper=info["upper_bound"])
        capped_log[col] = info["outlier_count"]
    logger.info("Outlier treatment: capped values in %d columns", len(capped_log))
    return df_clean, capped_log


def clean_dataset(df, target=None):
    logger.info("Starting cleaning pipeline. Shape: %s", df.shape)
    report = {"original_shape": df.shape}

    # duplicates
    df, rows_removed = remove_duplicates(df)
    report["duplicates"] = {"rows_removed": rows_removed}

    # missing values
    missing_report = detect_missing_values(df)
    df, fill_values, treat_report = treat_missing_values(df, missing_report)
    report["missing"] = treat_report
    report["fill_values"] = fill_values

    # outlier
    outlier_report = detect_outliers(df, target=target)
    df, capped_log = treat_outlier(df, outlier_report)
    report["outliers"] = outlier_report

    report["final_shape"] = df.shape
    logger.info("Cleaning pipeline complete. Final shape: %s", df.shape)
    return df, report


def print_cleaning_report(report):
    print("=" * 55)
    print("  CLEANING REPORT")
    print("=" * 55)
    print(f"Original shape : {report['original_shape']}")
    print(f"Final shape    : {report['final_shape']}")

    # Missing
    m = report["missing"]
    print(f"\n[Missing Values]")
    print(f"  Dropped columns : {m['dropped'] or 'None'}")
    print(f"  Filled columns  : {len(m['filled'])}")
    for col, method, value, count in m["filled"]:
        print(f"     - {col} ({method} = {value}, {count} filled)")

    # Duplicates
    d = report["duplicates"]
    print(f"\n[Duplicates]")
    print(f"  Removed : {d['rows_removed']} rows")

    # Outliers
    o = report["outliers"]
    print(f"\n[Outliers] (capped in {len(o)} columns)")
    for col, info in o.items():
        print(f"     - {col}: {info['outlier_count']} capped")