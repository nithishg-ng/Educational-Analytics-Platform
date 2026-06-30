import numpy as np
import pandas as pd

from utilities import get_logger

logger = get_logger(__name__)


def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded: %s (%d rows, %d cols)",
                    file_path, df.shape[0], df.shape[1])
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        return None
    except pd.errors.EmptyDataError:
        logger.error("File is empty: %s", file_path)
        return None
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return None


def analyse_dataset(df, target):
    print("=" * 60)
    print("DATASET ANALYSIS REPORT")
    print("=" * 60)

    # records & columns
    print(f"\nNumber of records(rows) : {df.shape[0]}")
    print(f"Number of columns : {df.shape[1]}")

    # memory usage
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage : {mem_mb:.2f} MB")

    # duplicates
    print(f"Duplicate rows : {df.duplicated().sum()}")

    # data types
    print("\n--- Data Types ---")
    print(df.dtypes)

    # missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, cnt in missing.items():
            pct = (cnt / len(df)) * 100
            print(f"{col}:{cnt} ({pct:.2f}%)")
    else:
        print("No Missing Values")

    # numeric feature summary
    print("\n--- Numeric Feature Summary ---")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().round(2).to_string())
    else:
        print("No Numeric Columns")

    # target distribution
    print(f"\n--- Target Distribution: '{target}' ---")
    print(df[target].value_counts())
    print("\npercentage:")
    print((df[target].value_counts(normalize=True) * 100).round(2))

    logger.info("Analysis complete for target '%s'", target)
    print("=" * 60)