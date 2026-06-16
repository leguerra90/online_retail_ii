"""
Online Retail II — data profiling
Project 1: Customer Intelligence

Profiles the raw file before any cleaning. Reports structure, types,
missing values, basic stats, and the specific quirks this dataset is
known for: cancellations, missing customer IDs, negative quantities,
zero/negative prices, non-product stock codes, and whitespace in text.

Nothing is changed here. No rows dropped, no columns altered.
Run this, read the output, then decide what cleaning the data needs.
"""

import sys
import contextlib

import pandas as pd

# Point this at your local copy.
DATA_PATH = "kaggle_customer_intelligence/online_retail_II.csv"

# Where the printed output gets saved.
REPORT_PATH = "kaggle_customer_intelligence/data_profile_report.txt"

# The date column is ISO: YYYY-MM-DD HH:MM:SS. (The raw UCI file uses
# dd/mm/yyyy, but the Kaggle mirror is reformatted to ISO.) We load it as
# text and parse it ourselves so we can see exactly what fails.
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Stock codes that aren't real products (postage, manual adjustments, etc.).
# We flag these for review rather than assuming the list is complete.
KNOWN_NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "BANK CHARGES", "C2", "PADS",
    "S", "AMAZONFEE", "CRUK", "B", "D", "ADJUST", "ADJUST2",
}


def line(char="-", width=70):
    print(char * width)


class _Tee:
    """Write to several streams at once (screen and file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def load_raw(path):
    """Load the file with minimal assumptions so we can see it as it is."""
    df = pd.read_csv(
        path,
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Description": "string",
            "InvoiceDate": "string",   # parsed later, on purpose
            "Country": "string",
        },
        # Customer ID left to default so we can see it arrive as float (NaNs).
    )
    return df


def basic_shape(df):
    line("=")
    print("SIZE AND SHAPE")
    line("=")
    rows, cols = df.shape
    print(f"Rows:    {rows:,}")
    print(f"Columns: {cols}")
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Memory:  {mem:,.1f} MB (deep)")
    print()


def columns_and_types(df):
    line("=")
    print("COLUMNS AND TYPES")
    line("=")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype("string"),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=True),
    })
    print(summary.to_string())
    print()


def sample_values(df, n=5):
    line("=")
    print(f"SAMPLE VALUES (first {n} non-null per column)")
    line("=")
    for col in df.columns:
        vals = df[col].dropna().unique()[:n]
        print(f"{col}:")
        print(f"  {list(vals)}")
    print()


def numeric_stats(df):
    line("=")
    print("NUMERIC SUMMARY (Quantity, Price)")
    line("=")
    for col in ["Quantity", "Price"]:
        if col not in df.columns:
            continue
        s = df[col]
        print(f"{col}:")
        print(f"  min:        {s.min()}")
        print(f"  max:        {s.max()}")
        print(f"  mean:       {s.mean():.4f}")
        print(f"  median:     {s.median()}")
        print(f"  negatives:  {(s < 0).sum():,}")
        print(f"  zeros:      {(s == 0).sum():,}")
        print()


def date_check(df):
    line("=")
    print("DATE PARSING (InvoiceDate)")
    line("=")
    parsed = pd.to_datetime(df["InvoiceDate"], format=DATE_FORMAT, errors="coerce")
    failed = parsed.isna() & df["InvoiceDate"].notna()
    print(f"Failed to parse: {failed.sum():,}")
    if failed.any():
        print("  examples of unparsed values:")
        print(f"  {list(df.loc[failed, 'InvoiceDate'].unique()[:5])}")
    valid = parsed.dropna()
    if not valid.empty:
        print(f"Date range:      {valid.min()}  to  {valid.max()}")
    print()


def issue_checks(df):
    line("=")
    print("KNOWN ISSUE CHECKS")
    line("=")
    rows = len(df)

    # Cancellations: invoice code starts with C (case-insensitive).
    cancel = df["Invoice"].str.upper().str.startswith("C", na=False)
    print(f"Cancellation invoices (start 'C'):  {cancel.sum():,}  "
          f"({cancel.mean() * 100:.2f}%)")

    # Missing customer IDs.
    miss_id = df["Customer ID"].isna()
    print(f"Missing Customer ID:                {miss_id.sum():,}  "
          f"({miss_id.mean() * 100:.2f}%)")

    # Negative quantity / non-positive price.
    print(f"Negative Quantity:                  {(df['Quantity'] < 0).sum():,}")
    print(f"Zero or negative Price:             {(df['Price'] <= 0).sum():,}")

    # Non-product stock codes.
    codes = df["StockCode"].str.upper().str.strip()
    non_product = codes.isin({c.upper() for c in KNOWN_NON_PRODUCT_CODES})
    print(f"Known non-product StockCodes:       {non_product.sum():,}")
    if non_product.any():
        print("  codes present:",
              sorted(codes[non_product].dropna().unique().tolist()))

    # Stock codes that are purely letters (often adjustments, worth a look).
    letters_only = codes.str.fullmatch(r"[A-Z]+", na=False)
    print(f"StockCodes that are letters only:   {letters_only.sum():,}")
    if letters_only.any():
        print("  examples:",
              sorted(codes[letters_only].dropna().unique().tolist())[:15])

    # Whitespace in Description (leading/trailing).
    desc = df["Description"]
    has_ws = desc.notna() & (desc != desc.str.strip())
    print(f"Descriptions with stray whitespace: {has_ws.sum():,}")

    # Fully duplicated rows.
    dupes = df.duplicated().sum()
    print(f"Fully duplicated rows:              {dupes:,}  "
          f"({dupes / rows * 100:.2f}%)")
    print()


def country_breakdown(df, top=15):
    line("=")
    print(f"COUNTRY BREAKDOWN (top {top} by row count)")
    line("=")
    counts = df["Country"].value_counts(dropna=False).head(top)
    for country, n in counts.items():
        print(f"  {str(country):<25} {n:>10,}  ({n / len(df) * 100:5.2f}%)")
    print(f"\nDistinct countries: {df['Country'].nunique(dropna=True)}")
    print()


def build_report(df):
    basic_shape(df)
    columns_and_types(df)
    sample_values(df)
    numeric_stats(df)
    date_check(df)
    issue_checks(df)
    country_breakdown(df)

    line("=")
    print("Done. Nothing was changed. Use this to plan the cleaning step.")
    line("=")


def main():
    print(f"Loading: {DATA_PATH}\n")
    df = load_raw(DATA_PATH)

    # Send everything to the screen and the report file at the same time.
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            build_report(df)

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
