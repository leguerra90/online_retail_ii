"""
Online Retail II — issue investigation
Project 1: Customer Intelligence

The profile flagged the problems. This script measures how they overlap,
so the cleaning rules are deliberate rather than guesses. It answers:

  - Do the dates parse now, and what is the real range?
  - Does one StockCode ever carry different Descriptions?
  - Can missing Descriptions be recovered from the StockCode?
  - How do cancellations, negative quantities, and bad prices overlap?
  - What do the zero/negative price rows actually look like?
  - What do the missing-Customer-ID rows look like?
  - What are the duplicated rows?

Still no cleaning here. Read the output, then we set the rules.
"""

import sys
import contextlib

import pandas as pd

DATA_PATH = "kaggle_customer_intelligence/online_retail_II.csv"
REPORT_PATH = "kaggle_customer_intelligence/issue_investigation_report.txt"

# The Kaggle mirror stores dates in ISO form, not the dd/mm/yyyy of the raw file.
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

pd.set_option("display.max_colwidth", 40)
pd.set_option("display.width", 120)


def line(char="-", width=70):
    print(char * width)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def load(path):
    df = pd.read_csv(
        path,
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Description": "string",
            "InvoiceDate": "string",
            "Country": "string",
        },
    )
    # Tidy text copies for comparison. We keep the originals untouched.
    df["desc_clean"] = df["Description"].str.strip().str.upper()
    df["code_clean"] = df["StockCode"].str.strip().str.upper()
    return df


def date_check(df):
    line("=")
    print("1. DATE PARSING (with the corrected ISO format)")
    line("=")
    parsed = pd.to_datetime(df["InvoiceDate"], format=DATE_FORMAT, errors="coerce")
    failed = parsed.isna() & df["InvoiceDate"].notna()
    print(f"Rows:            {len(df):,}")
    print(f"Failed to parse: {failed.sum():,}")
    print(f"Date range:      {parsed.min()}  to  {parsed.max()}")
    print()


def stockcode_vs_description(df):
    line("=")
    print("2. ONE STOCKCODE, MANY DESCRIPTIONS?")
    line("=")
    # Distinct cleaned descriptions per code (whitespace and case ignored).
    has_desc = df.dropna(subset=["desc_clean"])
    per_code = has_desc.groupby("code_clean")["desc_clean"].nunique()
    multi = per_code[per_code > 1].sort_values(ascending=False)

    print(f"StockCodes with a description:        {per_code.size:,}")
    print(f"StockCodes with >1 description:       {multi.size:,}  "
          f"({multi.size / per_code.size * 100:.1f}%)")
    print()
    print("Worst offenders (code: number of distinct descriptions):")
    for code, n in multi.head(8).items():
        print(f"\n  {code}  ->  {n} descriptions")
        variants = (
            has_desc.loc[has_desc["code_clean"] == code, "desc_clean"]
            .value_counts()
            .head(6)
        )
        for desc, count in variants.items():
            print(f"      {count:>7,}  {desc}")
    print()
    print("Note: many of these are blanks like '?', 'damaged', 'lost', or")
    print("genuine relabels. We pick one description per code in cleaning.")
    print()


def missing_description(df):
    line("=")
    print("3. MISSING DESCRIPTIONS — RECOVERABLE FROM STOCKCODE?")
    line("=")
    missing = df["Description"].isna()
    print(f"Rows with missing Description:        {missing.sum():,}")

    codes_with_desc = set(df.loc[df["desc_clean"].notna(), "code_clean"])
    missing_codes = df.loc[missing, "code_clean"]
    recoverable = missing_codes.isin(codes_with_desc)
    print(f"  of these, StockCode seen elsewhere: {recoverable.sum():,}  "
          f"(could backfill)")
    print(f"  StockCode never has a description:  {(~recoverable).sum():,}  "
          f"(likely junk/adjustment rows)")

    print("\n  StockCodes among the non-recoverable missing rows:")
    print("  ",
          sorted(missing_codes[~recoverable].dropna().unique().tolist())[:20])
    print()


def missing_customer_id(df):
    line("=")
    print("4. MISSING CUSTOMER ID — WHAT ARE THESE ROWS?")
    line("=")
    miss = df["Customer ID"].isna()
    print(f"Rows missing Customer ID:             {miss.sum():,}  "
          f"({miss.mean() * 100:.2f}%)")

    cancel = df["Invoice"].str.upper().str.startswith("C", na=False)
    pos_sale = (df["Quantity"] > 0) & (df["Price"] > 0)
    print(f"  of these, are cancellations:        {(miss & cancel).sum():,}")
    print(f"  of these, look like normal sales:   {(miss & pos_sale).sum():,}")
    print(f"  invoices affected:                  "
          f"{df.loc[miss, 'Invoice'].nunique():,}")
    print()
    print("  These can't be tied to a customer, so they're out for RFM and")
    print("  segmentation. They could still feed invoice-level basket rules.")
    print()


def overlaps(df):
    line("=")
    print("5. CANCELLATIONS, NEGATIVE QUANTITY, BAD PRICE — OVERLAP")
    line("=")
    cancel = df["Invoice"].str.upper().str.startswith("C", na=False)
    neg_qty = df["Quantity"] < 0
    zero_price = df["Price"] == 0
    neg_price = df["Price"] < 0

    print(f"Cancellation rows (invoice 'C'):      {cancel.sum():,}")
    print(f"Negative quantity rows:               {neg_qty.sum():,}")
    print(f"  ... that ARE cancellations:         {(neg_qty & cancel).sum():,}")
    print(f"  ... that are NOT cancellations:     {(neg_qty & ~cancel).sum():,}  "
          f"(adjustments?)")
    print(f"Cancellations WITHOUT negative qty:   {(cancel & ~neg_qty).sum():,}")
    print()
    print(f"Zero-price rows:                      {zero_price.sum():,}")
    print(f"Negative-price rows:                  {neg_price.sum():,}")
    print()

    print("Examples — negative quantity, NOT a cancellation:")
    cols = ["Invoice", "StockCode", "Description", "Quantity", "Price", "Customer ID"]
    print(df.loc[neg_qty & ~cancel, cols].head(8).to_string(index=False))
    print()

    print("Examples — zero price:")
    print(df.loc[zero_price, cols].head(8).to_string(index=False))
    print()

    if neg_price.any():
        print("Examples — negative price:")
        print(df.loc[neg_price, cols].head(8).to_string(index=False))
        print()


def duplicates(df):
    line("=")
    print("6. FULLY DUPLICATED ROWS")
    line("=")
    # Duplicates on the original columns only (ignore our helper columns).
    orig = [c for c in df.columns if c not in ("desc_clean", "code_clean")]
    dup_mask = df.duplicated(subset=orig, keep=False)
    print(f"Rows involved in any duplication:     {dup_mask.sum():,}")
    print(f"Rows that would be dropped (keep one): "
          f"{df.duplicated(subset=orig).sum():,}")
    print()
    print("A duplicated group, shown in full:")
    cols = ["Invoice", "StockCode", "Description", "Quantity",
            "InvoiceDate", "Price", "Customer ID"]
    dups = df[dup_mask].sort_values(["Invoice", "StockCode", "InvoiceDate"])
    print(dups[cols].head(10).to_string(index=False))
    print()
    print("Same invoice, same product, same minute, same price. Almost")
    print("certainly a logging artefact, so we keep one of each.")
    print()


def build_report(df):
    date_check(df)
    stockcode_vs_description(df)
    missing_description(df)
    missing_customer_id(df)
    overlaps(df)
    duplicates(df)
    line("=")
    print("Done. Diagnostics only. Cleaning rules come next.")
    line("=")


def main():
    print(f"Loading: {DATA_PATH}\n")
    df = load(DATA_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            build_report(df)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
