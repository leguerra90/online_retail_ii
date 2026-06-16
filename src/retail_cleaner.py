"""
Online Retail II — Step 1: load and clean

A single class that turns the raw file into a clean sales table and the
two views we agreed on:

  - sales          : every valid sale (positive quantity, positive price,
                     real product). Used for basket analysis, since those
                     rules run per invoice and anonymous baskets are fine.
  - customer_sales : the subset with a Customer ID. Used for RFM,
                     clustering, and co-clustering, which all need to tie
                     rows to a person.

Cleaning rules, decided from the diagnostics:
  1. Parse the ISO dates.
  2. Drop exact duplicate rows (logging artefacts; quantity already counts
     multiples, so identical rows are double entries).
  3. Drop non-sales: invoices starting 'C' (cancellations) or 'A'
     (accounting adjustments), and any row with quantity <= 0 or price <= 0.
  4. Drop non-product codes (postage, manual, fees, charity, gift, DCGS...).
     Real products match five digits with an optional letter suffix.
  5. Normalise descriptions and give each product one canonical name.
  6. Add Revenue = Quantity * Price.
"""

from pathlib import Path

import pandas as pd

# Real product codes: five digits, optional trailing letters (e.g. 85123A).
PRODUCT_CODE_PATTERN = r"\d{5}[A-Z]*"

# Invoice prefixes that mean "not a sale".
NON_SALE_PREFIXES = ("C", "A")


class RetailCleaner:
    def __init__(self, data_path, date_format="%Y-%m-%d %H:%M:%S", verbose=True):
        self.data_path = data_path
        self.date_format = date_format
        self.verbose = verbose
        self.df = None
        self._log = []  # (step, rows_before, rows_after)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _record(self, step, before):
        after = len(self.df)
        self._log.append((step, before, after))
        if self.verbose:
            dropped = before - after
            print(f"{step:<32} {before:>10,} -> {after:>10,}  "
                  f"(removed {dropped:,})")

    def _say(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # steps
    # ------------------------------------------------------------------
    def load(self):
        """Read the file with explicit types. Customer ID becomes nullable Int."""
        self.df = pd.read_csv(
            self.data_path,
            dtype={
                "Invoice": "string",
                "StockCode": "string",
                "Description": "string",
                "InvoiceDate": "string",
                "Country": "string",
            },
        )
        # No space in the column name makes the rest of the code tidier.
        self.df = self.df.rename(columns={"Customer ID": "CustomerID"})
        self.df["CustomerID"] = self.df["CustomerID"].astype("Int64")
        self._say(f"Loaded {len(self.df):,} rows from {self.data_path}\n")
        return self

    def parse_dates(self):
        self.df["InvoiceDate"] = pd.to_datetime(
            self.df["InvoiceDate"], format=self.date_format, errors="coerce"
        )
        bad = self.df["InvoiceDate"].isna().sum()
        if bad:
            self._say(f"Warning: {bad:,} dates failed to parse.")
        return self

    def drop_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        self._record("drop_duplicates", before)
        return self

    def remove_non_sales(self):
        """Cancellations, adjustments, and any non-positive quantity or price."""
        before = len(self.df)
        invoice = self.df["Invoice"].str.upper()
        is_sale_invoice = ~invoice.str.startswith(NON_SALE_PREFIXES, na=False)
        positive = (self.df["Quantity"] > 0) & (self.df["Price"] > 0)
        self.df = self.df[is_sale_invoice & positive].reset_index(drop=True)
        self._record("remove_non_sales", before)
        return self

    def remove_non_product_codes(self):
        """Keep only real product codes. Postage and fees go here, not above,
        because they carry positive prices and survive the sales filter."""
        before = len(self.df)
        code = self.df["StockCode"].str.upper().str.strip()
        is_product = code.str.fullmatch(PRODUCT_CODE_PATTERN, na=False)

        if self.verbose:
            dropped_codes = (
                self.df.loc[~is_product, "StockCode"]
                .str.upper().str.strip()
                .value_counts().head(20)
            )
            self._say("\nDropping these non-product codes (top 20 by rows) — "
                      "check they are all postage/fees/adjustments:")
            for c, n in dropped_codes.items():
                self._say(f"    {str(c):<14} {n:>8,}")
            self._say("")

        self.df = self.df[is_product].reset_index(drop=True)
        self._record("remove_non_product_codes", before)
        return self

    def clean_descriptions(self):
        """Collapse whitespace, then give every StockCode one canonical name
        (the most frequent description for that code). Backfill blanks."""
        desc = (
            self.df["Description"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        self.df["Description"] = desc

        canonical = (
            self.df.dropna(subset=["Description"])
            .groupby("StockCode")["Description"]
            .agg(lambda s: s.value_counts().idxmax())
        )
        mapped = self.df["StockCode"].map(canonical)
        self.df["Description"] = mapped.fillna(self.df["Description"])

        # Anything still blank has a code that never carried a name.
        still_blank = self.df["Description"].isna()
        n_blank = int(still_blank.sum())
        if n_blank:
            self.df.loc[still_blank, "Description"] = (
                "UNKNOWN " + self.df.loc[still_blank, "StockCode"]
            )
            self._say(f"Filled {n_blank:,} unrecoverable descriptions with "
                      f"a placeholder.")
        return self

    def add_revenue(self):
        self.df["Revenue"] = self.df["Quantity"] * self.df["Price"]
        return self

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def run(self):
        if self.verbose:
            print("=" * 64)
            print("CLEANING LOG")
            print("=" * 64)
        (self.load()
             .parse_dates()
             .drop_duplicates()
             .remove_non_sales()
             .remove_non_product_codes()
             .clean_descriptions()
             .add_revenue())
        if self.verbose:
            self.summary()
        return self

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------
    @property
    def sales(self):
        """Every valid sale, anonymous rows included. Use for basket analysis."""
        return self.df

    @property
    def customer_sales(self):
        """Valid sales tied to a customer. Use for RFM and segmentation."""
        return self.df[self.df["CustomerID"].notna()].reset_index(drop=True)

    # ------------------------------------------------------------------
    # reporting and persistence
    # ------------------------------------------------------------------
    @property
    def cleaning_log(self):
        return pd.DataFrame(self._log, columns=["step", "rows_before", "rows_after"])

    def summary(self):
        df = self.df
        cust = self.customer_sales
        print("\n" + "=" * 64)
        print("SUMMARY OF CLEANED DATA")
        print("=" * 64)
        print(f"Rows (all valid sales):     {len(df):,}")
        print(f"Rows with a Customer ID:    {len(cust):,}  "
              f"({len(cust) / len(df) * 100:.1f}%)")
        print(f"Date range:                 {df['InvoiceDate'].min()}  to  "
              f"{df['InvoiceDate'].max()}")
        print(f"Distinct invoices:          {df['Invoice'].nunique():,}")
        print(f"Distinct products:          {df['StockCode'].nunique():,}")
        print(f"Distinct customers:         {cust['CustomerID'].nunique():,}")
        print(f"Total revenue:              GBP {df['Revenue'].sum():,.2f}")
        print(f"Anonymous revenue (no ID):  GBP "
              f"{df.loc[df['CustomerID'].isna(), 'Revenue'].sum():,.2f}")
        print("=" * 64)

    def save(self, out_dir="kaggle_customer_intelligence"):
        """Persist the cleaned table for the next steps. Parquet keeps dtypes
        (dates, nullable ints); falls back to CSV if pyarrow is missing."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = out_dir / "clean_sales.parquet"
            self.df.to_parquet(path, index=False)
        except Exception:
            path = out_dir / "clean_sales.csv"
            self.df.to_csv(path, index=False)
            self._say("pyarrow not found, saved as CSV instead.")
        self._say(f"Saved cleaned data to: {path}")
        return self


if __name__ == "__main__":
    cleaner = RetailCleaner(
        data_path="kaggle_customer_intelligence/online_retail_II.csv"
    )
    cleaner.run()
    cleaner.save()
