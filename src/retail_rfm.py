"""
Online Retail II — Step 3: RFM features

Turns the customer-attributed sales into one row per customer with:

    recency    days since last purchase, counted back from the day after
               the data ends (2011-12-10)
    frequency  number of distinct invoices (shopping trips, not items)
    monetary   total spend over the whole period

then scores each on a 1-5 scale and inspects the distributions.

Two things to know about the numbers, recorded in rfm_notes.txt as well:

  - Monetary is GROSS. Cancellations were removed in cleaning, so a
    customer who bought and later returned goods still shows their full
    spend. We chose this for simplicity and documented the bias.
  - Two orders are dropped first: the 80,995-unit and 74,215-unit invoices
    that were each placed and cancelled the same day. We kept the positive
    sides in cleaning, so they have to go here or they crown customers who
    never really spent that.

Recency and monetary are scored with rank-based quintiles (five equal
groups). Frequency can't be, because 28% of customers share frequency = 1,
which is more than one quintile can hold and makes the bottom scores
arbitrary. So frequency uses fixed, meaningful bins instead:

    1 -> 1, 2 -> 2, 3-5 -> 3, 6-10 -> 4, 11+ -> 5

These bins are deliberately unequal in size, and no customer's score
depends on row order.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The two placed-and-cancelled phantom orders. Confirm with inspect_extremes().
DEFAULT_PHANTOM_INVOICES = ("541431", "581483")

# Frequency is too lumpy for quintiles (a quarter of customers bought once),
# so it gets hand-picked bins on meaningful thresholds instead.
#   1 -> 1, 2 -> 2, 3-5 -> 3, 6-10 -> 4, 11+ -> 5
FREQUENCY_BIN_EDGES = [0, 1, 2, 5, 10, float("inf")]
FREQUENCY_BIN_LABELS = [1, 2, 3, 4, 5]

COLOUR = "#2b6cb0"


class RFMBuilder:
    def __init__(self, customer_sales, reference_date=None,
                 exclude_invoices=DEFAULT_PHANTOM_INVOICES,
                 out_dir="results/rfm", verbose=True):
        df = customer_sales.copy()
        self.df = df[df["CustomerID"].notna()].reset_index(drop=True)
        self.reference_date = (
            pd.to_datetime(reference_date) if reference_date else None
        )
        self.exclude_invoices = {str(i) for i in (exclude_invoices or [])}
        self.verbose = verbose
        self.rfm = None

        self.out_dir = Path(out_dir)
        self.fig_dir = self.out_dir / "figures"
        self.tbl_dir = self.out_dir / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tbl_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _say(self, msg):
        if self.verbose:
            print(msg)

    def _save_table(self, df, name):
        df.to_csv(self.tbl_dir / f"{name}.csv")

    def _save_fig(self, fig, name):
        fig.savefig(self.fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _score(self, s, reverse):
        """Quintile score for recency and monetary. Rank first (ties broken by
        order), then cut into five equal groups. reverse=True means a low value
        earns a high score (used for recency)."""
        ranked = s.rank(method="first")
        labels = [5, 4, 3, 2, 1] if reverse else [1, 2, 3, 4, 5]
        return pd.qcut(ranked, 5, labels=labels).astype(int)

    def _score_frequency(self, s):
        """Fixed-bin score for frequency. Quintiles don't work here because a
        quarter of customers share frequency = 1, so we cut on meaningful
        thresholds: 1, 2, 3-5, 6-10, 11+. Bins are unequal on purpose, and the
        boundaries no longer depend on row order."""
        return pd.cut(s, bins=FREQUENCY_BIN_EDGES,
                      labels=FREQUENCY_BIN_LABELS).astype(int)

    # ------------------------------------------------------------------
    # 1. inspect before removing
    # ------------------------------------------------------------------
    def inspect_extremes(self, n=10):
        cols = ["Invoice", "CustomerID", "StockCode", "Description",
                "Quantity", "Price", "Revenue"]

        spend = (self.df.groupby("CustomerID")["Revenue"].sum()
                 .sort_values(ascending=False).head(n).round(2))
        largest_lines = self.df.nlargest(n, "Quantity")[cols]
        top_invoices = (self.df.groupby("Invoice")
                        .agg(customer=("CustomerID", "first"),
                             revenue=("Revenue", "sum"),
                             units=("Quantity", "sum"))
                        .sort_values("revenue", ascending=False).head(n).round(2))

        self._save_table(spend.to_frame(), "extremes_top_customers_by_spend")
        self._save_table(largest_lines, "extremes_largest_lines")
        self._save_table(top_invoices, "extremes_top_invoices_by_revenue")

        if self.verbose:
            print("Top customers by total spend:")
            print(spend.to_string())
            print("\nLargest single order lines:")
            print(largest_lines.to_string(index=False))
            print("\nTop invoices by revenue:")
            print(top_invoices.to_string())

            top_inv = str(self.df.loc[self.df["Quantity"].idxmax(), "Invoice"])
            if top_inv not in self.exclude_invoices:
                print(f"\nWARNING: the largest-quantity invoice ({top_inv}) is "
                      f"not in the drop list {sorted(self.exclude_invoices)}. "
                      f"Review before trusting monetary.")
            print()
        return self

    # ------------------------------------------------------------------
    # 2. remove the phantom
    # ------------------------------------------------------------------
    def remove_outliers(self):
        if not self.exclude_invoices:
            return self
        mask = self.df["Invoice"].astype(str).isin(self.exclude_invoices)
        if mask.any():
            removed = self.df[mask]
            self._say(f"Removing {mask.sum()} row(s) from invoice(s) "
                      f"{sorted(self.exclude_invoices)}, "
                      f"revenue GBP {removed['Revenue'].sum():,.2f}:")
            self._say(removed[["Invoice", "CustomerID", "StockCode",
                               "Quantity", "Price", "Revenue"]].to_string(index=False))
            self.df = self.df[~mask].reset_index(drop=True)
        else:
            self._say(f"None of {sorted(self.exclude_invoices)} found; "
                      f"nothing removed.")
        self._say("")
        return self

    # ------------------------------------------------------------------
    # 3. compute raw RFM
    # ------------------------------------------------------------------
    def compute_rfm(self):
        if self.reference_date is None:
            self.reference_date = (
                self.df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
            )
        g = self.df.groupby("CustomerID")
        last_purchase = g["InvoiceDate"].max()

        self.rfm = pd.DataFrame({
            "recency": (self.reference_date - last_purchase.dt.normalize()).dt.days,
            "frequency": g["Invoice"].nunique(),
            "monetary": g["Revenue"].sum().round(2),
        })
        self._say(f"Reference date for recency: "
                  f"{self.reference_date.date()}  "
                  f"(customers: {len(self.rfm):,})")
        return self

    # ------------------------------------------------------------------
    # 4. score 1-5
    # ------------------------------------------------------------------
    def score_rfm(self):
        self.rfm["R"] = self._score(self.rfm["recency"], reverse=True)
        self.rfm["F"] = self._score_frequency(self.rfm["frequency"])
        self.rfm["M"] = self._score(self.rfm["monetary"], reverse=False)
        self.rfm["RFM_cell"] = (
            self.rfm[["R", "F", "M"]].astype(str).agg("".join, axis=1)
        )
        self.rfm["RFM_sum"] = self.rfm[["R", "F", "M"]].sum(axis=1)
        return self

    # ------------------------------------------------------------------
    # 5. distributions
    # ------------------------------------------------------------------
    def distributions(self):
        stats = self.rfm[["recency", "frequency", "monetary"]].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        self._save_table(stats, "rfm_stats")

        # Raw values (frequency and monetary clipped so the tail doesn't flatten
        # the plot), with log versions underneath to show the skew tamed.
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        raw = [("recency", "Recency (days)"),
               ("frequency", "Frequency (invoices)"),
               ("monetary", "Monetary (GBP)")]
        for ax, (col, title) in zip(axes[0], raw):
            cap = self.rfm[col].quantile(0.99)
            ax.hist(self.rfm[col].clip(upper=cap), bins=40, color=COLOUR)
            ax.set_title(title)
            ax.set_ylabel("Customers")

        axes[1, 0].axis("off")
        for ax, col, title in [
            (axes[1, 1], "frequency", "log(1 + frequency)"),
            (axes[1, 2], "monetary", "log(1 + monetary)"),
        ]:
            ax.hist(np.log1p(self.rfm[col]), bins=40, color=COLOUR)
            ax.set_title(title)
            ax.set_ylabel("Customers")
        fig.suptitle("RFM distributions (raw clipped at 99th pct; log below)")
        self._save_fig(fig, "rfm_distributions")

        # Score counts. Recency and monetary are even by construction (quintiles).
        # Frequency uses fixed thresholds, so its bins are deliberately unequal,
        # which is the thing to look at.
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, col in zip(axes, ["R", "F", "M"]):
            counts = self.rfm[col].value_counts().sort_index()
            ax.bar(counts.index, counts.values, color=COLOUR)
            ax.set_title(f"{col} score")
            ax.set_xlabel("score 1-5")
        fig.suptitle("Customers per score")
        self._save_fig(fig, "rfm_score_counts")

        one_time = (self.rfm["frequency"] == 1).mean()
        self._say(f"\nMedian recency {self.rfm['recency'].median():.0f} days, "
                  f"median frequency {self.rfm['frequency'].median():.0f}, "
                  f"median monetary GBP {self.rfm['monetary'].median():,.2f}.")
        self._say(f"One-time buyers: {one_time * 100:.1f}% of customers.")
        return self

    # ------------------------------------------------------------------
    # notes, persistence, orchestration
    # ------------------------------------------------------------------
    def _write_notes(self):
        notes = (
            "RFM feature notes\n"
            "=================\n"
            f"Reference date (recency anchor): {self.reference_date.date()} "
            "(day after the last invoice).\n"
            "Frequency: count of distinct invoices per customer.\n"
            "Monetary: GROSS total spend. Cancellations were removed in "
            "cleaning, so returns are NOT netted out; a customer who returned "
            "goods still shows full spend. Known, documented bias.\n"
            f"Dropped invoices (placed-and-cancelled phantom): "
            f"{sorted(self.exclude_invoices)}.\n"
            "Scoring: recency and monetary use rank-based quintiles (1-5), "
            "recency reversed (recent = 5). Frequency uses fixed bins "
            "(1, 2, 3-5, 6-10, 11+) because a quarter of customers share "
            "frequency = 1, which makes quintiles arbitrary at the bottom.\n"
        )
        (self.tbl_dir / "rfm_notes.txt").write_text(notes, encoding="utf-8")

    def save(self, data_dir="kaggle_customer_intelligence"):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = data_dir / "rfm_features.parquet"
            self.rfm.to_parquet(path)
        except Exception:
            path = data_dir / "rfm_features.csv"
            self.rfm.to_csv(path)
            self._say("pyarrow not found, saved RFM features as CSV.")
        self._say(f"Saved RFM features to: {path}")
        return self

    @property
    def features(self):
        return self.rfm

    def run(self):
        if self.verbose:
            print("=" * 64)
            print("RFM FEATURES")
            print("=" * 64)
        (self.inspect_extremes()
             .remove_outliers()
             .compute_rfm()
             .score_rfm()
             .distributions())
        self._write_notes()
        self._say(f"\nSaved figures to {self.fig_dir} and tables to {self.tbl_dir}.")
        return self


if __name__ == "__main__":
    path = Path("kaggle_customer_intelligence/clean_sales.parquet")
    if path.exists():
        sales = pd.read_parquet(path)
    else:
        sales = pd.read_csv("kaggle_customer_intelligence/clean_sales.csv",
                            parse_dates=["InvoiceDate"])
        sales["CustomerID"] = sales["CustomerID"].astype("Int64")
    customer_sales = sales[sales["CustomerID"].notna()]
    RFMBuilder(customer_sales).run().save()
