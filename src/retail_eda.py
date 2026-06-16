"""
Online Retail II — Step 2: exploratory analysis

Takes the cleaned sales table and produces the charts, tables, and headline
numbers for the writeup. It also surfaces the data quirks worth describing.

Every method saves a table
(CSV) and a figure (PNG) into:

    <out_dir>/tables/
    <out_dir>/figures/

and keeps the table in self.tables[name] for inspection.

Pieces:
    revenue_over_time   monthly revenue, partial months flagged
    top_products        ranked by revenue and by quantity (they differ)
    country_split       row share, revenue share, average order value
    basket_sizes        distinct products, units, spend per invoice
    customer_long_tail  Pareto curve, one-time vs repeat
Quirks:
    extreme_orders      the giant single orders
    day_of_week         the suspected Saturday gap
    price_variation     same product, different prices
    anonymous_share     revenue with no Customer ID

Input is the full sales table (anonymous rows included). The customer view
(rows with a Customer ID) is derived internally for the customer-only pieces.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RetailEDA:
    def __init__(self, sales, out_dir="results/eda",
                 verbose=True):
        self.sales = sales.copy()
        self.customers = self.sales[self.sales["CustomerID"].notna()].copy()
        self.verbose = verbose

        self.out_dir = Path(out_dir)
        self.fig_dir = self.out_dir / "figures"
        self.tbl_dir = self.out_dir / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tbl_dir.mkdir(parents=True, exist_ok=True)

        self.tables = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _say(self, msg):
        if self.verbose:
            print(msg)

    def _save_table(self, df, name):
        df.to_csv(self.tbl_dir / f"{name}.csv")
        self.tables[name] = df

    def _save_fig(self, fig, name):
        fig.savefig(self.fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 1. revenue over time
    # ------------------------------------------------------------------
    def revenue_over_time(self):
        s = self.sales
        month = s["InvoiceDate"].dt.to_period("M")
        monthly = s.groupby(month)["Revenue"].sum().rename("revenue")
        monthly.index.name = "month"

        first, last = s["InvoiceDate"].min(), s["InvoiceDate"].max()
        first_p, last_p = first.to_period("M"), last.to_period("M")

        def partial(p):
            if p == first_p and first.day != 1:
                return True
            if p == last_p and last.day != p.days_in_month:
                return True
            return False

        table = monthly.to_frame()
        table["partial_month"] = [partial(p) for p in table.index]
        self._save_table(table, "revenue_over_time")

        fig, ax = plt.subplots(figsize=(11, 5))
        x = table.index.to_timestamp()
        ax.plot(x, table["revenue"], marker="o", color="#2b6cb0", label="full month")
        part = table[table["partial_month"]]
        if not part.empty:
            ax.plot(part.index.to_timestamp(), part["revenue"], "o",
                    color="#c53030", label="partial month")
        ax.set_title("Monthly revenue")
        ax.set_ylabel("Revenue (GBP)")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save_fig(fig, "revenue_over_time")

        self._say(f"Revenue over time: {len(table)} months, "
                  f"{int(table['partial_month'].sum())} partial.")
        return self

    # ------------------------------------------------------------------
    # 2. top products
    # ------------------------------------------------------------------
    def top_products(self, n=20):
        prod = self.sales.groupby("StockCode").agg(
            description=("Description", "first"),
            revenue=("Revenue", "sum"),
            quantity=("Quantity", "sum"),
            invoices=("Invoice", "nunique"),
        )

        by_rev = prod.sort_values("revenue", ascending=False).head(n)
        by_qty = prod.sort_values("quantity", ascending=False).head(n)
        self._save_table(by_rev, "top_products_by_revenue")
        self._save_table(by_qty, "top_products_by_quantity")

        for table, value, title, fname in [
            (by_rev, "revenue", "Top products by revenue", "top_products_by_revenue"),
            (by_qty, "quantity", "Top products by quantity", "top_products_by_quantity"),
        ]:
            labels = table["description"].str.slice(0, 32)[::-1]
            fig, ax = plt.subplots(figsize=(9, 8))
            ax.barh(labels, table[value][::-1], color="#2b6cb0")
            ax.set_title(title)
            ax.set_xlabel(value)
            self._save_fig(fig, fname)

        self._say("Top products: saved revenue and quantity rankings "
                  "(the lists differ).")
        return self

    # ------------------------------------------------------------------
    # 3. country split
    # ------------------------------------------------------------------
    def country_split(self):
        total_rows = len(self.sales)
        total_rev = self.sales["Revenue"].sum()

        g = self.sales.groupby("Country").agg(
            rows=("Invoice", "size"),
            invoices=("Invoice", "nunique"),
            revenue=("Revenue", "sum"),
        )
        g["row_share_%"] = (g["rows"] / total_rows * 100).round(2)
        g["revenue_share_%"] = (g["revenue"] / total_rev * 100).round(2)
        g["avg_order_value"] = (g["revenue"] / g["invoices"]).round(2)
        g = g.sort_values("revenue", ascending=False)
        self._save_table(g, "country_split")

        # Average order value for the busier markets shows who buys wholesale.
        top = g.sort_values("revenue", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top.index[::-1], top["avg_order_value"][::-1], color="#2b6cb0")
        ax.set_title("Average order value, top 12 markets by revenue")
        ax.set_xlabel("Average order value (GBP)")
        self._save_fig(fig, "country_avg_order_value")

        self._say("Country labels present: "
                  f"{sorted(g.index.tolist())}")
        return self

    # ------------------------------------------------------------------
    # 4. basket sizes
    # ------------------------------------------------------------------
    def basket_sizes(self):
        basket = self.sales.groupby("Invoice").agg(
            n_products=("StockCode", "nunique"),
            n_units=("Quantity", "sum"),
            spend=("Revenue", "sum"),
        )
        stats = basket.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        self._save_table(stats, "basket_size_stats")

        # Clip at the 99th percentile so the long tail doesn't flatten the plot.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col, title in [
            (axes[0], "n_products", "Distinct products per invoice"),
            (axes[1], "spend", "Spend per invoice (GBP)"),
        ]:
            cap = basket[col].quantile(0.99)
            ax.hist(basket[col].clip(upper=cap), bins=50, color="#2b6cb0")
            ax.set_title(title)
            ax.set_ylabel("Invoices")
        fig.suptitle("Basket sizes (clipped at 99th percentile)")
        self._save_fig(fig, "basket_sizes")

        self._say(f"Basket sizes: median {basket['n_products'].median():.0f} "
                  f"products, median spend GBP {basket['spend'].median():.2f}.")
        return self

    # ------------------------------------------------------------------
    # 5. long tail of customers
    # ------------------------------------------------------------------
    def customer_long_tail(self):
        cust = self.customers.groupby("CustomerID").agg(
            revenue=("Revenue", "sum"),
            invoices=("Invoice", "nunique"),
        ).sort_values("revenue", ascending=False)

        cust["cum_revenue_share"] = cust["revenue"].cumsum() / cust["revenue"].sum()
        cust["customer_rank_share"] = np.arange(1, len(cust) + 1) / len(cust)
        self._save_table(cust, "customer_revenue")

        # Headline: share of revenue from the top 20% of customers.
        top20 = cust.loc[cust["customer_rank_share"] <= 0.20, "revenue"].sum()
        top20_share = top20 / cust["revenue"].sum()
        one_time = (cust["invoices"] == 1).mean()

        summary = pd.DataFrame({
            "metric": ["customers", "top_20%_revenue_share",
                       "one_time_buyer_rate", "repeat_buyer_rate"],
            "value": [len(cust), round(top20_share, 4),
                      round(one_time, 4), round(1 - one_time, 4)],
        })
        self._save_table(summary, "customer_summary")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cust["customer_rank_share"], cust["cum_revenue_share"],
                color="#2b6cb0")
        ax.axvline(0.20, color="#c53030", linestyle="--")
        ax.set_title("Cumulative revenue by customer (Pareto)")
        ax.set_xlabel("Share of customers (richest first)")
        ax.set_ylabel("Cumulative share of revenue")
        ax.grid(alpha=0.3)
        self._save_fig(fig, "customer_pareto")

        self._say(f"Long tail: top 20% of customers hold "
                  f"{top20_share * 100:.1f}% of revenue; "
                  f"{one_time * 100:.1f}% buy only once.")
        return self

    # ------------------------------------------------------------------
    # quirks
    # ------------------------------------------------------------------
    def extreme_orders(self, n=10):
        cols = ["Invoice", "StockCode", "Description", "Quantity",
                "Price", "Revenue", "CustomerID", "Country"]
        top_qty = self.sales.nlargest(n, "Quantity")[cols]
        top_rev = self.sales.nlargest(n, "Revenue")[cols]
        self._save_table(top_qty, "extreme_orders_by_quantity")
        self._save_table(top_rev, "extreme_orders_by_revenue")
        self._say(f"Extreme orders: largest single line is "
                  f"{self.sales['Quantity'].max():,} units.")
        return self

    def day_of_week(self):
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = (self.sales.drop_duplicates("Invoice")["InvoiceDate"]
               .dt.dayofweek.value_counts().reindex(range(7), fill_value=0))
        table = pd.DataFrame({"weekday": days, "invoices": dow.values})
        self._save_table(table.set_index("weekday"), "invoices_by_weekday")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(days, table["invoices"], color="#2b6cb0")
        ax.set_title("Invoices by day of week")
        ax.set_ylabel("Invoices")
        self._save_fig(fig, "invoices_by_weekday")

        zero_days = table.loc[table["invoices"] == 0, "weekday"].tolist()
        self._say(f"Day of week: no-trade days = {zero_days or 'none'}.")
        return self

    def price_variation(self, n=20):
        p = self.sales.groupby("StockCode").agg(
            description=("Description", "first"),
            n_prices=("Price", "nunique"),
            min_price=("Price", "min"),
            max_price=("Price", "max"),
            mean_price=("Price", "mean"),
        )
        p["spread"] = p["max_price"] - p["min_price"]
        top = p.sort_values("spread", ascending=False).head(n)
        self._save_table(top, "price_variation")
        self._say(f"Price variation: {(p['n_prices'] > 1).mean() * 100:.1f}% "
                  f"of products sell at more than one price.")
        return self

    def anonymous_share(self):
        anon = self.sales["CustomerID"].isna()
        table = pd.DataFrame({
            "metric": ["row_share", "revenue_share"],
            "value": [round(anon.mean(), 4),
                      round(self.sales.loc[anon, "Revenue"].sum()
                            / self.sales["Revenue"].sum(), 4)],
        })
        self._save_table(table, "anonymous_share")
        self._say(f"Anonymous: {anon.mean() * 100:.1f}% of rows, "
                  f"{table['value'][1] * 100:.1f}% of revenue, have no ID.")
        return self

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def run(self):
        if self.verbose:
            print("=" * 64)
            print("EXPLORATORY ANALYSIS")
            print("=" * 64)
        (self.revenue_over_time()
             .top_products()
             .country_split()
             .basket_sizes()
             .customer_long_tail()
             .extreme_orders()
             .day_of_week()
             .price_variation()
             .anonymous_share())
        self._say(f"\nSaved figures to {self.fig_dir} and tables to {self.tbl_dir}.")
        return self


if __name__ == "__main__":
    # Standalone run: load the cleaned table the cleaner saved.
    path = Path("kaggle_customer_intelligence/clean_sales.parquet")
    if path.exists():
        sales = pd.read_parquet(path)
    else:
        sales = pd.read_csv("kaggle_customer_intelligence/clean_sales.csv",
                            parse_dates=["InvoiceDate"])
        sales["CustomerID"] = sales["CustomerID"].astype("Int64")
    RetailEDA(sales).run()
