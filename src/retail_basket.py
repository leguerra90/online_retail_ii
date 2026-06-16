"""
Online Retail II — Step 5: basket analysis

Finds products bought together, using association rules over invoices. The
unit here is the invoice, not the customer, so this runs on ALL valid
invoices, anonymous baskets included.

Three numbers define each rule (antecedent X -> consequent Y):
    support     how common the combination is across all baskets
    confidence  of baskets with X, the share that also have Y
    lift        how much more often X and Y occur together than if they
                were independent. 1 = unrelated, >1 = bought together,
                <1 = repel. We rank by lift, because it ignores popularity.

Items are identified by StockCode. Descriptions are attached
afterwards so the rules are readable.

Same shape as the other steps. Input is the cleaned sales (all valid rows).
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

DEFAULT_PHANTOM_INVOICES = ("541431", "581483")
COLOUR = "#2b6cb0"

# Words that describe a variant (colour, pattern, size, filler), not the
# product itself. Stripped before comparing two descriptions, so that
# "BLUE POLKADOT CUP" and "RED RETROSPOT CUP" both reduce to "CUP".
VARIANT_WORDS = {
    # colours
    "RED", "BLUE", "PINK", "GREEN", "BLACK", "WHITE", "ORANGE", "YELLOW",
    "PURPLE", "BROWN", "GREY", "GRAY", "GOLD", "SILVER", "IVORY", "CREAM",
    "TURQUOISE", "NAVY", "BEIGE",
    # patterns / ranges
    "POLKADOT", "RETROSPOT", "SPOTTY", "SPOT", "WOODLAND", "STRAWBERRY",
    "SUKI", "FLORAL", "PAISLEY", "GINGHAM", "VINTAGE", "DOILY", "REGENCY",
    "CHRISTMAS", "EASTER", "HEARTS", "STAR", "STARS",
    # sizes / filler
    "SMALL", "LARGE", "MEDIUM", "MINI", "SET", "OF", "AND", "THE", "A",
    "WITH", "IN", "DESIGN", "ASSORTED", "SIZE",
}

# How the two sides of a rule relate, by share of overlapping core words.
PATTERN_VARIANT_THRESHOLD = 0.8   # same named product, different colour/pattern
SAME_TYPE_THRESHOLD = 0.4         # same product type, different sub-product


class BasketAnalyser:
    def __init__(self, sales, min_support=0.01, min_lift=1.0,
                 exclude_invoices=DEFAULT_PHANTOM_INVOICES,
                 out_dir="results/basket", verbose=True):
        df = sales.copy()
        drop = {str(i) for i in (exclude_invoices or [])}
        self.sales = df[~df["Invoice"].astype(str).isin(drop)].reset_index(drop=True)

        self.min_support = min_support
        self.min_lift = min_lift
        self.verbose = verbose

        self.basket = None       # sparse invoice x product presence
        self.itemsets = None
        self.rules = None
        self.code_to_desc = None

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
        df.to_csv(self.tbl_dir / f"{name}.csv", index=False)

    def _save_fig(self, fig, name):
        fig.savefig(self.fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _names(self, codes):
        """Turn a frozenset of stock codes into a readable string."""
        return ", ".join(self.code_to_desc.get(c, c) for c in sorted(codes))

    # ------------------------------------------------------------------
    # 1. build the basket matrix (sparse)
    # ------------------------------------------------------------------
    def build_basket(self):
        t0 = time.time()
        # One canonical description per code, for reading rules later.
        self.code_to_desc = (
            self.sales.groupby("StockCode")["Description"]
            .agg(lambda s: s.value_counts().idxmax()).to_dict()
        )

        # Presence (not quantity): a product is in a basket or it isn't.
        pairs = self.sales[["Invoice", "StockCode"]].drop_duplicates()
        invoices = pd.Categorical(pairs["Invoice"])
        products = pd.Categorical(pairs["StockCode"])

        # Sparse one-hot: rows = invoices, cols = products.
        from scipy.sparse import csr_matrix
        data = np.ones(len(pairs), dtype="uint8")
        mat = csr_matrix((data, (invoices.codes, products.codes)),
                         shape=(len(invoices.categories), len(products.categories)))
        self.basket = pd.DataFrame.sparse.from_spmatrix(
            mat, index=invoices.categories, columns=products.categories
        ).astype(bool)

        n_baskets, n_products = self.basket.shape
        self._say(f"Basket matrix: {n_baskets:,} invoices x {n_products:,} "
                  f"products (sparse).  built in {time.time() - t0:.1f}s")
        return self

    # ------------------------------------------------------------------
    # 2. frequent itemsets
    # ------------------------------------------------------------------
    def find_itemsets(self):
        # Pre-run size warning: how many products clear the floor on their own?
        col_support = self.basket.mean(axis=0)
        n_clear = int((col_support >= self.min_support).sum())
        self._say(f"Products clearing {self.min_support:.1%} support: "
                  f"{n_clear:,} of {self.basket.shape[1]:,}.")
        if n_clear > 800:
            self._say("  (that's a lot of single items; itemset search may be "
                      "slow. Consider raising min_support.)")

        t0 = time.time()
        self.itemsets = fpgrowth(self.basket, min_support=self.min_support,
                                 use_colnames=True)
        self.itemsets["n_items"] = self.itemsets["itemsets"].apply(len)
        self._say(f"Frequent itemsets: {len(self.itemsets):,}  "
                  f"(found in {time.time() - t0:.1f}s)")
        return self

    # ------------------------------------------------------------------
    # 3. rules
    # ------------------------------------------------------------------
    def make_rules(self):
        if (self.itemsets["n_items"] >= 2).sum() == 0:
            self._say("No itemsets of 2+ products at this support; "
                      "nothing to make rules from. Lower min_support.")
            self.rules = pd.DataFrame()
            return self

        t0 = time.time()
        rules = association_rules(self.itemsets, metric="lift",
                                  min_threshold=self.min_lift)
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

        # Readable descriptions alongside the coded sets.
        rows = rules.itertuples()
        if _HAS_TQDM:
            rows = tqdm(rows, total=len(rules), desc="naming rules")
        ante, cons = [], []
        for r in rows:
            ante.append(self._names(r.antecedents))
            cons.append(self._names(r.consequents))
        rules["antecedent_names"] = ante
        rules["consequent_names"] = cons

        # Classify how the two sides relate, after stripping variant words.
        overlaps = [self._family_overlap(a, c) for a, c in zip(ante, cons)]
        rules["family_overlap"] = np.round(overlaps, 3)
        rules["family_relation"] = [self._relation(o) for o in overlaps]

        self.rules = rules
        self._say(f"Rules (lift >= {self.min_lift}): {len(rules):,}  "
                  f"(in {time.time() - t0:.1f}s)")
        counts = rules["family_relation"].value_counts()
        self._say("  by relation: " +
                  ", ".join(f"{k} {v}" for k, v in counts.items()))
        self._save_table(
            rules[["antecedent_names", "consequent_names", "support",
                   "confidence", "lift", "family_overlap", "family_relation"]],
            "association_rules"
        )
        return self

    @staticmethod
    def _core_words(text):
        """Words left after dropping colour/pattern/size/filler words. These
        identify the product itself, so two variants of one product share them."""
        words = re.findall(r"[A-Z]+", text.upper())
        return {w for w in words if w not in VARIANT_WORDS and len(w) > 1}

    def _family_overlap(self, a, b):
        """Share of core words common to both sides (0 to 1). High means the
        two sides are the same product or product type; low means genuinely
        different products."""
        wa, wb = self._core_words(a), self._core_words(b)
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / min(len(wa), len(wb))

    @staticmethod
    def _relation(overlap):
        if overlap >= PATTERN_VARIANT_THRESHOLD:
            return "pattern_variant"   # same product, different colour/pattern
        if overlap >= SAME_TYPE_THRESHOLD:
            return "same_type"         # same product type, different sub-product
        return "cross_type"            # genuinely different products

    # ------------------------------------------------------------------
    # 4. report
    # ------------------------------------------------------------------
    def top_rules(self, n=15):
        if self.rules is None or self.rules.empty:
            return self
        cols = ["antecedent_names", "consequent_names", "support",
                "confidence", "lift"]

        cross = self.rules[self.rules["family_relation"] == "cross_type"]
        variant = self.rules[self.rules["family_relation"] == "pattern_variant"]

        # The interesting table: genuinely different products bought together.
        self._save_table(cross[cols].head(n), "top_rules_cross_type")
        # The pattern-collector table: same product, different colour/pattern.
        self._save_table(variant[cols].head(n), "top_rules_pattern_variant")

        if self.verbose:
            print(f"\nTop {n} CROSS-TYPE rules by lift (different products "
                  f"bought together, the interesting ones):")
            with pd.option_context("display.max_colwidth", 35,
                                   "display.width", 140):
                if cross.empty:
                    print("  (none at this support; loosen min_support or "
                          "check the variant word list)")
                else:
                    print(cross[cols].head(n).round(3).to_string(index=False))

        # A scatter of the rule set: support vs confidence, sized by lift,
        # coloured by relation so the variants and cross-type rules separate.
        palette = {"pattern_variant": "#cbd5e0",
                   "same_type": "#90cdf4",
                   "cross_type": "#c53030"}
        fig, ax = plt.subplots(figsize=(8, 6))
        for rel, colour in palette.items():
            sub = self.rules[self.rules["family_relation"] == rel]
            ax.scatter(sub["support"], sub["confidence"], s=sub["lift"] * 4,
                       alpha=0.4, color=colour, label=rel)
        ax.set_xlabel("support")
        ax.set_ylabel("confidence")
        ax.set_title("Association rules (size = lift, colour = relation)")
        ax.legend(markerscale=1, framealpha=0.9)
        ax.grid(alpha=0.3)
        self._save_fig(fig, "rules_scatter")
        return self

    # ------------------------------------------------------------------
    # notes, orchestration
    # ------------------------------------------------------------------
    def _write_notes(self):
        n_rules = 0 if self.rules is None else len(self.rules)
        notes = (
            "Basket analysis notes\n"
            "=====================\n"
            "Unit: invoice (not customer). Run on ALL valid invoices, "
            "anonymous baskets included.\n"
            "Items identified by StockCode; presence/absence, not quantity.\n"
            f"min_support = {self.min_support} ; min_lift = {self.min_lift}.\n"
            f"Rules found: {n_rules}.\n"
            "Ranked by lift (popularity-independent). Each rule is classified "
            "by 'family_relation', from the overlap of core words after "
            "stripping colour/pattern/size words: 'pattern_variant' (same "
            "product, different colour/pattern), 'same_type' (same product "
            "type, different sub-product), 'cross_type' (genuinely different "
            "products, the interesting ones). family_overlap is the 0-1 score "
            "behind the label.\n"
            "Limit: rules show co-occurrence, not cause; both items may be "
            "driven by season or promotion.\n"
            f"Phantom invoices excluded: {sorted(DEFAULT_PHANTOM_INVOICES)}.\n"
        )
        (self.tbl_dir / "basket_notes.txt").write_text(notes, encoding="utf-8")

    def run(self):
        if self.verbose:
            print("=" * 64)
            print(f"BASKET ANALYSIS (min_support={self.min_support:.1%})")
            print("=" * 64)
        (self.build_basket()
             .find_itemsets()
             .make_rules()
             .top_rules())
        self._write_notes()
        self._say(f"\nSaved figures to {self.fig_dir} and tables to {self.tbl_dir}.")
        return self


if __name__ == "__main__":
    base = Path("kaggle_customer_intelligence")
    p = base / "clean_sales.parquet"
    if p.exists():
        sales = pd.read_parquet(p)
    else:
        sales = pd.read_csv(base / "clean_sales.csv", parse_dates=["InvoiceDate"])
    BasketAnalyser(sales, min_support=0.01).run()
