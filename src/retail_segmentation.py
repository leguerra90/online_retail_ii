"""
Online Retail II — Step 4: segmentation

Clusters customers with k-means on six features, then profiles each cluster
in plain terms so they can be named.

Features (one row per customer):
    recency            days since last purchase (from the RFM step)
    frequency          distinct invoices
    monetary           gross total spend
    spend_per_invoice  monetary / frequency
    items_per_invoice  total units / frequency
    variety_ratio      distinct products / frequency
                       (history-wide repertoire breadth; shaky for one-time
                        buyers, where it is just the size of their one basket)

Recency/frequency/monetary say when, how often, how much. The two
basket measures say how each trip looks. The variety ratio says how broad the
customer's range is across their whole history, which is close to independent
of the others and is what should expose wholesalers (heavy spend, narrow range
bought again and again -> low ratio).

Pipeline note for k-means: it uses straight-line distance, so a feature on a
bigger scale dominates, and its centroids chase outliers. So I logged the skewed
features, then standardised all six to z-scores. Monetary, frequency, and both
basket measures are logged; recency is left raw (skewed but not savage); the
variety ratio is logged too.

Two-stage use, because choosing k is a human decision:

    seg = CustomerSegmenter(rfm, customer_sales)
    seg.run()                 # features, transform, heatmap, elbow + silhouette
    # look at the charts, pick k, then:
    seg.fit(6).profile().save()

then:  seg.run(k=6).save()
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

DEFAULT_PHANTOM_INVOICES = ("541431", "581483")
LOG_FEATURES = ["frequency", "monetary", "spend_per_invoice",
                "items_per_invoice", "variety_ratio"]
FEATURES = ["recency", "frequency", "monetary", "spend_per_invoice",
            "items_per_invoice", "variety_ratio"]
COLOUR = "#2b6cb0"

# Names for the k=6 segmentation, tied to the fixed-seed run (random_state=42).
# k-means numbers clusters arbitrarily, so check these against the printed
# profile. label_clusters() warns if the clusters present don't match the keys.
CLUSTER_NAMES_K6 = {
    0: "Big-basket bulk buyers",
    1: "Frequent regulars",
    2: "Occasional low-value",
    3: "Lapsed one-off buyers",
    4: "Champions",
    5: "Dormant / lost",
}


class CustomerSegmenter:
    def __init__(self, rfm, customer_sales,
                 exclude_invoices=DEFAULT_PHANTOM_INVOICES,
                 k_range=range(2, 16), random_state=42,
                 out_dir="results/segmentation",
                 verbose=True):
        self.rfm = rfm.copy()

        sales = customer_sales.copy()
        sales = sales[sales["CustomerID"].notna()]
        drop = {str(i) for i in (exclude_invoices or [])}
        sales = sales[~sales["Invoice"].astype(str).isin(drop)]
        self.sales = sales.reset_index(drop=True)

        self.k_range = list(k_range)
        self.random_state = random_state
        self.verbose = verbose

        self.features = None   # raw feature table (+ cluster after fit)
        self.scaled = None     # standardised features used for clustering
        self.k_search = None   # inertia and silhouette per k
        self.model = None
        self.k = None

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

    # ------------------------------------------------------------------
    # 1. build features
    # ------------------------------------------------------------------
    def build_features(self):
        g = self.sales.groupby("CustomerID")
        agg = g.agg(
            frequency=("Invoice", "nunique"),
            monetary=("Revenue", "sum"),
            total_units=("Quantity", "sum"),
            distinct_products=("StockCode", "nunique"),
            country=("Country", lambda s: s.value_counts().idxmax()),
        )
        agg["spend_per_invoice"] = agg["monetary"] / agg["frequency"]
        agg["items_per_invoice"] = agg["total_units"] / agg["frequency"]
        agg["variety_ratio"] = agg["distinct_products"] / agg["frequency"]

        # Recency comes from the RFM step (it carries the reference date).
        feats = agg.join(self.rfm["recency"], how="inner")
        self.features = feats[["recency", "frequency", "monetary",
                               "spend_per_invoice", "items_per_invoice",
                               "variety_ratio", "country"]].copy()
        self._say(f"Built features for {len(self.features):,} customers.")
        return self

    # ------------------------------------------------------------------
    # 2. transform
    # ------------------------------------------------------------------
    def transform(self):
        x = self.features[FEATURES].copy()
        for col in LOG_FEATURES:
            x[col] = np.log1p(x[col])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(x)
        self.scaled = pd.DataFrame(scaled, index=x.index, columns=FEATURES)
        self._save_table(self.scaled, "scaled_features")
        self._say("Logged skewed features and standardised all six to z-scores.")
        return self

    # ------------------------------------------------------------------
    # 3. correlation heatmap
    # ------------------------------------------------------------------
    def correlation_heatmap(self):
        # On the transformed features, since that is what k-means sees.
        # (Standardising doesn't change correlation, so self.scaled is fine.)
        corr = self.scaled.corr()
        self._save_table(corr, "feature_correlation")

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(FEATURES)))
        ax.set_yticks(range(len(FEATURES)))
        ax.set_xticklabels(FEATURES, rotation=45, ha="right")
        ax.set_yticklabels(FEATURES)
        for i in range(len(FEATURES)):
            for j in range(len(FEATURES)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center",
                        va="center", fontsize=8)
        ax.set_title("Feature correlation (transformed)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        self._save_fig(fig, "feature_correlation")
        self._say("Saved correlation heatmap (check the variety fix worked).")
        return self

    # ------------------------------------------------------------------
    # 4. choose k
    # ------------------------------------------------------------------
    def search_k(self, silhouette_sample=2000):
        rows = []
        for k in self.k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(self.scaled)
            sil = silhouette_score(self.scaled, labels,
                                   sample_size=silhouette_sample,
                                   random_state=self.random_state)
            rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
            self._say(f"  k={k:>2}  inertia={km.inertia_:>12,.0f}  "
                      f"silhouette={sil:.3f}")
        self.k_search = pd.DataFrame(rows).set_index("k")
        self._save_table(self.k_search, "k_search")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(self.k_search.index, self.k_search["inertia"],
                     marker="o", color=COLOUR)
        axes[0].set_title("Elbow (inertia)")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("inertia")
        axes[0].grid(alpha=0.3)
        axes[1].plot(self.k_search.index, self.k_search["silhouette"],
                     marker="o", color=COLOUR)
        axes[1].set_title("Silhouette")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("score")
        axes[1].grid(alpha=0.3)
        self._save_fig(fig, "k_search")
        return self

    # ------------------------------------------------------------------
    # 5. fit a chosen k
    # ------------------------------------------------------------------
    def fit(self, k):
        self.k = k
        self.model = KMeans(n_clusters=k, n_init=10,
                            random_state=self.random_state)
        self.features["cluster"] = self.model.fit_predict(self.scaled)
        self._say(f"Fitted k-means with k={k}.")
        return self

    # ------------------------------------------------------------------
    # 6. profile clusters
    # ------------------------------------------------------------------
    def profile(self):
        if "cluster" not in self.features:
            raise RuntimeError("Call fit(k) before profile().")

        f = self.features
        prof = f.groupby("cluster").agg(
            customers=("recency", "size"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
            spend_per_invoice=("spend_per_invoice", "mean"),
            items_per_invoice=("items_per_invoice", "mean"),
            variety_ratio=("variety_ratio", "mean"),
        )
        prof["pct_customers"] = (prof["customers"] / len(f) * 100).round(1)
        prof["uk_share"] = (
            f.assign(uk=f["country"].eq("United Kingdom"))
             .groupby("cluster")["uk"].mean().round(3)
        )
        prof["top_country"] = (
            f.groupby("cluster")["country"].agg(lambda s: s.value_counts().idxmax())
        )
        prof = prof.round(2)
        self._save_table(prof, "cluster_profile")

        if self.verbose:
            print("\nCluster profile (raw averages):")
            print(prof.to_string())
        return self

    # ------------------------------------------------------------------
    # 7. name the clusters
    # ------------------------------------------------------------------
    def label_clusters(self, mapping=None):
        if "cluster" not in self.features:
            raise RuntimeError("Call fit(k) before label_clusters().")
        if mapping is None:
            if self.k == 6:
                mapping = CLUSTER_NAMES_K6
            else:
                raise ValueError(
                    f"No default names for k={self.k}; pass a mapping dict.")

        present = set(self.features["cluster"].unique())
        if present != set(mapping):
            self._say(f"WARNING: clusters present {sorted(present)} do not match "
                      f"the name mapping keys {sorted(mapping)}. The numbering may "
                      f"have shifted; check against the profile before trusting "
                      f"the labels.")
        self.cluster_names = mapping
        self.features["segment"] = self.features["cluster"].map(mapping)
        self._say(f"Labelled clusters: {mapping}")
        return self

    # ------------------------------------------------------------------
    # 8. plots
    # ------------------------------------------------------------------
    def _colours(self):
        cmap = plt.get_cmap("tab10")
        clusters = sorted(self.features["cluster"].unique())
        return {c: cmap(i % 10) for i, c in enumerate(clusters)}

    def _label_for(self, c):
        names = getattr(self, "cluster_names", None)
        return names[c] if names else f"cluster {c}"

    def plot_pca(self):
        """PCA biplot: clusters as points, feature directions as arrows.
        Interpretable axes, honest about overlap. Put this one in the writeup."""
        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(self.scaled)
        var = pca.explained_variance_ratio_ * 100
        colours = self._colours()

        fig, ax = plt.subplots(figsize=(10, 8))
        for c in sorted(self.features["cluster"].unique()):
            m = self.features["cluster"].values == c
            ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.35,
                       color=colours[c], label=self._label_for(c))

        # Feature arrows, scaled to sit over the cloud.
        load = pca.components_.T
        scale = np.abs(coords).max() * 0.85
        for i, feat in enumerate(FEATURES):
            ax.arrow(0, 0, load[i, 0] * scale, load[i, 1] * scale,
                     color="black", alpha=0.8, head_width=scale * 0.02,
                     length_includes_head=True)
            ax.text(load[i, 0] * scale * 1.08, load[i, 1] * scale * 1.08,
                    feat, fontsize=9, ha="center")

        ax.set_xlabel(f"PC1 ({var[0]:.0f}% of variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.0f}% of variance)")
        ax.set_title("Customer segments in PCA space (biplot)")
        ax.legend(markerscale=2, framealpha=0.9, loc="best")
        ax.grid(alpha=0.2)
        self._save_fig(fig, "segments_pca")
        self._say(f"Saved PCA biplot (PC1+PC2 hold {var[0] + var[1]:.0f}% "
                  f"of variance).")
        return self

    def plot_umap(self, n_neighbors=15, min_dist=0.1):
        """UMAP: emphasises separation. Layout and distances are NOT meaningful,
        so read it only as 'do the six clusters hold together visually'."""
        try:
            import umap
        except ImportError:
            self._say("umap-learn not installed; skipping UMAP. "
                      "Install with: pip install umap-learn")
            return self

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=self.random_state)
        coords = reducer.fit_transform(self.scaled)
        colours = self._colours()

        fig, ax = plt.subplots(figsize=(10, 8))
        for c in sorted(self.features["cluster"].unique()):
            m = self.features["cluster"].values == c
            ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.35,
                       color=colours[c], label=self._label_for(c))
        ax.set_xlabel("UMAP 1 (arbitrary)")
        ax.set_ylabel("UMAP 2 (arbitrary)")
        ax.set_title("Customer segments via UMAP (separation only; "
                     "distances not meaningful)")
        ax.legend(markerscale=2, framealpha=0.9, loc="best")
        self._save_fig(fig, "segments_umap")
        self._say("Saved UMAP plot.")
        return self
    def _write_notes(self):
        notes = (
            "Segmentation notes\n"
            "==================\n"
            f"Features: {', '.join(FEATURES)}.\n"
            "variety_ratio = distinct products / invoices (history-wide range). "
            "Shaky for one-time buyers, where it is just their single basket "
            "size; only meaningful for repeat customers.\n"
            f"Logged before scaling: {', '.join(LOG_FEATURES)} (recency left raw). "
            "All six then standardised to z-scores.\n"
            f"Phantom invoices excluded: {sorted(DEFAULT_PHANTOM_INVOICES)}.\n"
            f"k-means: n_init=10, random_state={self.random_state}. "
            "Silhouette computed on a sample for speed.\n"
            "Country is profiled per cluster, not used as a clustering feature.\n"
        )
        names = getattr(self, "cluster_names", None)
        if names:
            notes += "Cluster names (k=%d): %s\n" % (
                self.k, ", ".join(f"{c} = {n}" for c, n in names.items()))
        (self.tbl_dir / "segmentation_notes.txt").write_text(notes, encoding="utf-8")

    def save(self, data_dir="kaggle_customer_intelligence"):
        self._write_notes()
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = data_dir / "customer_segments.parquet"
            self.features.to_parquet(path)
        except Exception:
            path = data_dir / "customer_segments.csv"
            self.features.to_csv(path)
            self._say("pyarrow not found, saved segments as CSV.")
        self._say(f"Saved labelled customers to: {path}")
        return self

    def run(self, k=None, silhouette_sample=2000, search=True):
        if self.verbose:
            print("=" * 64)
            print("SEGMENTATION")
            print("=" * 64)
        self.build_features().transform().correlation_heatmap()
        if search:
            self.search_k(silhouette_sample=silhouette_sample)
        if k is not None:
            self.fit(k).profile()
            try:
                self.label_clusters()
            except ValueError:
                self._say(f"No default names for k={k}; "
                          "call label_clusters(mapping) yourself.")
            self.plot_pca().plot_umap()
        else:
            self._say("\nReview the elbow and silhouette charts, then call "
                      ".fit(k).profile().label_clusters().plot_pca().plot_umap()"
                      ".save() with your chosen k.")
        self._say(f"\nSaved figures to {self.fig_dir} and tables to {self.tbl_dir}.")
        return self


if __name__ == "__main__":
    base = Path("kaggle_customer_intelligence")

    def _load(parquet, csv, **kw):
        p = base / parquet
        if p.exists():
            return pd.read_parquet(p)
        return pd.read_csv(base / csv, **kw)

    rfm = _load("rfm_features.parquet", "rfm_features.csv", index_col=0)
    sales = _load("clean_sales.parquet", "clean_sales.csv",
                  parse_dates=["InvoiceDate"])
    if not pd.api.types.is_integer_dtype(sales["CustomerID"]):
        sales["CustomerID"] = sales["CustomerID"].astype("Int64")
    customer_sales = sales[sales["CustomerID"].notna()]

    seg = CustomerSegmenter(rfm, customer_sales)
    # k chosen from the elbow/silhouette charts. This fits, profiles, names the
    # clusters, draws the PCA biplot and UMAP plot, then saves everything.
    seg.run(k=6).save()
