"""
RFM-style customer segmentation for telecom.

Standard retail RFM (Recency, Frequency, Monetary) adapted for a subscription
business where customers haven't "left" each session — we redefine:

  R = lifecycle stage      → 1 / (tenure + 1)   (new customers = high R)
  F = service intensity    → count of paid add-on services
  M = monetary value       → MonthlyCharges

K-means is applied to standardized RFM features. Optimal K is chosen
via elbow + silhouette. Each cluster gets a business-readable label.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import FIGURES_DIR, MASTER_DATASET, PROCESSED_DIR, RANDOM_STATE

ADD_ON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R, F, M for each customer."""
    rfm = pd.DataFrame(index=df.index)
    rfm["R_lifecycle"] = 1.0 / (df["tenure"] + 1.0)
    # "Yes" counts as 1, anything else (No, "No internet service") as 0
    rfm["F_services"] = sum((df[c] == "Yes").astype(int) for c in ADD_ON_SERVICES)
    rfm["M_monthly"] = df["MonthlyCharges"]
    return rfm


def pick_k(scaled: np.ndarray, k_range: range = range(2, 9)) -> dict:
    """Run elbow + silhouette across K. Returns metrics dict."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(scaled)
        inertias.append(km.inertia_)
        # silhouette is O(n^2) — sample for speed
        sample = np.random.default_rng(RANDOM_STATE).choice(
            len(scaled), size=min(2000, len(scaled)), replace=False,
        )
        silhouettes.append(silhouette_score(scaled[sample], labels[sample]))
    return {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes}


def plot_k_selection(metrics: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

    ax1.plot(metrics["k"], metrics["inertia"], marker="o", color="#534AB7", linewidth=2)
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia (within-cluster SSE)")
    ax1.set_title("Elbow method")
    ax1.grid(alpha=0.3)

    ax2.plot(metrics["k"], metrics["silhouette"], marker="o", color="#0F6E56", linewidth=2)
    best_k = metrics["k"][int(np.argmax(metrics["silhouette"]))]
    ax2.axvline(best_k, color="#D85A30", linestyle="--", linewidth=1.5,
                label=f"Best silhouette: K={best_k}")
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette score")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "08_k_selection.png", bbox_inches="tight")
    plt.close()
    print(f"  saved → 08_k_selection.png  (best K by silhouette = {best_k})")
    return best_k


def label_clusters(profile: pd.DataFrame) -> dict:
    """
    Assign business-readable names to clusters based on their RFM centroids.

    Uses absolute thresholds (tenure in months, monthly $) so labels remain
    meaningful regardless of how percentile-ranks happen to fall.
    """
    names = {}
    for cluster_id in profile.index:
        tenure = profile.loc[cluster_id, "tenure"]
        monthly = profile.loc[cluster_id, "MonthlyCharges"]
        services = profile.loc[cluster_id, "F_services"]

        is_new = tenure < 12
        is_loyal = tenure > 24
        is_high_spend = monthly > 75
        is_low_spend = monthly < 50
        is_bundled = services >= 3

        if is_new and is_high_spend:
            names[cluster_id] = "New & high-spend (FLIGHT RISK)"
        elif is_new:
            names[cluster_id] = "New & light"
        elif is_loyal and is_high_spend and is_bundled:
            names[cluster_id] = "Loyal premium"
        elif is_loyal and is_high_spend:
            names[cluster_id] = "Loyal high-spend (no bundle)"
        elif is_loyal and is_low_spend:
            names[cluster_id] = "Loyal value"
        elif is_bundled:
            names[cluster_id] = "Bundled engaged"
        else:
            names[cluster_id] = "Mid-tenure standard"
    return names


def main(k_override: int = None) -> None:
    print(f"[load] {MASTER_DATASET}")
    df = pd.read_parquet(MASTER_DATASET)

    rfm = build_rfm_features(df)
    print(f"[features] R/F/M built — shape {rfm.shape}")
    print(rfm.describe().round(2).to_string())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)

    print("\n[k-selection] running elbow + silhouette for K=2..8...")
    metrics = pick_k(scaled)
    best_k_auto = plot_k_selection(metrics)

    K = k_override or best_k_auto
    print(f"\n[cluster] fitting final K-means with K={K}")
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
    df["cluster"] = km.fit_predict(scaled)

    # Cluster centroids in original (un-scaled) space
    profile = df.groupby("cluster")[["tenure", "MonthlyCharges"]].mean()
    profile["F_services"] = rfm.groupby(df["cluster"])["F_services"].mean()
    profile["R_lifecycle"] = rfm.groupby(df["cluster"])["R_lifecycle"].mean()
    profile["size"] = df.groupby("cluster").size()
    profile["churn_rate"] = df.groupby("cluster")["churn_flag"].mean()
    profile = profile[["size", "tenure", "R_lifecycle", "F_services",
                       "MonthlyCharges", "churn_rate"]]

    cluster_names = label_clusters(profile)
    profile["segment_name"] = profile.index.map(cluster_names)
    df["segment_name"] = df["cluster"].map(cluster_names)

    print("\n[profile] Cluster summary:")
    print(profile.round(3).to_string())

    # Persist
    out_path = PROCESSED_DIR / "customers_segmented.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[ok] Segmented dataset → {out_path}")

    # Final chart — churn rate per segment
    plot_segment_churn(profile)


def plot_segment_churn(profile: pd.DataFrame) -> None:
    """Bar chart of churn rate by named segment, sorted high to low."""
    sorted_profile = profile.sort_values("churn_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["#D85A30" if "FLIGHT RISK" in name else "#534AB7"
              for name in sorted_profile["segment_name"]]
    bars = ax.barh(sorted_profile["segment_name"], sorted_profile["churn_rate"],
                    color=colors)

    for bar, rate, size in zip(bars, sorted_profile["churn_rate"], sorted_profile["size"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{rate:.1%}  (n={size:,})", va="center", fontsize=9)

    ax.set_xlabel("Churn rate within segment")
    ax.set_title("Customer segments ranked by churn risk", fontsize=12, loc="left")
    ax.set_xlim(0, sorted_profile["churn_rate"].max() * 1.30)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_segment_churn.png", bbox_inches="tight")
    plt.close()
    print("  saved → 09_segment_churn.png")


if __name__ == "__main__":
    main()
