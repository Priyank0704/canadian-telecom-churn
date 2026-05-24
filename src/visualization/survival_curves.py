"""
Cohort survival analysis.

Uses the Kaplan-Meier estimator from lifelines to answer:
    "Among customers who signed up X months ago, what fraction are still here?"

The chart shows three curves by Contract type. The two-year curve stays flat;
the month-to-month curve drops quickly in the first 12 months. This is the
visualization that lets a retention manager answer 'when do we lose them?'
which then drives 'when should we intervene?'

Outputs:
  reports/figures/14_km_survival_by_contract.png
  reports/figures/15_km_survival_by_segment.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import FIGURES_DIR, PROCESSED_DIR

SEGMENTED_PATH = PROCESSED_DIR / "customers_segmented.parquet"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10})

CONTRACT_COLORS = {
    "Month-to-month": "#D85A30",  # coral — high churn
    "One year":       "#EF9F27",  # amber — medium
    "Two year":       "#0F6E56",  # teal  — low churn
}


def km_by_contract(df: pd.DataFrame) -> None:
    """Kaplan-Meier survival curve per contract type."""
    fig, ax = plt.subplots(figsize=(9, 5))

    contracts = ["Month-to-month", "One year", "Two year"]
    median_survivals = {}

    for contract in contracts:
        sub = df[df["Contract"] == contract]
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=sub["tenure"],
            event_observed=sub["churn_flag"],
            label=f"{contract} (n={len(sub):,})",
        )
        kmf.plot_survival_function(ax=ax, color=CONTRACT_COLORS[contract],
                                    linewidth=2.5, ci_show=True, ci_alpha=0.15)
        median_survivals[contract] = kmf.median_survival_time_

    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Probability of still being a customer")
    ax.set_title("Customer survival by contract type — the 'honeymoon cliff' in action",
                  fontsize=12, loc="left", pad=10)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=False)

    # Log-rank test between month-to-month and two-year
    mtm = df[df["Contract"] == "Month-to-month"]
    two = df[df["Contract"] == "Two year"]
    lr = logrank_test(
        mtm["tenure"], two["tenure"],
        event_observed_A=mtm["churn_flag"], event_observed_B=two["churn_flag"],
    )
    ax.text(
        0.98, 0.97,
        f"Log-rank test\n(month-to-month vs two-year)\np = {lr.p_value:.2e}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#26215C",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEEDFE", edgecolor="#534AB7"),
    )

    plt.tight_layout()
    out = FIGURES_DIR / "14_km_survival_by_contract.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out.name}")
    print("\n  Median survival times (months):")
    for contract, median in median_survivals.items():
        m = "never reached" if median is None or pd.isna(median) else f"{median:.0f}"
        print(f"    {contract:18s}: {m}")
    print(f"  Log-rank p-value: {lr.p_value:.2e}")


def km_by_segment(df: pd.DataFrame) -> None:
    """KM by the customer segments we built in M2."""
    fig, ax = plt.subplots(figsize=(9, 5))

    palette = ["#534AB7", "#0F6E56", "#D85A30", "#185FA5", "#854F0B", "#993556"]
    segments = df["segment_name"].value_counts().index.tolist()

    for color, segment in zip(palette, segments):
        sub = df[df["segment_name"] == segment]
        if len(sub) < 30:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=sub["tenure"],
            event_observed=sub["churn_flag"],
            label=f"{segment} (n={len(sub):,})",
        )
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2.2, ci_show=False)

    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Probability of still being a customer")
    ax.set_title("Customer survival by RFM segment", fontsize=12, loc="left", pad=10)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "15_km_survival_by_segment.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out.name}")


def main() -> None:
    print(f"[load] {SEGMENTED_PATH}")
    df = pd.read_parquet(SEGMENTED_PATH)
    print(f"[load] {len(df):,} customers")

    print("\n[plot] Kaplan-Meier by contract type")
    km_by_contract(df)

    print("\n[plot] Kaplan-Meier by RFM segment")
    km_by_segment(df)


if __name__ == "__main__":
    main()
