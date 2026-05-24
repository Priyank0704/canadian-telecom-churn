"""
THE FLAGSHIP CHART for the portfolio website.

A two-panel figure that tells the Canadian affordability + churn story
in one visual. This is the screenshot that goes on your portfolio site.

Panel 1: Scatter — monthly charges vs spend-as-pct-of-provincial-avg,
         coloured by churn. Shows the high-spend cluster of leavers.

Panel 2: Bar — churn rate by affordability band. The lift number lives here.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import FIGURES_DIR, MASTER_DATASET

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160,
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

STAY_COLOR = "#1D9E75"
CHURN_COLOR = "#D85A30"


def make_affordability_chart() -> Path:
    df = pd.read_parquet(MASTER_DATASET)

    # spend_vs_province_avg is (customer monthly $) / (provincial avg monthly $)
    # Convert to percentage for readability
    df["spend_pct_of_avg"] = df["spend_vs_province_avg"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- Panel 1: scatter ----
    stayed = df[df["churn_flag"] == 0]
    churned = df[df["churn_flag"] == 1]

    ax1.scatter(stayed["MonthlyCharges"], stayed["spend_pct_of_avg"],
                s=10, alpha=0.25, color=STAY_COLOR, label=f"Stayed (n={len(stayed):,})")
    ax1.scatter(churned["MonthlyCharges"], churned["spend_pct_of_avg"],
                s=14, alpha=0.45, color=CHURN_COLOR, label=f"Churned (n={len(churned):,})")

    # Reference line: 100% of provincial avg
    ax1.axhline(100, color="#444441", linestyle="--", linewidth=1, alpha=0.6)
    ax1.text(df["MonthlyCharges"].max() * 0.98, 103,
             "= provincial avg household telecom spend",
             ha="right", fontsize=8.5, color="#444441", style="italic")

    ax1.set_xlabel("Customer monthly charges (CAD)")
    ax1.set_ylabel("% of provincial avg household telecom spend")
    ax1.set_title("Customers paying near or above their province's average\nare visibly more likely to churn",
                  fontsize=11, loc="left", pad=10)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    # ---- Panel 2: bars ----
    df["affordability_band"] = pd.cut(
        df["spend_pct_of_avg"],
        bins=[0, 30, 50, 70, 100, 1000],
        labels=["<30%", "30-50%", "50-70%", "70-100%", ">100%"],
    )
    band_stats = df.groupby("affordability_band", observed=True)["churn_flag"].agg(["mean", "count"])
    band_stats = band_stats[band_stats["count"] >= 50]

    bars = ax2.bar(band_stats.index.astype(str), band_stats["mean"],
                    color="#534AB7", width=0.7)

    # Highlight the highest band in coral
    if len(bars) > 0:
        bars[-1].set_color(CHURN_COLOR)

    for bar, rate, n in zip(bars, band_stats["mean"], band_stats["count"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                  f"{rate:.1%}", ha="center", fontsize=11, fontweight="bold")
        ax2.text(bar.get_x() + bar.get_width() / 2, -0.025,
                  f"n={n:,}", ha="center", fontsize=8, color="#888780")

    # Lift annotation
    if len(band_stats) >= 2:
        top = band_stats["mean"].iloc[-1]
        bot = band_stats["mean"].iloc[0]
        lift = top / bot if bot > 0 else 0
        ax2.text(0.98, 0.95,
                  f"{lift:.1f}× lift\ntop band vs bottom",
                  transform=ax2.transAxes, ha="right", va="top",
                  fontsize=11, fontweight="bold", color=CHURN_COLOR,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAECE7",
                            edgecolor=CHURN_COLOR, linewidth=1.2))

    ax2.set_ylabel("Churn rate")
    ax2.set_xlabel("Customer spend as % of provincial avg")
    ax2.set_title("Churn rate climbs with affordability burden",
                  fontsize=11, loc="left", pad=10)
    ax2.set_ylim(0, max(band_stats["mean"]) * 1.25)

    # Overall figure title
    fig.suptitle(
        "The Canadian telecom affordability story",
        fontsize=14, fontweight="bold", y=1.02, x=0.07, ha="left",
    )
    fig.text(0.07, 0.97,
              "Customer monthly charges contextualized against StatCan provincial household telecom spending",
              fontsize=9.5, color="#5F5E5A", ha="left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "FLAGSHIP_affordability.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[ok] Flagship chart → {out_path}")
    return out_path


if __name__ == "__main__":
    make_affordability_chart()
