"""
Milestone 2 — Exploratory Data Analysis

Produces 7 figures saved to reports/figures/:
  01_churn_balance.png            — target class distribution
  02_churn_by_contract.png        — contract type is usually the #1 driver
  03_churn_by_tenure.png          — tenure curve, the "honeymoon cliff"
  04_churn_by_monthly_charges.png — price-sensitivity histogram
  05_churn_by_payment.png         — electronic check is a classic red flag
  06_churn_by_province.png        — Canadian geographic angle
  07_churn_by_provider.png        — Big-3 vs flanker dynamics

Console output produces the numbers you'll quote in your case study.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import FIGURES_DIR, MASTER_DATASET

sns.set_style("whitegrid")
PALETTE = {"stayed": "#1D9E75", "churned": "#D85A30"}  # teal vs coral
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 10})


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(MASTER_DATASET)
    print(f"[load] {len(df):,} rows, {len(df.columns)} cols")
    print(f"[load] overall churn rate: {df['churn_flag'].mean():.1%}\n")
    return df


def save_fig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path.name}")


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return churn rate + count per category, sorted by churn rate desc."""
    grouped = df.groupby(col)["churn_flag"].agg(["mean", "count"]).reset_index()
    grouped.columns = [col, "churn_rate", "n_customers"]
    return grouped.sort_values("churn_rate", ascending=False)


# ---- Plots ----

def plot_target_balance(df: pd.DataFrame) -> None:
    counts = df["churn_flag"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(
        ["Stayed", "Churned"], counts.values,
        color=[PALETTE["stayed"], PALETTE["churned"]],
    )
    for bar, val in zip(bars, counts.values):
        pct = val / counts.sum()
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
            f"{val:,}\n({pct:.1%})", ha="center", fontsize=10,
        )
    ax.set_ylabel("Customers")
    ax.set_title("Class imbalance — ~26% positive class needs SMOTE in M3")
    ax.set_ylim(0, counts.max() * 1.15)
    save_fig("01_churn_balance.png")


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    order = ["Month-to-month", "One year", "Two year"]
    rates = df.groupby("Contract")["churn_flag"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(rates.index, rates.values, color="#378ADD")
    for bar, val in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{val:.1%}", ha="center", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Churn rate")
    ax.set_title("Contract type — the strongest single churn signal")
    ax.set_ylim(0, rates.max() * 1.25)
    save_fig("02_churn_by_contract.png")


def plot_churn_by_tenure(df: pd.DataFrame) -> None:
    bins = [0, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    rates = df.groupby("tenure_bucket", observed=True)["churn_flag"].mean()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rates.index.astype(str), rates.values, marker="o", color="#D85A30", linewidth=2.5)
    for x, y in zip(range(len(rates)), rates.values):
        ax.annotate(
            f"{y:.0%}", (x, y), textcoords="offset points",
            xytext=(0, 8), ha="center", fontsize=9,
        )
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate")
    ax.set_title("The 'honeymoon cliff' — first 6 months are where customers leave")
    save_fig("03_churn_by_tenure.png")


def plot_churn_by_monthly_charges(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = 30
    ax.hist(
        df[df["churn_flag"] == 0]["MonthlyCharges"], bins=bins,
        alpha=0.65, label="Stayed", color=PALETTE["stayed"],
    )
    ax.hist(
        df[df["churn_flag"] == 1]["MonthlyCharges"], bins=bins,
        alpha=0.65, label="Churned", color=PALETTE["churned"],
    )
    ax.set_xlabel("Monthly charges (CAD)")
    ax.set_ylabel("Customers")
    ax.set_title("Churners pay more — and the distribution shifts right")
    ax.legend()
    save_fig("04_churn_by_monthly_charges.png")


def plot_churn_by_payment(df: pd.DataFrame) -> None:
    rates = churn_rate_by(df, "PaymentMethod")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(rates["PaymentMethod"], rates["churn_rate"], color="#534AB7")
    for bar, val, n in zip(bars, rates["churn_rate"], rates["n_customers"]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%} (n={n:,})", va="center", fontsize=9,
        )
    ax.set_xlabel("Churn rate")
    ax.set_title("Payment method — electronic check users churn 2-3× more")
    ax.set_xlim(0, rates["churn_rate"].max() * 1.30)
    save_fig("05_churn_by_payment.png")


def plot_churn_by_province(df: pd.DataFrame) -> None:
    rates = churn_rate_by(df, "province_code")
    rates = rates[rates["n_customers"] >= 100]  # drop tiny provinces

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(rates["province_code"], rates["churn_rate"], color="#0F6E56")
    for bar, val in zip(bars, rates["churn_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.1%}", ha="center", fontsize=9,
        )
    ax.axhline(df["churn_flag"].mean(), color="#D85A30", linestyle="--", linewidth=1.5,
               label=f"National avg ({df['churn_flag'].mean():.1%})")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn rate by province (≥100 customers)")
    ax.legend(loc="upper right", fontsize=9)
    save_fig("06_churn_by_province.png")


def plot_churn_by_provider(df: pd.DataFrame) -> None:
    rates = churn_rate_by(df, "provider")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#0C447C" if p in ("Rogers", "Bell", "Telus") else "#888780"
              for p in rates["provider"]]
    bars = ax.bar(rates["provider"], rates["churn_rate"], color=colors)
    for bar, val in zip(bars, rates["churn_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.1%}", ha="center", fontsize=9,
        )
    ax.set_ylabel("Churn rate")
    ax.set_title("Big-3 (blue) vs flanker brands (gray)")
    save_fig("07_churn_by_provider.png")


# ---- Summary stats ----

def print_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("KEY CHURN DRIVERS — quote these in your case study")
    print("=" * 70)

    overall = df["churn_flag"].mean()
    print(f"\nOverall churn rate: {overall:.1%}\n")

    for col in ["Contract", "PaymentMethod", "InternetService", "provider"]:
        print(f"\n→ Churn rate by {col}")
        print(churn_rate_by(df, col).to_string(index=False))

    # Affordability angle — the Canadian story
    print("\n" + "=" * 70)
    print("AFFORDABILITY — the Canadian narrative")
    print("=" * 70)
    df["affordability_band"] = pd.cut(
        df["spend_vs_province_avg"],
        bins=[0, 0.3, 0.5, 0.7, 1.0, 10],
        labels=["<30% of avg", "30-50%", "50-70%", "70-100%", ">100% of avg"],
    )
    aff = df.groupby("affordability_band", observed=True)["churn_flag"].agg(["mean", "count"])
    aff.columns = ["churn_rate", "n"]
    print(aff.to_string())

    # Compare top band (highest spend-to-avg) vs bottom band — robust if some are empty
    populated = aff[aff["n"] >= 50]
    if len(populated) >= 2:
        top_band = populated.index[-1]
        bottom_band = populated.index[0]
        lift = populated.loc[top_band, "churn_rate"] / populated.loc[bottom_band, "churn_rate"]
        print(f"\n>> Customers in the '{top_band}' band churn {lift:.2f}× more often")
        print(f"   than those in the '{bottom_band}' band — this is your headline number.\n")
    else:
        print("\n>> Not enough customers per affordability band to compute lift.\n")


def main() -> None:
    df = load_data()

    print("[plot] generating 7 EDA figures...")
    plot_target_balance(df)
    plot_churn_by_contract(df)
    plot_churn_by_tenure(df)
    plot_churn_by_monthly_charges(df)
    plot_churn_by_payment(df)
    plot_churn_by_province(df)
    plot_churn_by_provider(df)
    print()

    print_summary(df)


if __name__ == "__main__":
    main()
