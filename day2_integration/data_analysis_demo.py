"""Read a CSV, compute statistics, and plot a simple chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sales.csv"
OUTPUT_PLOT = Path(__file__).resolve().parents[1] / "data" / "sales_chart.png"


def load_data() -> pd.DataFrame:
    """Load the sales CSV into a pandas DataFrame."""
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def show_summary(df: pd.DataFrame) -> None:
    """Print simple statistics to the terminal."""
    print("First few rows of the data:")
    print(df.head())

    print("\nOverall statistics:")
    print(df.describe(include="all"))

    print("\nTotal revenue by region:")
    print(df.groupby("region")["revenue"].sum())


def create_chart(df: pd.DataFrame) -> None:
    """Create a bar chart showing revenue by region."""
    revenue_by_region = df.groupby("region")["revenue"].sum()

    plt.figure(figsize=(6, 4))
    revenue_by_region.plot(kind="bar", color="skyblue")
    plt.title("Revenue by Region")
    plt.xlabel("Region")
    plt.ylabel("Revenue (USD)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.close()

    print(f"Chart saved to {OUTPUT_PLOT}")


def main() -> None:
    """Run the data analysis demo."""
    print(f"Loading data from {DATA_PATH}")
    df = load_data()

    show_summary(df)
    create_chart(df)
    print("\nData analysis complete. Use the chart in training discussions.")


if __name__ == "__main__":
    main()
