from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path(__file__).with_name("processed_chemical_experiment.csv")
OUTPUT_CHART = Path(__file__).with_name("experiment_summary.png")


def load_experiment_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the prepared chemical experiment dataset."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"The dataset at {path} is empty.")
    return df


def summarize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for each numeric column."""
    numeric_df = df.select_dtypes(include="number")
    summary = numeric_df.agg(["count", "mean", "std", "min", "max"]).T
    summary["coefficient_of_variation_%"] = (
        summary["std"] / summary["mean"].abs() * 100
    )
    return summary.round(4)


def compare_experiments_by_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation for each experiment."""
    grouped = (
        df.groupby("Experiment_ID", as_index=False)["Temperature_C"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Mean_Temperature_C", "std": "Std_Temperature_C"})
        .reset_index()
    )
    return grouped.round(4)


def identify_most_variable_measurement(df: pd.DataFrame) -> tuple[str, float]:
    """Identify the numeric measurement with the largest relative variability."""
    numeric_df = df.select_dtypes(include="number")
    cv_by_column = (
        numeric_df.std(ddof=1) / numeric_df.mean().abs() * 100
    ).dropna()
    most_variable = cv_by_column.idxmax()
    return most_variable, round(float(cv_by_column.max()), 2)


def plot_experiment_means(df: pd.DataFrame, output_path: Path = OUTPUT_CHART) -> None:
    """Create a simple bar chart comparing mean temperature across experiments."""
    grouped = compare_experiments_by_temperature(df)
    plt.figure(figsize=(8, 5))
    plt.bar(grouped["Experiment_ID"], grouped["Mean_Temperature_C"], color="steelblue")
    plt.axhline(grouped["Mean_Temperature_C"].mean(), color="red", linestyle="--", linewidth=1.5)
    plt.title("Mean Temperature by Experiment")
    plt.xlabel("Experiment ID")
    plt.ylabel("Mean Temperature (°C)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    df = load_experiment_data()

    print("=== Chemical Experiment Dataset ===")
    print(df.head().to_string(index=False))
    print("\n=== Summary Statistics for Numeric Columns ===")
    print(summarize_numeric_columns(df).to_string())

    print("\n=== Mean and Standard Deviation by Experiment ===")
    print(compare_experiments_by_temperature(df).to_string(index=False))

    most_variable, cv_value = identify_most_variable_measurement(df)
    print(f"\nMost variable measurement: {most_variable} ({cv_value}% coefficient of variation)")

    plot_experiment_means(df)
    print(f"\nSaved chart to: {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
