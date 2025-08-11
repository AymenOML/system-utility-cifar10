import pandas as pd
import numpy as np

def load_datasets(stats_path, sys_path, conf_path):
    """Load all three datasets."""
    all_clients_stats = pd.read_csv(stats_path)
    system_metrics = pd.read_csv(sys_path)
    confusion_tidy = pd.read_csv(conf_path)
    return all_clients_stats, system_metrics, confusion_tidy


def aggregate_confusion(confusion_tidy):
    """Aggregate confusion matrix values per client & round."""
    confusion_agg = (
        confusion_tidy
        .groupby(["client_id", "round"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"client_id": "client_rank", "value": "confusion_mean"})
    )
    return confusion_agg


def merge_datasets(all_clients_stats, system_metrics, confusion_agg):
    """Merge statistical, system metrics, and aggregated confusion matrix."""
    merged = pd.merge(
        all_clients_stats,
        system_metrics,
        on=["client_rank", "round"],
        suffixes=("_stat", "_sys")
    )
    merged_all = pd.merge(
        merged,
        confusion_agg,
        on=["client_rank", "round"],
        how="left"
    )
    return merged_all


def classify_equal_importance(df):
    """
    Apply equal-importance median threshold classification.
    Returns DataFrame with classification_score and class_label.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    thresholds = df[numeric_cols].median()
    binary_scores = (df[numeric_cols] > thresholds).astype(int)
    df["classification_score"] = binary_scores.sum(axis=1)
    cutoff = len(numeric_cols) / 2
    df["class_label"] = (df["classification_score"] >= cutoff).astype(int)
    return df, thresholds, cutoff


def save_dataset(df, output_path):
    """Save the final DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"File saved as: {output_path}")


if __name__ == "__main__":
    # === Paths to your CSV files ===
    stats_path = "Code/Labelling/Data/all_clients_stats_normalized.csv"
    sys_path = "Code/Labelling/Data/client_system_metrics_normalized.csv"
    conf_path = "Code/Labelling/Data/confusion_matrices_tidy.csv"
    output_path = "Code/Labelling/Data/clients_labeled_equal_importance_with_confusion.csv"

    # === Workflow ===
    all_clients_stats, system_metrics, confusion_tidy = load_datasets(stats_path, sys_path, conf_path)
    confusion_agg = aggregate_confusion(confusion_tidy)
    merged_all = merge_datasets(all_clients_stats, system_metrics, confusion_agg)
    classified_df, thresholds, cutoff = classify_equal_importance(merged_all)
    save_dataset(classified_df, output_path)

    # === Optional: print thresholds and cutoff ===
    print("\nMedian thresholds per metric:\n", thresholds)
    print(f"\nCutoff for high utility classification: {cutoff}")
