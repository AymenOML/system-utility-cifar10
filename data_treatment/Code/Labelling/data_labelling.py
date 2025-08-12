import os
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


def _numeric_feature_cols(df):
    """Numeric feature columns excluding identifiers."""
    id_cols = {"client_rank", "round"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_cols]


def classify_equal_importance(df, per_round=False):
    """
    Apply equal-importance threshold classification.
    If per_round=True, thresholds are computed separately within each round.
    Returns: df_with_scores, thresholds (Series or dict[round->Series]), cutoff, numeric_cols
    """
    df = df.copy()
    numeric_cols = _numeric_feature_cols(df)
    cutoff = len(numeric_cols) / 2

    if not per_round:
        thresholds = df[numeric_cols].median()
        binary_scores = (df[numeric_cols] > thresholds).astype(int)
        df["classification_score"] = binary_scores.sum(axis=1)
        df["class_label"] = (df["classification_score"] >= cutoff).astype(int)
        return df, thresholds, cutoff, numeric_cols

    # Per-round thresholds and scores
    thresholds_by_round = {}
    scores = np.zeros(len(df), dtype=int)

    for rnd, idx in df.groupby("round").groups.items():
        sub = df.loc[idx, numeric_cols]
        thr = sub.median()
        thresholds_by_round[rnd] = thr
        bin_sub = (sub > thr).astype(int)
        scores[idx] = bin_sub.sum(axis=1).values

    df["classification_score"] = scores
    df["class_label"] = (df["classification_score"] >= cutoff).astype(int)
    return df, thresholds_by_round, cutoff, numeric_cols


def write_selection_journal(df, numeric_cols, txt_path):
    """
    For each round, write:
      - per-round medians for all numeric features
      - list of selected client_ranks (class_label=1)
    """
    lines = []
    rounds = sorted(df["round"].unique())
    for rnd in rounds:
        sub = df[df["round"] == rnd]
        med = sub[numeric_cols].median().sort_index()
        selected = sorted(sub.loc[sub["class_label"] == 1, "client_rank"].tolist())
        total = len(sub)
        lines.append(f"=== Round {rnd} ===")
        lines.append("Per-round medians:")
        for k, v in med.items():
            lines.append(f"  - {k}: {v:.6f}")
        lines.append(f"Selected clients ({len(selected)}/{total}): {selected}")
        lines.append("")  # blank line

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Journal written to: {txt_path}")


def save_dataset(df, output_path):
    """Save the final DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"File saved as: {output_path}")


if __name__ == "__main__":
    # === Paths to your CSV files ===
    stats_path = "Code/Labelling/Data/all_clients_stats_normalized.csv"
    sys_path = "Code/Labelling/Data/client_system_metrics_normalized.csv"
    conf_path = "Code/Labelling/Data/confusion_matrices_tidy.csv"
    output_path = "Code/Labelling/Data/clients_labeled_equal_importance_with_confusion.csv"
    journal_path = "Code/Labelling/Data/selection_journal.txt"

    # === Workflow ===
    all_clients_stats, system_metrics, confusion_tidy = load_datasets(stats_path, sys_path, conf_path)
    confusion_agg = aggregate_confusion(confusion_tidy)
    merged_all = merge_datasets(all_clients_stats, system_metrics, confusion_agg)

    # Choose thresholding mode:
    #   per_round=True  -> thresholds computed within each round (recommended for your journal)
    #   per_round=False -> single global thresholds over all rounds
    classified_df, thresholds, cutoff, numeric_cols = classify_equal_importance(merged_all, per_round=True)

    save_dataset(classified_df, output_path)
    write_selection_journal(classified_df, numeric_cols, journal_path)

    # Optional: print quick summary
    print(f"\nCutoff used: {cutoff} (half of {len(numeric_cols)} features)")
