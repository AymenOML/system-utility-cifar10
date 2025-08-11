# data_interpretation/normalize_confusion_csvs.py
import os
import re
import sys
import numpy as np
import pandas as pd

CONFUSION_DIR = "Data/confusion_csv"  # base folder containing client subfolders
OUT_DIR = os.path.join(CONFUSION_DIR, "normalized")
TIDY_CSV_PATH = os.path.join(OUT_DIR, "confusion_matrices_tidy.csv")


def minmax_normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Min–max normalize numeric cells of a DataFrame to [0, 1], preserving non-numeric."""
    df_num = df.apply(pd.to_numeric, errors="coerce")
    vmin = np.nanmin(df_num.values)
    vmax = np.nanmax(df_num.values)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmax, vmin):
        # constant / all-NaN -> zeros (keep NaNs as NaN)
        scaled = df_num.where(df_num.isna(), 0.0)
    else:
        scaled = (df_num - vmin) / (vmax - vmin)

    # return numeric-only scaled frame (same shape), no headers or index changes here
    return scaled


def load_csv_matrix(path: str) -> pd.DataFrame:
    """
    Load a confusion matrix saved as plain CSV (no headers).
    Returns a DataFrame with integer row/column labels 0..n-1.
    """
    df = pd.read_csv(path, header=None)
    n_cols = df.shape[1]
    df.columns = list(range(n_cols))
    df.index = list(range(df.shape[0]))
    return df


def tidy_matrix(df: pd.DataFrame, client_id: int, round_id: int) -> pd.DataFrame:
    """Convert a confusion matrix into long-form rows."""
    df = df.copy()
    df.index.name = "true_label"
    df.columns.name = "pred_label"
    long_df = df.reset_index().melt(
        id_vars=["true_label"],
        var_name="pred_label",
        value_name="value",
    )
    long_df.insert(0, "client_id", client_id)
    long_df.insert(1, "round", round_id)
    return long_df


def extract_ids(client_dir_name: str, filename: str):
    """
    Extract client_id from folder name (supports 'client_7' or '7')
    and round_id from filename 'round_<n>.csv' (or 'client_7_round_<n>.csv').
    """
    m_client = re.search(r"(\d+)$", client_dir_name)
    client_id = int(m_client.group(1)) if m_client else None

    m_round = re.search(r"round[_-]?(\d+)", filename)
    round_id = int(m_round.group(1)) if m_round else None
    return client_id, round_id


def main():
    if not os.path.isdir(CONFUSION_DIR):
        print(f"Not found: {CONFUSION_DIR}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_long = []

    # Walk client subfolders
    for entry in sorted(os.listdir(CONFUSION_DIR)):
        client_path = os.path.join(CONFUSION_DIR, entry)
        if not os.path.isdir(client_path):
            continue  # skip files at root

        # Identify client id
        for fname in sorted(os.listdir(client_path)):
            if not fname.lower().endswith(".csv"):
                continue

            client_id, round_id = extract_ids(entry, fname)
            if client_id is None or round_id is None:
                print(f"[WARN] Skipping {os.path.join(client_path, fname)} (cannot parse client/round)", file=sys.stderr)
                continue

            in_csv = os.path.join(client_path, fname)

            try:
                df = load_csv_matrix(in_csv)
                df_norm = minmax_normalize_frame(df)

                # write normalized csv mirroring structure
                out_client_dir = os.path.join(OUT_DIR, str(client_id))
                os.makedirs(out_client_dir, exist_ok=True)
                out_csv = os.path.join(out_client_dir, f"round_{round_id}.csv")
                # save as plain matrix (no headers/index)
                df_norm.to_csv(out_csv, header=False, index=False)

                # accumulate tidy rows (from normalized values)
                all_long.append(tidy_matrix(df_norm, client_id, round_id))
            except Exception as e:
                print(f"[WARN] Skipped {in_csv}: {e}", file=sys.stderr)
                continue

    if all_long:
        tidy = pd.concat(all_long, ignore_index=True)
        tidy.to_csv(TIDY_CSV_PATH, index=False)
        print(f"Wrote tidy CSV: {TIDY_CSV_PATH}")
        print(f"Normalized matrices under: {OUT_DIR}")
    else:
        print("No matrices processed.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
