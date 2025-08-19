"""
postprocess_pipeline.py

Utilities for:
1) Normalizing stats/system CSVs and confusion matrices (batch mode).
2) Building a labeled training table (equal-importance, per-round).
3) **Per-round selection**: normalize current round, apply cost inversion,
   label, run the trained MLP, select clients for the next round, and journal.

Intended usage in your training loop (server side):
    from postprocess_pipeline import run_round_selection
    pred_csv, selected = run_round_selection(
        round_id=rnd,
        base_dir="Data",
        artifacts_dir="Models",  # change to "Model" if that's your folder name
        threshold=0.5,
        journal_path="Data/selection_journal.txt",
    )

Batch post-processing (optional, produces a single labeled CSV for the whole run):
    run_postprocessing(base_dir="Data", out_csv="Data/clients_labeled_equal_importance_with_confusion.csv")
"""

import os
import glob
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd


# -------------------------
# Paths & constants
# -------------------------

ID_COLS: Set[str] = {"client_rank", "round", "timestamp"}

# Default logs layout
STATS_CSV_REL = os.path.join("logs", "clients_stats.csv")
SYSTEM_CSV_REL = os.path.join("logs", "clients_system.csv")
CONFUSION_DIR_REL = os.path.join("logs", "confusion_csv")

# Feature direction map (MUST MATCH your MLP training script)
# "benefit" = higher is better; "cost" = lower is better
BENEFIT_OR_COST: Dict[str, str] = {
    "data_size": "benefit",
    "data_variance": "benefit",
    "local_loss": "cost",
    "local_accuracy": "benefit",
    "cpu_time": "cost",
    "ram_used_mb": "cost",
    "net_sent_bytes": "cost",
    "net_recv_bytes": "cost",
    "gpu_mem_used_mb": "cost",
    "gpu_load": "cost",
    "confusion_mean": "benefit",
}


# -------------------------
# Small helpers
# -------------------------

def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _minmax01(df: pd.DataFrame, exclude: Set[str]) -> pd.DataFrame:
    """Column-wise min–max normalization to [0,1] for numeric columns not in exclude."""
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if not np.issubdtype(out[c].dtype, np.number):
            # try to coerce
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[c] = pd.to_numeric(out[c], errors="coerce")
        col = out[c]
        if not np.issubdtype(col.dtype, np.number):
            continue
        mn = pd.Series(col).min(skipna=True)
        mx = pd.Series(col).max(skipna=True)
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            out[c] = 0.0
        else:
            out[c] = (col - mn) / (mx - mn)
    return out


def _invert_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply directionality: for 'cost' features, invert.
    Prefer 1-x if values are in [0,1]; otherwise use -x.
    """
    out = df.copy()
    for c in out.columns:
        if c in ID_COLS:
            continue
        role = BENEFIT_OR_COST.get(c, "benefit")
        if role != "cost":
            continue
        col = pd.to_numeric(out[c], errors="coerce")
        mask_valid = col.notna()
        if mask_valid.sum() == 0:
            out[c] = col
            continue
        mn, mx = float(col[mask_valid].min()), float(col[mask_valid].max())
        if mn >= 0.0 and mx <= 1.0:
            out.loc[mask_valid, c] = 1.0 - col[mask_valid]
        else:
            # fallback
            warnings.warn(
                f"[Directionality] Feature '{c}' not in [0,1] (min={mn:.3g}, max={mx:.3g}). "
                f"Falling back to sign flip (-x). Consider min–max normalizing before this step."
            )
            out.loc[mask_valid, c] = -col[mask_valid]
    return out


def _label_equal_importance_per_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a SINGLE round: label = 1 if a row is above the round median on at least half of features.
    Uses strict '>' (not '>='), same as your current heuristic.
    """
    feats = [c for c in df.columns if c not in ID_COLS]
    if len(feats) == 0:
        raise ValueError("No feature columns found to label.")
    meds = df[feats].median(axis=0, skipna=True)
    above = (df[feats].gt(meds, axis=1)).sum(axis=1)
    cutoff = max(1, len(feats) // 2)
    out = df.copy()
    out["classification_score"] = above
    out["class_label"] = (above >= cutoff).astype(int)
    return out


def _read_confusion_mean_for_round(round_id: int, root: str) -> pd.DataFrame:
    """
    Reads raw confusion matrices and returns a dataframe:
        client_rank, round, confusion_mean
    Each matrix is min–max normalized (matrix-wise) before taking the mean.
    """
    rows = []
    pattern = os.path.join(root, "client_*")
    for client_dir in sorted(glob.glob(pattern)):
        try:
            client_id = int(os.path.basename(client_dir).split("_")[-1])
        except Exception:
            continue
        path = os.path.join(client_dir, f"round_{round_id}.csv")
        if not os.path.exists(path):
            continue
        try:
            mat = np.loadtxt(path, delimiter=",")
            if mat.size == 0:
                v = 0.0
            else:
                mn, mx = float(mat.min()), float(mat.max())
                v = float(((mat - mn) / (mx - mn)).mean()) if mx > mn else 0.0
        except Exception:
            v = 0.0
        rows.append({"client_rank": client_id, "round": round_id, "confusion_mean": v})
    return pd.DataFrame(rows)


# -------------------------
# Batch Mode (optional)
# -------------------------

def _normalize_stats_and_system(base_dir: str) -> Tuple[str, str]:
    """Normalize the full stats/system CSVs (all rounds) – batch helper."""
    stats_csv = os.path.join(base_dir, STATS_CSV_REL)
    system_csv = os.path.join(base_dir, SYSTEM_CSV_REL)

    if not os.path.exists(stats_csv):
        raise FileNotFoundError(f"Missing stats CSV: {stats_csv}")
    if not os.path.exists(system_csv):
        raise FileNotFoundError(f"Missing system CSV: {system_csv}")

    df_stats = pd.read_csv(stats_csv)
    df_sys = pd.read_csv(system_csv)

    df_stats_n = _minmax01(df_stats, exclude=ID_COLS)
    df_sys_n = _minmax01(df_sys, exclude=ID_COLS)

    out_stats = os.path.join(base_dir, "logs", "clients_stats_normalized.csv")
    out_sys = os.path.join(base_dir, "logs", "clients_system_normalized.csv")
    _ensure_dir(os.path.dirname(out_stats))
    df_stats_n.to_csv(out_stats, index=False)
    df_sys_n.to_csv(out_sys, index=False)
    return out_stats, out_sys


def _build_labeled_table_all_rounds(base_dir: str) -> pd.DataFrame:
    """
    Build a single labeled table across all rounds (batch mode).
    Per your current heuristic, labeling is **per-round**.
    """
    stats_csv = os.path.join(base_dir, STATS_CSV_REL)
    system_csv = os.path.join(base_dir, SYSTEM_CSV_REL)
    conf_root = os.path.join(base_dir, CONFUSION_DIR_REL)

    if not os.path.exists(stats_csv) or not os.path.exists(system_csv):
        raise FileNotFoundError("Missing stats or system CSV.")

    df_stats = pd.read_csv(stats_csv)
    df_sys = pd.read_csv(system_csv)

    # Normalize globally before we split per round
    df_stats_n = _minmax01(df_stats, exclude=ID_COLS)
    df_sys_n = _minmax01(df_sys, exclude=ID_COLS)

    # Merge (outer to keep all seen clients/rounds)
    df_all = df_stats_n.merge(df_sys_n, on=["client_rank", "round"], how="outer")

    # Add confusion_mean per round
    all_rounds = sorted(pd.unique(df_all["round"].dropna().astype(int)))
    conf_rows = []
    for r in all_rounds:
        conf_rows.append(_read_confusion_mean_for_round(r, root=conf_root))
    df_conf = pd.concat(conf_rows, ignore_index=True) if len(conf_rows) else pd.DataFrame(
        columns=["client_rank", "round", "confusion_mean"]
    )
    df_all = df_all.merge(df_conf, on=["client_rank", "round"], how="left").fillna(0.0)

    # Apply directionality
    df_all = _invert_costs(df_all)

    # Label **per round**
    labeled_parts = []
    for r in all_rounds:
        chunk = df_all[df_all["round"] == r].copy()
        if len(chunk) == 0:
            continue
        labeled_parts.append(_label_equal_importance_per_round(chunk))
    return pd.concat(labeled_parts, ignore_index=True) if labeled_parts else pd.DataFrame()


def run_postprocessing(base_dir: str = "Data", out_csv: str = None) -> str:
    """
    Batch mode: produces a **single** labeled CSV across all rounds.
    Returns the output CSV path.
    """
    df_labeled = _build_labeled_table_all_rounds(base_dir=base_dir)
    if df_labeled.empty:
        raise RuntimeError("No labeled data could be produced.")

    if out_csv is None:
        out_csv = os.path.join(base_dir, "clients_labeled_equal_importance_with_confusion.csv")
    _ensure_dir(os.path.dirname(out_csv))
    df_labeled.to_csv(out_csv, index=False)
    return out_csv


# -------------------------
# Per-round Selection
# -------------------------

def _mlp_predict(df_labeled: pd.DataFrame, artifacts_dir: str, threshold: float) -> pd.DataFrame:
    """Load artifacts and add p_select, mlp_label."""
    # Lazy import TF to avoid heavy import when only batch helpers are used
    import pickle
    from tensorflow.keras.models import load_model  # type: ignore

    # Load artifacts
    with open(os.path.join(artifacts_dir, "features.json"), "r") as f:
        features = json.load(f)
    with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    model = load_model(os.path.join(artifacts_dir, "mlp_client_selector.h5"))

    # Ensure all features exist
    missing = [c for c in features if c not in df_labeled.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns for MLP: {missing}")

    X = df_labeled[features].astype("float32").values
    Xs = scaler.transform(X)

    # Predict
    p = model.predict(Xs, verbose=0).ravel()
    out = df_labeled.copy()
    out["p_select"] = p
    out["mlp_label"] = (p >= float(threshold)).astype(int)
    return out


def _load_round_slice(base_dir: str, round_id: int) -> pd.DataFrame:
    """
    Load, normalize (per-round), merge stats+system+confusion for a given round,
    and apply directionality.
    """
    stats_csv = os.path.join(base_dir, STATS_CSV_REL)
    system_csv = os.path.join(base_dir, SYSTEM_CSV_REL)
    conf_root = os.path.join(base_dir, CONFUSION_DIR_REL)

    if not os.path.exists(stats_csv):
        raise FileNotFoundError(f"Missing stats CSV: {stats_csv}")
    if not os.path.exists(system_csv):
        raise FileNotFoundError(f"Missing system CSV: {system_csv}")

    df_stats = pd.read_csv(stats_csv)
    df_sys = pd.read_csv(system_csv)

    # Filter to current round
    df_stats = df_stats[df_stats["round"] == round_id].copy()
    df_sys = df_sys[df_sys["round"] == round_id].copy()

    if df_stats.empty and df_sys.empty:
        raise RuntimeError(f"No stats/system rows found for round {round_id}.")

    # Per-round min–max
    df_stats_n = _minmax01(df_stats, exclude=ID_COLS) if not df_stats.empty else df_stats
    df_sys_n = _minmax01(df_sys, exclude=ID_COLS) if not df_sys.empty else df_sys

    # Merge
    df = df_stats_n.merge(df_sys_n, on=["client_rank", "round"], how="outer")

    # Confusion mean
    df_conf = _read_confusion_mean_for_round(round_id, root=conf_root)
    if df_conf.empty:
        # No confusion written? Still proceed with 0.0
        df["confusion_mean"] = 0.0
    else:
        df = df.merge(df_conf, on=["client_rank", "round"], how="left")
        df["confusion_mean"] = df["confusion_mean"].fillna(0.0)

    # Apply directionality
    df = _invert_costs(df)

    # Clean client_rank
    if "client_rank" in df.columns:
        df["client_rank"] = pd.to_numeric(df["client_rank"], errors="coerce").astype("Int64")

    return df


def run_round_selection(
    round_id: int,
    base_dir: str = "Data",
    artifacts_dir: str = "Models",
    threshold: float = 0.5,
    journal_path: str = "Data/selection_journal.txt",
) -> Tuple[str, List[int]]:
    """
    Per-round pipeline:
      1) Read round slice of stats/system/confusion.
      2) Per-round min–max normalization.
      3) Apply cost inversion (directionality).
      4) Label via equal-importance per-round.
      5) Predict with MLP -> p_select, mlp_label (threshold).
      6) Select: clients with mlp_label==1; if none, top-3 by p_select.
      7) Persist features + predictions under Data/round_<id>/ and append journal.

    Returns:
      (pred_csv_path, selected_client_ids)
    """
    # 1–3) Load & normalize & invert
    df_round = _load_round_slice(base_dir=base_dir, round_id=round_id)

    if df_round.empty:
        raise RuntimeError(f"No data available to select on round {round_id}.")

    # 4) Label (equal-importance, per-round)
    df_lab = _label_equal_importance_per_round(df_round)

    # 5) Predict with MLP
    df_pred = _mlp_predict(df_lab, artifacts_dir=artifacts_dir, threshold=threshold)

    # 6) Select
    selected = df_pred.loc[df_pred["mlp_label"] == 1, "client_rank"].dropna().astype(int).tolist()
    if len(selected) == 0:
        # edge case: choose top-3 by p_select
        topk = (
            df_pred.sort_values("p_select", ascending=False)
            .head(3)["client_rank"]
            .dropna()
            .astype(int)
            .tolist()
        )
        selected = topk

    # 7) Persist artifacts + journal
    out_dir = os.path.join(base_dir, f"round_{round_id}")
    _ensure_dir(out_dir)

    features_csv = os.path.join(out_dir, "clients_labeled_round.csv")
    df_lab.to_csv(features_csv, index=False)

    pred_csv = os.path.join(out_dir, "clients_with_mlp_round.csv")
    df_pred.to_csv(pred_csv, index=False)

    _ensure_dir(os.path.dirname(journal_path))
    with open(journal_path, "a") as jf:
        jf.write(f"Round {round_id} — threshold={threshold}\n")
        jf.write("p_select by client:\n")
        for _, row in df_pred.sort_values("client_rank").iterrows():
            # guard for NaN client ids
            cid = int(row["client_rank"]) if pd.notna(row["client_rank"]) else -1
            jf.write(f"  c{cid}: {row['p_select']:.4f} (mlp={int(row['mlp_label'])})\n")
        jf.write(f"Selected clients for next round: {selected}\n\n")

    return pred_csv, selected


# -------------------------
# CLI (optional)
# -------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Post-processing & per-round selection")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("all", help="Batch post-processing for the entire run")
    p_all.add_argument("--base-dir", default="Data")
    p_all.add_argument("--out", default=None)

    p_rs = sub.add_parser("round-select", help="Run per-round selection (normalize+label+MLP)")
    p_rs.add_argument("--round", type=int, required=True)
    p_rs.add_argument("--base-dir", default="Data")
    p_rs.add_argument("--artifacts", default="Models")
    p_rs.add_argument("--threshold", type=float, default=0.5)
    p_rs.add_argument("--journal", default="Data/selection_journal.txt")

    args = parser.parse_args()

    if args.cmd == "all":
        outp = run_postprocessing(base_dir=args.base_dir, out_csv=args.out)
        print(f"[Saved] Labeled dataset: {outp}")
    elif args.cmd == "round-select":
        pred_csv, selected = run_round_selection(
            round_id=args.round,
            base_dir=args.base_dir,
            artifacts_dir=args.artifacts,
            threshold=args.threshold,
            journal_path=args.journal,
        )
        print(f"[Saved] Predictions: {pred_csv}")
        print(f"[Selected] {selected}")
