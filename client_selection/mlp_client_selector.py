"""
Shallow MLP for client selection (binary):
- Trains on labeled CSV (expects column `class_label`)
- Splits by round (no leakage)
- Scales features
- Handles class imbalance with class weights
- Saves artifacts (model, scaler, feature list)
- Predicts on new CSVs that have the same feature columns

Usage:
  Train:
    python mlp_client_selector.py train --data Data/clients_labeled_equal_importance_with_confusion.csv --artifacts Models --epochs 200 --patience 10

  Predict:
    python mlp_client_selector.py predict --data Data/new_client_metrics.csv --artifacts Models --out Data/new_client_predictions.csv --threshold 0.5

  Make template of required feature columns (empty CSV):
    python mlp_client_selector.py make-template --train-data Data/clients_labeled_equal_importance_with_confusion.csv --out Data/new_client_template.csv
"""

import os
import json
import argparse
import pickle
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# Columns
# ----------------------------
ID_COLS = ["client_rank", "round"]
LABEL_COL = "class_label"
# We exclude IDs, the label, and the derived "classification_score" from features
DROP_COLS = ID_COLS + [LABEL_COL, "classification_score"]

# ----------------------------
# Data utilities
# ----------------------------
def load_labeled_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected target column '{LABEL_COL}' in dataset; got columns: {df.columns.tolist()}")
    return df

def get_feature_columns(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in DROP_COLS]
    if not features:
        raise ValueError("No numeric features remain after excluding IDs/label/classification_score.")
    return features

def split_by_round(df: pd.DataFrame, test_size=0.2, val_size=0.2):
    """Group-aware splits so rounds don't leak across splits."""
    rounds = df["round"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    idx_trainval, idx_test = next(gss.split(df, groups=rounds))
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    rounds_tv = df_trainval["round"].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
    idx_train, idx_val = next(gss2.split(df_trainval, groups=rounds_tv))
    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val = df_trainval.iloc[idx_val].reset_index(drop=True)
    return df_train, df_val, df_test

def prepare_arrays(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].astype(float).values
    y = df[LABEL_COL].astype(int).values
    meta = df[ID_COLS].copy() if all(c in df.columns for c in ID_COLS) else None
    return X, y, meta

# ----------------------------
# Model
# ----------------------------
def build_shallow_mlp(input_dim: int) -> keras.Model:
    """Shallow = very few layers; here: 1 hidden layer + output."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),  # probability of "select"
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="AUC"),
                 keras.metrics.Precision(name="Precision"),
                 keras.metrics.Recall(name="Recall")]
    )
    return model

def compute_class_weights(y: np.ndarray):
    """Balanced class weights = N / (2 * count_c)."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    n = len(y)
    if pos == 0 or neg == 0:
        # Degenerate; fallback to equal weights so training still runs
        return {0: 1.0, 1: 1.0}
    return {0: n / (2.0 * neg), 1: n / (2.0 * pos)}

# ----------------------------
# Train / Evaluate / Save
# ----------------------------
def train(csv_path: str, artifacts_dir: str, epochs=200, batch_size=256, patience=10):
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path  = os.path.join(artifacts_dir, "mlp_client_selector.h5")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    feats_path  = os.path.join(artifacts_dir, "features.json")

    # Load & features
    df = load_labeled_dataset(csv_path)
    feature_cols = get_feature_columns(df)

    # Group-aware split by round
    df_train, df_val, df_test = split_by_round(df, test_size=0.2, val_size=0.2)
    X_tr, y_tr, _ = prepare_arrays(df_train, feature_cols)
    X_va, y_va, _ = prepare_arrays(df_val, feature_cols)
    X_te, y_te, meta_te = prepare_arrays(df_test, feature_cols)

    # Scale numeric features
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    # Model
    model = build_shallow_mlp(input_dim=X_tr.shape[1])

    # Class weights for imbalance
    class_weight = compute_class_weights(y_tr)

    # Train
    es = keras.callbacks.EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        monitor="val_auc",
        mode="max"
    )
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[es],
        verbose=1
    )

    # Evaluate on held-out test rounds
    p_te = model.predict(X_te, batch_size=batch_size).ravel()
    auc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")
    y_hat = (p_te >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_hat, average="binary", zero_division=0)
    cm = confusion_matrix(y_te, y_hat)

    # Save artifacts
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(feats_path, "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)

    # Report
    print("\n=== Evaluation on held-out test rounds ===")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)

    print("\nArtifacts saved:")
    print(f"  Model:  {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Feats:  {feats_path}")

def predict(csv_path_new: str, artifacts_dir: str, out_csv: str = None, threshold: float = 0.5):
    model_path  = os.path.join(artifacts_dir, "mlp_client_selector.h5")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    feats_path  = os.path.join(artifacts_dir, "features.json")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feats_path)):
        raise FileNotFoundError("Model/scaler/features not found. Train first.")

    model = keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(feats_path, "r") as f:
        feature_cols = json.load(f)["features"]

    df = pd.read_csv(csv_path_new)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[feature_cols].astype(float).values
    X = scaler.transform(X)
    p = model.predict(X, batch_size=256).ravel()
    y_hat = (p >= threshold).astype(int)

    out_df = df.copy()
    out_df["p_select"] = p
    out_df["mlp_label"] = y_hat

    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print(f"Wrote predictions to: {out_csv}")
    else:
        print(out_df.head())
    return out_df

def make_template(train_csv: str, out_csv: str):
    """Create an empty template CSV containing exactly the feature columns needed for prediction."""
    df = load_labeled_dataset(train_csv)
    feature_cols = get_feature_columns(df)
    tmpl = pd.DataFrame(columns=feature_cols)
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    tmpl.to_csv(out_csv, index=False)
    print(f"Template written to: {out_csv}")
    print("Columns:", feature_cols)

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Shallow MLP for client selection (binary).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train and save model/scaler/features.")
    p_train.add_argument("--data", type=str, required=True, help="Path to labeled training CSV.")
    p_train.add_argument("--artifacts", type=str, default="Models", help="Directory to save model artifacts.")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--patience", type=int, default=10)

    p_pred = sub.add_parser("predict", help="Predict on new csv with same feature columns.")
    p_pred.add_argument("--data", type=str, required=True, help="CSV to label.")
    p_pred.add_argument("--artifacts", type=str, default="Models", help="Directory containing model artifacts.")
    p_pred.add_argument("--out", type=str, default=None, help="Optional path to write predictions csv.")
    p_pred.add_argument("--threshold", type=float, default=0.5)

    p_tmpl = sub.add_parser("make-template", help="Build a feature-only template CSV from a training CSV.")
    p_tmpl.add_argument("--train-data", type=str, required=True, help="Path to labeled training CSV.")
    p_tmpl.add_argument("--out", type=str, required=True, help="Where to write the empty template CSV.")

    args = parser.parse_args()

    if args.cmd == "train":
        train(csv_path=args.data, artifacts_dir=args.artifacts, epochs=args.epochs,
              batch_size=args.batch_size, patience=args.patience)
    elif args.cmd == "predict":
        predict(csv_path_new=args.data, artifacts_dir=args.artifacts, out_csv=args.out, threshold=args.threshold)
    elif args.cmd == "make-template":
        make_template(train_csv=args.train_data, out_csv=args.out)

if __name__ == "__main__":
    main()
