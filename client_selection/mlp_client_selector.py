"""
Shallow MLP for client selection (binary) — with feature directionality (cost inversion).

- Trains on a normalized + labeled CSV (expects `class_label` and a `round` column)
- Uses per-round group split to avoid leakage
- Applies directionality: benefit features kept as-is; cost features inverted (1 - x)
- Scales features with StandardScaler
- Handles class imbalance via class weights
- Saves artifacts (model, scaler, feature list)
- Predicts on new CSVs with the same features; adds p_select and mlp_label

Usage
-----
Train:
  python mlp_client_selector.py train \
    --data Data/clients_labeled_equal_importance_with_confusion.csv \
    --artifacts Models \
    --epochs 200 --patience 10

Predict:
  python mlp_client_selector.py predict \
    --data Data/new_clients_heuristic_labelling.csv \
    --artifacts Models \
    --out Data/new_client_quantile_grid_predictions.csv \
    --threshold 0.5

Make a template (empty rows, correct headers):
  python mlp_client_selector.py make-template \
    --train-data Data/clients_labeled_equal_importance_with_confusion.csv \
    --out Data/new_client_template.csv
"""

import os
import json
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# Sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Keras / TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ----------------------------
# Configuration
# ----------------------------

# Default features (order matters). Will be saved to features.json at train time.
DEFAULT_FEATURES: List[str] = [
    "data_size",
    "data_variance",
    "local_loss",
    "local_accuracy",
    "cpu_time",
    "ram_used_mb",
    "net_sent_bytes",
    "net_recv_bytes",
    "gpu_mem_used_mb",
    "gpu_load",
    "confusion_mean",
]

# Direction map: "benefit" (higher is better) or "cost" (lower is better).
# Adjust as needed.
DIRECTION_MAP: Dict[str, str] = {
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

ID_COLS = {"client_rank", "round", "timestamp"}
TARGET_COL = "class_label"

# ----------------------------
# Utilities
# ----------------------------

_warned_out_of_unit = set()

def _invert_cost_feature(series: pd.Series, name: str) -> pd.Series:
    """Invert a cost feature.
    Preferred: 1 - x (assuming input in [0,1]).
    Fallback:  -x if values are not within [0,1].
    """
    s = series.astype("float32")

    if s.dropna().empty:
        return s

    mn = float(np.nanmin(s))
    mx = float(np.nanmax(s))

    if mn >= 0.0 and mx <= 1.0:
        return 1.0 - s

    # Fallback with warning (once per feature)
    if name not in _warned_out_of_unit:
        warnings.warn(
            f"[Directionality] Feature '{name}' not in [0,1] (min={mn:.3g}, max={mx:.3g}). "
            "Falling back to sign flip (-x). Consider min–max normalizing before this step."
        )
        _warned_out_of_unit.add(name)
    return -s


def apply_directionality(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Apply benefit/cost transformation in-place for the given features."""
    df = df.copy()
    for feat in feature_names:
        if feat not in df.columns:
            continue
        role = DIRECTION_MAP.get(feat, "benefit")
        if role == "cost":
            df[feat] = _invert_cost_feature(df[feat], feat)
    return df


def select_features(df: pd.DataFrame, desired: List[str]) -> List[str]:
    """Keep only features available in df, preserving desired order."""
    available = [c for c in desired if c in df.columns]
    missing = [c for c in desired if c not in df.columns]
    if missing:
        warnings.warn(f"[Features] Missing columns: {missing}")
    if not available:
        raise ValueError("No desired features are present in the data.")
    return available


def build_mlp(input_dim: int) -> tf.keras.Model:
    """A small, robust MLP."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def train(csv_path: str, artifacts_dir: str, epochs: int, batch_size: int, patience: int):
    # Load data
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column `{TARGET_COL}` in {csv_path}")

    if "round" not in df.columns:
        raise ValueError("Missing `round` column (needed for group split).")

    # Drop completely empty rows (common in templates)
    df = df.dropna(how="all")

    # Feature selection (start from defaults, keep what exists)
    features = select_features(df, DEFAULT_FEATURES)

    # Apply directionality on a copy
    df_dir = apply_directionality(df, features)

    # Prepare X, y, groups
    X = df_dir[features].astype("float32").values
    y = df[TARGET_COL].astype("int32").values
    groups = df["round"].values

    # Train/valid/test split by round to avoid leakage
    # First, split train+val vs test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (trainval_idx, test_idx) = next(gss.split(X, y, groups=groups))

    X_trainval, y_trainval, groups_trainval = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Split train vs val (still group-aware)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    (train_idx, val_idx) = next(gss2.split(X_trainval, y_trainval, groups=groups_trainval))

    X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
    X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Class weights
    classes = np.unique(y_train)
    cw_values = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw_values)}

    # Model
    model = build_mlp(input_dim=X_train_s.shape[1])

    es = callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=max(2, patience // 2), factor=0.5)

    model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[es, rlrop],
        verbose=2,
    )

    # Evaluation
    y_test_pred = model.predict(X_test_s).ravel()
    y_test_hat = (y_test_pred >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, y_test_pred)
    except Exception:
        auc = float("nan")

    print("\n=== Test Metrics ===")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_test_hat, digits=4))
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, y_test_hat))

    # Save artifacts
    ensure_dir(artifacts_dir)

    # Save features list (order matters)
    with open(os.path.join(artifacts_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=2)

    # Save scaler
    with open(os.path.join(artifacts_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save model
    model.save(os.path.join(artifacts_dir, "mlp_client_selector.h5"))
    print(f"\n[Saved] Artifacts to '{artifacts_dir}'")


def _load_artifacts(artifacts_dir: str) -> Tuple[List[str], StandardScaler, tf.keras.Model]:
    with open(os.path.join(artifacts_dir, "features.json"), "r") as f:
        features = json.load(f)
    with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    model = tf.keras.models.load_model(os.path.join(artifacts_dir, "mlp_client_selector.h5"))
    return features, scaler, model


def predict(csv_path_new: str, artifacts_dir: str, out_csv: str, threshold: float = 0.5):
    # Load new data
    df_new = pd.read_csv(csv_path_new)
    df_new = df_new.dropna(how="all").copy()

    # Load artifacts
    features, scaler, model = _load_artifacts(artifacts_dir)

    # Apply directionality to the feature slice only
    # (we don’t touch original columns that are not features)
    feature_slice = df_new[features].copy()
    feature_slice = apply_directionality(feature_slice, features)

    # Drop rows with any missing features
    mask_complete = ~feature_slice.isnull().any(axis=1)
    if not mask_complete.all():
        n_drop = int((~mask_complete).sum())
        warnings.warn(f"[Predict] Dropping {n_drop} rows with missing feature values.")
    feature_slice = feature_slice[mask_complete]
    df_out = df_new.loc[mask_complete].copy()

    # Scale
    X_s = scaler.transform(feature_slice.astype("float32").values)

    # Predict
    p_select = model.predict(X_s).ravel()
    mlp_label = (p_select >= float(threshold)).astype(int)

    # Attach
    df_out["p_select"] = p_select
    df_out["mlp_label"] = mlp_label

    # Write
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[Saved] Predictions to '{out_csv}' with threshold={threshold}")


def make_template(train_csv: str, out_csv: str):
    df = pd.read_csv(train_csv)
    df = df.dropna(how="all")

    # Use defaults but keep only those present
    feats = select_features(df, DEFAULT_FEATURES)

    # Build an empty frame with identifiers + features (in order)
    cols = ["client_rank", "round"] + feats
    tmpl = pd.DataFrame(columns=cols)
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    tmpl.to_csv(out_csv, index=False)
    print(f"[Saved] Empty template to '{out_csv}'")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="MLP Client Selector (with cost inversion)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    p_train = sub.add_parser("train", help="Train the MLP selector")
    p_train.add_argument("--data", type=str, required=True, help="Path to labeled CSV.")
    p_train.add_argument("--artifacts", type=str, default="Models", help="Dir to save artifacts.")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--patience", type=int, default=10)

    # Predict
    p_pred = sub.add_parser("predict", help="Predict with the trained selector")
    p_pred.add_argument("--data", type=str, required=True, help="Path to normalized+labelled CSV.")
    p_pred.add_argument("--artifacts", type=str, default="Models", help="Dir containing saved artifacts.")
    p_pred.add_argument("--out", type=str, required=True, help="Where to write predictions CSV.")
    p_pred.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5).")

    # Template
    p_tmpl = sub.add_parser("make-template", help="Make an empty CSV with the expected feature columns")
    p_tmpl.add_argument("--train-data", type=str, required=True, help="A labeled CSV to infer feature columns.")
    p_tmpl.add_argument("--out", type=str, required=True, help="Where to write the empty template CSV.")

    args = parser.parse_args()

    if args.cmd == "train":
        train(csv_path=args.data, artifacts_dir=args.artifacts,
              epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
    elif args.cmd == "predict":
        predict(csv_path_new=args.data, artifacts_dir=args.artifacts,
                out_csv=args.out, threshold=args.threshold)
    elif args.cmd == "make-template":
        make_template(train_csv=args.train_data, out_csv=args.out)


if __name__ == "__main__":
    main()
