# federated_mpi/round_selector.py
import os, json, pickle
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

# Optional: online scaler if you don't have a saved one
class OnlineStandardScaler:
    def __init__(self, n_features, eps=1e-8):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)
        self.eps = eps
    def partial_fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        for x in X:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        var = self.M2 / max(self.n - 1, 1)
        std = np.sqrt(np.maximum(var, self.eps))
        return (X - self.mean) / std

ID_COLS = ["client_rank", "round"]
LABEL_COL = "class_label"
DROP_COLS = ID_COLS + [LABEL_COL, "classification_score"]

def _locate(path_candidates):
    for p in path_candidates:
        if p and os.path.isfile(p):
            return p
    return None

class RoundSelector:
    """
    Per-round: build features -> normalize -> label (heuristic) -> predict with MLP -> pick next clients.
    Saves round artifacts under Data/logs/round_XXX/.
    """
    def __init__(self, model_dir="Model", top_k=10):
        self.model_dir = model_dir
        self.model_path = _locate([
            os.path.join(model_dir, "mlp_client_selector.h5"),
            os.path.join("Models", "mlp_client_selector.h5"),
        ])
        if not self.model_path:
            raise FileNotFoundError("Could not find mlp_client_selector.h5 in Model/ or Models/")

        self.model = load_model(self.model_path)

        # Try to load artifacts from training (preferred)
        self.features_path = _locate([
            os.path.join(model_dir, "features.json"),
            os.path.join("Models", "features.json"),
        ])
        self.scaler_path = _locate([
            os.path.join(model_dir, "scaler.pkl"),
            os.path.join("Models", "scaler.pkl"),
        ])

        self.feature_order = None
        self.scaler = None
        if self.features_path:
            with open(self.features_path, "r") as f:
                self.feature_order = json.load(f)
        if self.scaler_path:
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self.top_k = top_k
        self._online_scaler = None  # created on first use if needed

        # Heuristic labelling (optional, for traceability)
        self._dl = None
        try:
            import data_labelling as dl
            self._dl = dl
        except Exception:
            # If not importable via PYTHONPATH, just skip heuristic labelling in-round.
            self._dl = None

    def _ensure_online_scaler(self, n_features):
        if self.scaler is None:
            if self._online_scaler is None:
                self._online_scaler = OnlineStandardScaler(n_features)
        return self._online_scaler

    def build_feature_df(self, client_reports, feature_order=None):
        """
        client_reports: dict {client_id:int -> dict metric_name->value}
        feature_order: list of metric names; if None, infer from first item (excluding ID/LABEL/DROP_COLS).
        Returns: df with ID cols + features (float), and the inferred feature list used.
        """
        cids = sorted(client_reports.keys())
        rows = []
        # Infer features if not provided
        inferred = feature_order
        if inferred is None:
            # Take all numeric-like keys found in first report, exclude IDs/labels
            first = client_reports[cids[0]]
            inferred = [k for k in first.keys() if k not in DROP_COLS]

        for cid in cids:
            rec = client_reports[cid].copy()
            rec["client_rank"] = cid
            rows.append(rec)

        df = pd.DataFrame(rows)
        # Ensure required columns exist
        for col in ["round"] + inferred:
            if col not in df.columns:
                df[col] = 0.0

        # Coerce features to float, keep ID cols
        for col in inferred:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        df["client_rank"] = df["client_rank"].astype(int)
        if "round" in df.columns:
            df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(0).astype(int)
        else:
            df["round"] = 0

        return df[ID_COLS + inferred], inferred

    def normalize(self, df_features_only):
        X = df_features_only.values.astype(float)
        n_features = X.shape[1]
        if self.scaler is not None:
            Xn = self.scaler.transform(X)
        else:
            scaler = self._ensure_online_scaler(n_features)
            scaler.partial_fit(X)
            Xn = scaler.transform(X)
        return Xn

    def label_heuristic(self, df_full):
        """
        Optional: add heuristic labels for auditing.
        Returns: df_labeled (includes class_label), thresholds dict (if available)
        """
        if self._dl is None:
            return df_full.copy(), {}
        try:
            # Reuse your existing method if available
            labeled, thresholds, cutoff, numeric_cols = self._dl.classify_equal_importance(
                df_full.copy(), per_round=True
            )
            return labeled, {"thresholds": thresholds, "cutoff": cutoff, "numeric_cols": numeric_cols}
        except Exception:
            # Fallback: mean of z-scored features >= 0 selects class 1
            feats = [c for c in df_full.columns if c not in DROP_COLS]
            Z = (df_full[feats] - df_full[feats].mean()) / (df_full[feats].std(ddof=0) + 1e-8)
            score = Z.mean(axis=1)
            out = df_full.copy()
            out["class_label"] = (score >= 0.0).astype(int)
            return out, {"fallback": "z-mean>=0"}

    def predict_and_select(self, Xn, client_ids, threshold=None):
        probs = self.model.predict(Xn, verbose=0).reshape(-1)
        if threshold is None:
            idx = np.argsort(-probs)[: self.top_k]
        else:
            idx = np.where(probs >= threshold)[0]
            if idx.size > self.top_k:
                idx = idx[np.argsort(-probs[idx])[: self.top_k]]
        selected = [int(client_ids[i]) for i in idx]
        return selected, probs

    def run_for_round(self, r, client_reports, out_dir):
        """
        Complete per-round pipeline. Returns list of selected client IDs (ints).
        Saves CSVs under out_dir.
        """
        os.makedirs(out_dir, exist_ok=True)

        # 1) Build features DF (IDs + features)
        # Prefer training feature order if available
        df_raw, inferred = self.build_feature_df(
            client_reports,
            feature_order=self.feature_order if self.feature_order else None
        )
        feats_used = self.feature_order if self.feature_order else inferred
        df_id = df_raw[ID_COLS].copy()
        df_feats = df_raw[feats_used].copy()

        # 2) Normalize (using saved scaler or online)
        Xn = self.normalize(df_feats)
        df_norm = pd.DataFrame(Xn, columns=feats_used)
        df_norm = pd.concat([df_id, df_norm], axis=1)

        # 3) Heuristic labels for audit
        df_labeled, lab_meta = self.label_heuristic(pd.concat([df_id, df_feats], axis=1))

        # 4) Predict with the MLP
        selected, probs = self.predict_and_select(Xn, df_id["client_rank"].values)

        # 5) Persist round artifacts
        df_raw.to_csv(os.path.join(out_dir, "features_raw.csv"), index=False)
        df_norm.to_csv(os.path.join(out_dir, "features_normalized.csv"), index=False)
        df_labeled.to_csv(os.path.join(out_dir, "labels_heuristic.csv"), index=False)
        pd.DataFrame({
            "client_rank": df_id["client_rank"].values,
            "p_select": probs
        }).to_csv(os.path.join(out_dir, "selector_probs.csv"), index=False)
        with open(os.path.join(out_dir, "selected_clients.txt"), "w") as f:
            f.write(",".join(map(str, selected)))

        return selected
