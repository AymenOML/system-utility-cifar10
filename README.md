# Federated CIFAR-10 with TensorFlow, MPI, and Client Selection (MLP)

This project trains CIFAR-10 with both **simulated** and **MPI-based** federated learning, logs per-client/per-round metrics, normalizes and **labelizes** clients with an interpretable rule, and then trains a **shallow MLP** that can label new clients for future rounds.

- **TensorFlow / Keras** for models  
- **MPI (mpi4py)** for distributed client–server FL  
- **Pandas / NumPy** for data treatment  
- **scikit-learn** for scaling & metrics  
- **Matplotlib** (optional) for plots

---

## 🚀 Setup

```bash
python -m venv venv
source venv/bin/activate           # Windows: .\venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt    # must include: tensorflow, mpi4py, pandas, numpy, scikit-learn, psutil, gputil
```

> If you see an error installing `sklearn`, install **`scikit-learn`** (the correct package).

---

## 📁 Project Structure (key parts)

```
client_selection/
├─ Data/
│  ├─ clients_labeled_equal_importance_with_confusion.csv   # output of labelization
│  ├─ new_client_quantile_grid.csv                          # sample inputs for testing
│  └─ ...                                                   # normalized CSVs
├─ Models/
│  ├─ mlp_client_selector.h5                                # trained MLP
│  ├─ scaler.pkl                                            # StandardScaler (train fit)
│  └─ features.json                                         # feature names used by the model
├─ mlp_client_selector.py                                   # train / predict CLI
├─ Code/
│  ├─ Labelling/
│  │  └─ data_labelling.py                                  # merge + equal-importance labels + journal
│  └─ Normalization/
│     ├─ normalize_csvs.py                                  # normalize stats/system logs
│     └─ normalize_confusion_csvs.py                        # tidy/aggregate confusion matrices
├─ federated_mpi/
│  ├─ mpi_main.py
│  ├─ mpi_client.py
│  ├─ mpi_server.py
│  └─ mpi_utils.py
├─ config/
│  └─ config.py                                             # e.g., NUM_ROUNDS, etc.
└─ logs/
   ├─ confusion_csv/client_*/round_*.csv                    # full matrices (optional)
   └─ confusion_tidy/client_*.csv                           # tidy append format (optional)
```

---

## 1) 🤝 Run Federated Learning with MPI

Use **1 server + N clients** (total processes = N+1). Example with 4 clients:

```bash
mpiexec -n 5 python federated_mpi/mpi_main.py
```

What happens:
- Data is partitioned across clients.
- Each round: server broadcasts weights → clients train locally → send weights/metrics back → server aggregates.
- Clients log **statistical utility** (accuracy/loss/variance), **system metrics** (CPU time, RAM, NET, GPU), and optionally **confusion matrices**.

Outputs you should see after a run (paths may vary):
- `all_clients_stats.csv`
- `client_system_metrics.csv`
- Confusion matrices:
  - per-round under `logs/confusion_csv/client_{k}/round_{r}.csv`
  - or **tidy** appends under `logs/confusion_tidy/client_{k}.csv`

> On clusters: replace `mpiexec` with your launcher (`mpirun`, `srun`) and set `OMP_NUM_THREADS=1` if needed.

---

## 2) 🧼 Normalize Logs

Normalize raw per-client/per-round CSVs so features are on a comparable [0,1] scale.

```bash
python Code/Normalization/normalize_csvs.py
python Code/Normalization/normalize_confusion_csvs.py
```

Expected outputs (under `Data/`, names may match your scripts):
- `all_clients_stats_normalized.csv`
- `client_system_metrics_normalized.csv`
- `confusion_matrices_tidy.csv` (and/or an aggregated column such as `confusion_mean`)

> If paths differ on your machine, open each script and adjust the input/output paths at the top.

---

## 3) 🏷️ Labelization (Equal-Importance Thresholds)

Create an interpretable label for **high-utility** clients using **equal-importance** thresholds:

**How it works**
1. For each numeric feature, compute its **median** (global or per round).
2. For a client/round, assign **1** if the feature value > median, else **0**.
3. **classification_score** = sum of these 0/1s over all features.
4. **cutoff** = ½ × (#features).
5. **class_label** = 1 if `classification_score ≥ cutoff`, else 0.

Run:

```bash
python Code/Labelling/data_labelling.py
```

Outputs:
- `Data/clients_labeled_equal_importance_with_confusion.csv`
- `Data/selection_journal.txt` (per-round medians + selected client ranks)

---

## 4) 🧠 Train the Shallow MLP (Supervised)

Train an MLP to predict `class_label` from the normalized features (IDs & derived columns excluded).  
The script also **splits by round** to avoid leakage and saves artifacts.

```bash
python mlp_client_selector.py train   --data Data/clients_labeled_equal_importance_with_confusion.csv   --artifacts Models   --epochs 200   --patience 10
```

Artifacts:
- `Models/mlp_client_selector.h5` – trained model
- `Models/scaler.pkl` – StandardScaler fitted on train
- `Models/features.json` – exact feature names used

**Model architecture (shallow)**
- Input d → **Dense(64, ReLU)** → **Dropout(0.2)** → **Dense(1, Sigmoid)**
- Loss: Binary Cross-Entropy; Optimizer: Adam; EarlyStopping on val AUC
- Class weights to offset imbalance

---

## 5) 🔮 Predict on New Client Rows

Prepare a CSV with the **same feature columns** (use the template command below). Then:

```bash
python mlp_client_selector.py predict   --data Data/new_client_metrics.csv   --artifacts Models   --out Data/new_client_predictions.csv   --threshold 0.5
```

Output columns added:
- `p_select` — probability the client should be selected  
- `mlp_label` — 0/1 decision given the threshold

**Make a feature template** (empty header with exactly the required columns):

```bash
python mlp_client_selector.py make-template   --train-data Data/clients_labeled_equal_importance_with_confusion.csv   --out Data/new_client_template.csv
```

**Tip (deployment):** For each round, rank clients by `p_select` and choose **top-k** that fit your budget; this is often better than a global threshold.

---

## 🧪 Quick sanity tests

- `Data/new_client_quantile_grid.csv` — rows at ≈20th/50th/80th percentiles → should produce a mix of labels.
- You can also lower/raise `--threshold` (e.g., 0.45 vs 0.6) to trade recall vs precision.

---

## 🧷 Troubleshooting

- **All predictions = 1** on random inputs: random [0,1] may not match your feature distribution. Use the **quantile grid** or real rows; tune `--threshold`.
- **FileNotFoundError on paths**: run commands from the repository root, or adjust the paths at the top of each script to your layout.
- **MPI run hangs**: ensure each client sends both weights and metrics; our client guards skip training if a client has zero samples and still sends back unchanged weights.
- **psutil GPU/CPU metrics**: GPU may be absent; `cpu_freq()` can be `None`—the client code already guards these.

---

## 📝 Repro Notes

- Set `SEED=42` (already in `mlp_client_selector.py`) for reproducibility.
- Edit `config/config.py` for round counts and other FL knobs (e.g., `NUM_ROUNDS`).
- Keep raw logs + normalized CSVs under version control (or at least schema versions) for auditability.

---

## ✅ End-to-End Checklist

1. `mpiexec -n N+1 python federated_mpi/mpi_main.py`  
2. `python Code/Normalization/normalize_csvs.py`  
3. `python Code/Normalization/normalize_confusion_csvs.py`  
4. `python Code/Labelling/data_labelling.py`  
5. `python mlp_client_selector.py train --data Data/clients_labeled_equal_importance_with_confusion.csv --artifacts Models`  
6. `python mlp_client_selector.py predict --data Data/new_client_metrics.csv --artifacts Models --out Data/new_client_predictions.csv`
