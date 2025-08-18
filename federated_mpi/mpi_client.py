import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import psutil
import GPUtil
import tensorflow as tf
from data_loader import load_and_preprocess_data
from model import build_cnn_model
from mpi_utils import serialize_weights, deserialize_weights
from tensorflow.keras.utils import to_categorical
from mpi4py import MPI
from config.config import NUM_ROUNDS
from datetime import datetime
import csv
import time

from sklearn.metrics import confusion_matrix  # for CM feature
from pathlib import Path


# -----------------------------
# Helpers for metrics & logging
# -----------------------------

def evaluate_keras_model(model, x, y):
    results = model.evaluate(x, y, verbose=0)
    loss = float(results[0])
    accuracy = float(results[1]) * 100.0  # percentage
    return loss, accuracy

def compute_data_variance_tf(x):
    x_flat = x.reshape(x.shape[0], -1)
    return float(np.var(x_flat))

def log_statistical_utility_tf(rank, round_num, x, y, model):
    """
    Writes to Data/logs/clients_stats.csv (ONLY when active/selected).
    Returns the same values as a dict for sending to server.
    """
    csv_path = "Data/logs/clients_stats.csv"
    os.makedirs("Data/logs", exist_ok=True)

    data_size = int(x.shape[0])
    data_variance = compute_data_variance_tf(x)
    local_loss, local_accuracy = evaluate_keras_model(model, x, y)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "client_rank", "round", "data_size",
                "data_variance", "local_loss", "local_accuracy"
            ])
        writer.writerow([
            rank, round_num, data_size,
            data_variance, local_loss, local_accuracy
        ])

    return {
        "client_rank": rank,
        "round": round_num,
        "data_size": data_size,
        "data_variance": data_variance,
        "local_loss": local_loss,
        "local_accuracy": local_accuracy
    }

def log_confusion_matrix(rank, round_num, model, x_data, y_data_oh):
    """
    Computes & saves the multiclass confusion matrix (ONLY when active/selected).
    Returns micro TP/FP/FN/TN, macro precision/recall/F1, and 'confusion_mean'.
    """
    base_dir = Path("Data/logs/confusion_csv") / f"client_{rank}"
    base_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.argmax(y_data_oh, axis=1)
    y_pred = np.argmax(model.predict(x_data, batch_size=256, verbose=0), axis=1)
    cm = confusion_matrix(y_true, y_pred)  # (C, C)

    # Save full confusion matrix as CSV
    out_csv = base_dir / f"round_{round_num:03d}.csv"
    np.savetxt(out_csv, cm, fmt="%d", delimiter=",")

    # Row-normalize then average all cells → stable scalar feature for selector
    # Row-normalize to get per-class distributions
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-9
    cm_norm = cm / row_sums

    # Use the mean of the diagonal of the normalized CM (avg per-class accuracy)
    confusion_mean = float(np.trace(cm_norm) / cm_norm.shape[0])


    # Micro-aggregated "correct vs incorrect"
    correct = (y_true == y_pred)
    tp = int(np.sum(correct))
    total = int(len(y_true))
    fp = int(np.sum(~correct))
    fn = fp
    tn = int(total - tp - fp - fn)  # 0 in this collapse

    # Macro precision/recall/F1 across classes
    per_class_prec, per_class_rec, per_class_f1 = [], [], []
    C = int(cm.shape[0])
    for c in range(C):
        tp_c = int(cm[c, c])
        fp_c = int(cm[:, c].sum() - tp_c)
        fn_c = int(cm[c, :].sum() - tp_c)
        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
        rec_c  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
        f1_c   = (2 * prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) else 0.0
        per_class_prec.append(prec_c)
        per_class_rec.append(rec_c)
        per_class_f1.append(f1_c)

    precision_macro = float(np.mean(per_class_prec))
    recall_macro    = float(np.mean(per_class_rec))
    f1_macro        = float(np.mean(per_class_f1))

    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "confusion_mean": confusion_mean,  # REQUIRED by features.json
    }

def collect_system_metrics(rank, round_num):
    p = psutil.Process(os.getpid())
    cpu_times = psutil.cpu_times()
    cf = psutil.cpu_freq()
    cpu_freq = cf.current if cf else 0.0
    # RSS in MB
    ram_used = float(p.memory_info().rss) / 1e6
    net = psutil.net_io_counters()
    net_sent = int(net.bytes_sent)
    net_recv = int(net.bytes_recv)

    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_mem_used = float(gpu.memoryUsed)
            gpu_load = float(gpu.load)
        else:
            gpu_mem_used = 0.0
            gpu_load = 0.0
    except Exception:
        gpu_mem_used = 0.0
        gpu_load = 0.0

    return {
        "client_rank": rank,
        "round": round_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_time": float(cpu_times.user + cpu_times.system),
        "cpu_freq_mhz": float(cpu_freq),
        "ram_used_mb": ram_used,
        "net_sent_bytes": net_sent,
        "net_recv_bytes": net_recv,
        "gpu_mem_used_mb": gpu_mem_used,
        "gpu_load": gpu_load
    }

def compute_per_round_metrics(start_snapshot, end_snapshot):
    # Ensure numerics (some systems may return None)
    gpu_mem = end_snapshot.get("gpu_mem_used_mb", 0.0)
    gpu_ld  = end_snapshot.get("gpu_load", 0.0)
    gpu_mem = float(gpu_mem) if gpu_mem is not None else 0.0
    gpu_ld  = float(gpu_ld)  if gpu_ld  is not None else 0.0

    return {
        "cpu_time": float(end_snapshot["cpu_time"] - start_snapshot["cpu_time"]),
        "ram_used_mb": float(end_snapshot["ram_used_mb"]),
        "net_sent_bytes": int(end_snapshot["net_sent_bytes"] - start_snapshot["net_sent_bytes"]),
        "net_recv_bytes": int(end_snapshot["net_recv_bytes"] - start_snapshot["net_recv_bytes"]),
        "gpu_mem_used_mb": gpu_mem,
        "gpu_load": gpu_ld
    }


# -----------------------------
# Main client loop
# -----------------------------

def run_client(comm, rank):
    print(f"    [Client {rank}] Initializing...", flush=True)
    print(f"    [Client {rank}] Starting on host: {os.uname().nodename}", flush=True)

    enforce_selection = os.getenv("FEDSEL_ENFORCE", "0") == "1"

    if rank == 0:
        print(f"[Server {rank}] Nothing to do in run_client()", flush=True)
        return

    # === Load shared data (memory-mapped) ===
    (x_train_mmap, y_train_mmap), _ = load_and_preprocess_data(mmap_mode="r", preprocess=False)
    num_clients = comm.Get_size() - 1
    total_samples = x_train_mmap.shape[0]

    # === Dirichlet-based unbalanced slicing ===
    if rank == 1:
        proportions = np.random.dirichlet(alpha=[0.5] * num_clients)
        print(f"[Client {rank}] Generated proportions: {proportions}", flush=True)
    else:
        proportions = None
    proportions = comm.bcast(proportions, root=1)

    cumulative = np.cumsum(proportions)
    start_idx = int(total_samples * cumulative[rank - 2]) if rank > 1 else 0
    end_idx = int(total_samples * cumulative[rank - 1])

    x_client = np.array(x_train_mmap[start_idx:end_idx])
    y_client = np.array(y_train_mmap[start_idx:end_idx])

    print(f"[Client {rank}] Assigned {x_client.shape[0]} samples (from {start_idx} to {end_idx})", flush=True)

    x_client = x_client.astype("float32") / 255.0
    y_client = to_categorical(y_client, 10)

    for round_num in range(1, NUM_ROUNDS + 1):
        model = build_cnn_model()

        # === (A) Receive selection list IF enforcement is enabled ===
        active = True
        if enforce_selection:
            # Server broadcasts: once before round 1, then after each round
            selected = comm.bcast(None, root=0)
            active = (rank in selected)
            if not active:
                print(f"    [Client {rank}] Round {round_num} - SKIP (not selected).", flush=True)
            else:
                print(f"    [Client {rank}] Round {round_num} - Selected, will train.", flush=True)

        # === (B) Receive global weights, set model ===
        print(f"    [Client {rank}] Round {round_num} - Waiting for global weights...", flush=True)
        global_weights = comm.bcast(None, root=0)
        model.set_weights(global_weights)

        # === (C) Train if active; otherwise skip and echo weights ===
        start_snapshot = collect_system_metrics(rank, round_num)
        if active:
            model.fit(x_client, y_client, epochs=1, batch_size=32, verbose=0)
        end_snapshot = collect_system_metrics(rank, round_num)

        # System metrics (always send, lightweight)
        metrics = compute_per_round_metrics(start_snapshot, end_snapshot)
        metrics.update({
            "client_rank": rank,
            "round": round_num,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "participated": 1 if active else 0,
        })

        # === (D) Statistical utility + confusion ===
        if active:
            # Only selected clients journalize heavy CSVs
            stats = log_statistical_utility_tf(rank, round_num, x_client, y_client, model) or {}
            conf  = log_confusion_matrix(rank, round_num, model, x_client, y_client) or {}
            metrics.update(stats)
            metrics.update(conf)
        else:
            # Placeholders so selector has *all* required features each round
            metrics.update({
                "data_size": int(x_client.shape[0]),
                "data_variance": float(np.var(x_client.astype(np.float32))),
                "local_loss": 0.0,
                "local_accuracy": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                "confusion_mean": 0.0,   # REQUIRED by features.json
            })

        # === (E) Prepare weights to send ===
        if active:
            updated_weights = serialize_weights(model.get_weights())
        else:
            # Echo the global weights unchanged so server can mask non-participants
            updated_weights = serialize_weights(global_weights)

        print(f"    [Client {rank}] Round {round_num} - Sending updated weights to server...", flush=True)

        # === (F) Send weights and metrics ===
        try:
            comm.send(updated_weights, dest=0, tag=rank)
        except Exception as e:
            print(f"[Client {rank}] Failed to send weights: {e}", flush=True)

        try:
            comm.send(metrics, dest=0, tag=rank + 100)
        except Exception as e:
            print(f"[Client {rank}] Failed to send metrics: {e}", flush=True)

        # === (G) End-of-round barrier (aligns with server before next cohort bcast) ===
        try:
            comm.Barrier()
        except Exception as e:
            print(f"[Client {rank}] Barrier failed at round {round_num}: {e}", flush=True)

    print(f"    [Client {rank}] Training complete. Waiting for others...", flush=True)
    print(f"    [Client {rank}] Exiting.", flush=True)
    return
