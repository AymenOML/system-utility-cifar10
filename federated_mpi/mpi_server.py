import sys
import os
from config.config import NUM_ROUNDS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import builtins

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator, PercentFormatter
from model import build_cnn_model
from data_loader import load_and_preprocess_data
from mpi_utils import serialize_weights, deserialize_weights
from mpi4py import MPI
import csv
import pandas as pd

# >>> ADD: selection pipeline and small helpers
from pathlib import Path
from postprocess_pipeline import run_round_selection

# >>> ADD: paths & selection config
DATA_DIR = "Data"
LOGS_DIR = os.path.join(DATA_DIR, "logs")

SYSTEM_CSV = os.path.join(LOGS_DIR, "clients_system.csv")
SYSTEM_FIELDS = [
    "client_rank", "round",
    "cpu_time", "ram_used_mb", "net_sent_bytes", "net_recv_bytes",
    "gpu_mem_used_mb", "gpu_load", "timestamp",
]

SELECTION_JOURNAL = os.path.join(DATA_DIR, "selection_journal.txt")
ARTIFACTS_DIR = "Models"      # change to "Model" if that's your folder name
MLP_THRESHOLD = 0.5 # Put to 0 for data collection


# >>> ADD: safe CSV appender (doesn't affect your plotting)
def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def _append_dict_csv(csv_path: str, row: dict, fieldnames: list):
    _ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def average_weights(weight_list):
    if not weight_list or any(w is None for w in weight_list):
        raise ValueError("Invalid weights received from clients.")
    return [np.mean(weights, axis=0) for weights in zip(*weight_list)]


def run_server(comm):

    print("=== Federated Server Started ===", flush=True)
    num_clients = comm.Get_size() - 1
    print(f"Total processes: {num_clients + 1} (1 server + {num_clients} clients)\n", flush=True)

    model = build_cnn_model()
    global_weights = model.get_weights()

    # Load test data once
    _, (x_test, y_test) = load_and_preprocess_data()

    acc_list = []
    loss_list = []
    rounds = []
    all_client_metrics = []

    # >>> ADD: Round-1 policy — everyone trains
    selected_for_this_round = list(range(1, num_clients + 1))
    selected_for_next_round = selected_for_this_round[:]

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n===== Round {round_num} =====", flush=True)

        # >>> ADD: broadcast who should TRAIN this round
        comm.bcast(selected_for_this_round, root=0)

        print("Broadcasting global model to all clients...", flush=True)
        # Keep your existing broadcast (list of numpy arrays)
        comm.bcast(global_weights, root=0)

        client_weights = []
        for i in range(1, num_clients + 1):
            print(f"Waiting for model weights from client {i}...", flush=True)

            try:
                received = comm.recv(source=i, tag=i)
            except Exception as e:
                print(f"[Server] Error receiving from client {i}: {e}", flush=True)
                received = None

            print(f"Received weights from client {i}", flush=True)

            # >>> CHANGE (non-breaking): only aggregate from selected trainers
            if received is not None and (i in selected_for_this_round):
                try:
                    client_weights.append(deserialize_weights(received))
                except Exception as e:
                    print(f"[Server][WARN] Could not deserialize weights from client {i}: {e}", flush=True)

        client_metrics = []
        for i in range(1, num_clients + 1):

            try:
                metrics = comm.recv(source=i, tag=i + 100)
            except Exception as e:
                print(f"[Server] Error receiving from client {i}: {e}", flush=True)
                metrics = None

            # >>> CHANGE (robustness): guard against None
            if not isinstance(metrics, dict):
                metrics = {}

            metrics['round'] = round_num  # Add round info
            client_metrics.append(metrics)
            print(f"Received system metrics from client {i}: {metrics}")

            # >>> ADD: write system metrics during the round (append 1 row per client)
            system_row = {
                "client_rank": i,
                "round": round_num,
                "cpu_time": metrics.get("cpu_time"),
                "ram_used_mb": metrics.get("ram_used_mb"),
                "net_sent_bytes": metrics.get("net_sent_bytes"),
                "net_recv_bytes": metrics.get("net_recv_bytes"),
                "gpu_mem_used_mb": metrics.get("gpu_mem_used_mb"),
                "gpu_load": metrics.get("gpu_load"),
                "timestamp": metrics.get("timestamp", pd.Timestamp.utcnow().timestamp()),
            }
            _append_dict_csv(SYSTEM_CSV, system_row, SYSTEM_FIELDS)

        all_client_metrics.extend(client_metrics)

        print("Averaging model weights...")
        if len(client_weights) == 0:
            print("[Server] No weight updates this round — keeping previous global weights.", flush=True)
        else:
            global_weights = average_weights(client_weights)
            model.set_weights(global_weights)

        print("Evaluating updated global model on test set...", flush=True)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Round {round_num} Evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}", flush=True)

        rounds.append(round_num)
        acc_list.append(accuracy)
        loss_list.append(loss)

        # >>> ADD: per-round selection (normalize -> label -> MLP -> select)
        try:
            pred_csv_path, selected_for_next_round = run_round_selection(
                round_id=round_num,
                base_dir=DATA_DIR,
                artifacts_dir=ARTIFACTS_DIR,
                threshold=MLP_THRESHOLD,
                journal_path=SELECTION_JOURNAL,
            )
            print(f"[Server] Round {round_num}: selection file: {pred_csv_path}", flush=True)
            print(f"[Server] Selected for next round: {selected_for_next_round}", flush=True)
        except Exception as e:
            print(f"[Server][WARN] Selection failed at round {round_num}: {e}  -> Falling back to all-train.", flush=True)
            selected_for_next_round = list(range(1, num_clients + 1))

        # >>> ADD: carry decision forward
        selected_for_this_round = selected_for_next_round[:]

    print("\n=== Federated Training Complete ===", flush=True)
    print("Generating training metrics plot...", flush=True)

    # >>> CHANGE (non-breaking): avoid duplicating system CSV if we already appended per-round
    if not os.path.exists(SYSTEM_CSV):
        df_metrics = pd.DataFrame(all_client_metrics)
        _ensure_dir(os.path.dirname(SYSTEM_CSV))
        df_metrics.to_csv(SYSTEM_CSV, index=False)
    else:
        print("[Server] System CSV already written per-round; skipping final rewrite.", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax1, ax2 = axes

    # --- Accuracy ---------------------------------------------------------------
    ax1.plot(rounds, acc_list, marker='o')
    ax1.set_title("Federated Test Accuracy")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")

    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))  # 0.1 précision
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))   # pas de 5%
    ax1.yaxis.set_minor_locator(MultipleLocator(0.01))   # pas de 1%

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    ax1.grid(True, which='major', linewidth=0.8)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.6)

    ax2.plot(rounds, loss_list, marker='o', color='orange')
    ax2.set_title("Federated Test Loss")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")

    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_minor_locator(AutoMinorLocator())

    ax2.grid(True, which='major', linewidth=0.8)
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.6)

    fig.savefig("Data/federated_metrics.png", dpi=200)

    return
