import sys
import os
from config.config import NUM_ROUNDS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import builtins

# Use a headless backend for plots (must be before importing pyplot)
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator, PercentFormatter
from model import build_cnn_model
from data_loader import load_and_preprocess_data
from mpi_utils import serialize_weights, deserialize_weights
from mpi4py import MPI
import csv
import pandas as pd

# MLP round selector
from federated_mpi.round_selector import RoundSelector

# ---- NEW: selected-only journaling flag
SELECTED_ONLY = os.getenv("FEDSEL_JOURNAL_SELECTED_ONLY", "0") == "1"


def average_weights(weight_list):
    if not weight_list or any(w is None for w in weight_list):
        raise ValueError("Invalid weights received from clients.")
    return [np.mean(weights, axis=0) for weights in zip(*weight_list)]


def _select_next_clients_with_mlp(selector, client_metrics, round_idx, num_clients):
    """
    selector: RoundSelector
    client_metrics: list[dict] or dict[int->dict] with per-client metrics for this round.
    round_idx: int
    num_clients: int
    Returns: list[int] selected client ids for next round
    """
    client_reports = {}

    if isinstance(client_metrics, dict):
        # {cid -> metrics_dict}
        for cid, md in client_metrics.items():
            rec = dict(md) if isinstance(md, dict) else {}
            rec["round"] = round_idx
            client_reports[int(cid)] = rec
    elif isinstance(client_metrics, list):
        # [metrics_dict, ...] (optionally each contains 'client_rank')
        for idx, md in enumerate(client_metrics, start=1):
            if isinstance(md, dict):
                cid = int(md.get("client_rank", idx))
                rec = dict(md)
            else:
                cid = idx
                rec = {}
            rec["round"] = round_idx
            client_reports[cid] = rec
    else:
        # Fallback: select everyone if we can't parse
        return list(range(1, num_clients + 1))

    round_dir = os.path.join("Data", "logs", f"round_{round_idx:03d}")
    os.makedirs(round_dir, exist_ok=True)  # ensure the folder exists even if selector errors
    selected = selector.run_for_round(round_idx, client_reports, out_dir=round_dir)
    return selected


def run_server(comm):

    print("=== Federated Server Started ===", flush=True)
    print(f"[server] CWD = {os.getcwd()}", flush=True)
    os.makedirs("Data/logs", exist_ok=True)

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

    # Selector (threshold mode in your round_selector.py)
    try:
        selector = RoundSelector(model_dir="Model")  # threshold mode ignores top_k
        # If your RoundSelector has .threshold (from the patch), log it:
        th = getattr(selector, "threshold", None)
        if th is not None:
            print(f"[selector] Loaded MLP (threshold={th})", flush=True)
        else:
            print(f"[selector] Loaded MLP", flush=True)
    except Exception as e:
        print("[selector] WARNING: could not initialize MLP selector:", e, flush=True)
        selector = None

    # Enforcement flag and initial cohort (for round 1)
    ENFORCE = os.getenv("FEDSEL_ENFORCE", "0") == "1"
    print(f"[selector] ENFORCE = {ENFORCE}", flush=True)
    next_selected = list(range(1, num_clients + 1))  # default cohort = all clients

    if ENFORCE:
        # Broadcast initial cohort so clients can gate round 1
        comm.bcast(next_selected, root=0)

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n===== Round {round_num} =====", flush=True)
        print("Broadcasting global model to all clients...", flush=True)
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
            client_weights.append(deserialize_weights(received) if received is not None else None)

        client_metrics = []
        for i in range(1, num_clients + 1):
            try:
                metrics = comm.recv(source=i, tag=i + 100)
            except Exception as e:
                print(f"[Server] Error receiving from client {i}: {e}", flush=True)
                metrics = None

            if metrics is None:
                metrics = {"client_rank": i, "participated": 1}  # guard to avoid crashes
            metrics["round"] = round_num  # Add round info

            client_metrics.append(metrics)
            print(f"Received system metrics from client {i}: {metrics}")

        all_client_metrics.extend(client_metrics)

        # === Selector hook: compute next cohort (logged every round) ===
        if selector is not None:
            try:
                client_reports = {cid: md for cid, md in enumerate(client_metrics, start=1)}
                round_dir = os.path.join("Data", "logs", f"round_{round_num:03d}")
                os.makedirs(round_dir, exist_ok=True)  # make sure folder exists
                next_selected = selector.run_for_round(round_num, client_reports, out_dir=round_dir)
                print(f"[selector] Round {round_num}: selected for next round: {next_selected}", flush=True)
                # Append a concise participation journal (selected-only)
                journal_path = "Data/participation_journal.csv"
                new_file = not os.path.exists(journal_path)
                with open(journal_path, "a") as jf:
                    if new_file:
                        jf.write("round,client_rank\n")
                    for cid in next_selected:
                        jf.write(f"{round_num},{cid}\n")
            except Exception as e:
                print(f"[selector] ERROR at round {round_num}: {e}", flush=True)
                next_selected = list(range(1, num_clients + 1))  # fallback: everyone
        else:
            next_selected = list(range(1, num_clients + 1))  # no selector: everyone

        # --- Broadcast next round's cohort (only when enforcing) ---
        comm.Barrier()  # keep ranks in sync
        if ENFORCE and round_num < NUM_ROUNDS:
            comm.bcast(next_selected, root=0)

        # --- Average weights (mask out non-participants if enforcing) ---
        participants_mask = [(m or {}).get("participated", 1) == 1 for m in client_metrics]
        trained_count = sum(participants_mask)
        print(f"[selector] Round {round_num}: participants used in averaging = {trained_count}/{num_clients}", flush=True)

        # ---- NEW: quick cross-check with selection length (informative only)
        if ENFORCE and selector is not None:
            try:
                if len(next_selected) != trained_count:
                    print(f"[selector][warn] mismatch: selected={len(next_selected)} vs participated={trained_count}", flush=True)
            except Exception:
                pass

        if ENFORCE:
            client_weights_used = [w for w, keep in zip(client_weights, participants_mask) if keep]
            if len(client_weights_used) == 0:
                # Edge case: no one trained (e.g., high threshold and all filtered out)
                print("[selector] No participants passed threshold; reusing previous global weights.", flush=True)
                client_weights_used = None
        else:
            client_weights_used = client_weights

        print("Averaging model weights...")
        if client_weights_used:
            global_weights = average_weights(client_weights_used)
            model.set_weights(global_weights)
        else:
            # keep model/weights unchanged this round
            pass

        print("Evaluating updated global model on test set...", flush=True)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Round {round_num} Evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}", flush=True)

        rounds.append(round_num)
        acc_list.append(accuracy)
        loss_list.append(loss)

    print("\n=== Federated Training Complete ===", flush=True)
    print("Generating training metrics plot...", flush=True)

    # Persist system metrics CSV
    df_metrics = pd.DataFrame(all_client_metrics)
    os.makedirs("Data/logs", exist_ok=True)

    # ---- NEW: selected-only journaling for the big system CSV (optional)
    if SELECTED_ONLY and "participated" in df_metrics.columns:
        df_metrics = df_metrics[df_metrics["participated"] == 1]

    df_metrics.to_csv("Data/logs/clients_system.csv", index=False)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax1, ax2 = axes

    # --- Accuracy ---------------------------------------------------------------
    ax1.plot(rounds, acc_list, marker='o')
    ax1.set_title("Federated Test Accuracy")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")

    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.01))

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    ax1.grid(True, which='major', linewidth=0.8)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.6)

    # --- Loss -------------------------------------------------------------------
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

    os.makedirs("Data", exist_ok=True)
    fig.savefig("Data/federated_metrics.png", dpi=200)

    return
