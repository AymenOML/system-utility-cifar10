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
            client_weights.append(deserialize_weights(received))

        client_metrics = []
        for i in range(1, num_clients + 1):

            try:
                metrics = comm.recv(source=i, tag=i + 100)
            except Exception as e:
                print(f"[Server] Error receiving from client {i}: {e}", flush=True)
                metrics = None

            metrics['round'] = round_num  # Add round info
            client_metrics.append(metrics)
            print(f"Received system metrics from client {i}: {metrics}")

        all_client_metrics.extend(client_metrics)

        print("Averaging model weights...")
        global_weights = average_weights(client_weights)
        model.set_weights(global_weights)

        print("Evaluating updated global model on test set...", flush=True)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Round {round_num} Evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}", flush=True)

        rounds.append(round_num)
        acc_list.append(accuracy)
        loss_list.append(loss)

    print("\n=== Federated Training Complete ===", flush=True)
    print("Generating training metrics plot...", flush=True)

    df_metrics = pd.DataFrame(all_client_metrics)
    df_metrics.to_csv("Data/logs/clients_system.csv", index=False)

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
