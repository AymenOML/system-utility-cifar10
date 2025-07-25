import sys
import os
from config.config import NUM_ROUNDS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import builtins

import numpy as np
import matplotlib.pyplot as plt
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
    df_metrics.to_csv("client_system_metrics.csv", index=False)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, acc_list, marker='o')
    plt.title("Federated Test Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(rounds, loss_list, marker='o', color='orange')
    plt.title("Federated Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("federated_metrics.png")
    return
