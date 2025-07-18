import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import builtins

import numpy as np
import matplotlib.pyplot as plt
from model import build_cnn_model
from data_loader import load_and_preprocess_data
from mpi_utils import serialize_weights, deserialize_weights

def average_weights(weight_list):
    avg_weights = []
    for weights in zip(*weight_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

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

    for round_num in range(1, 21):
        print(f"\n===== Round {round_num} =====",flush=True)
        print("Broadcasting global model to all clients...", flush=True)

        comm.bcast(global_weights, root=0)

        client_weights = []

        for i in range(1, num_clients + 1):
            print(f"Waiting for model weights from client {i}...", flush=True)
            received = comm.recv(source=i, tag=i)
            print(f"Received weights from client {i}", flush=True)
            client_weights.append(deserialize_weights(received))

        print("Averaging model weights...")
        global_weights = average_weights(client_weights)
        model.set_weights(global_weights)

        print("Evaluating updated global model on test set...", flush=True)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Round {round_num} Evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}", flush=True)

        rounds.append(round_num)
        acc_list.append(accuracy)
        loss_list.append(loss)

    print("\n=== Federated Training Complete ===",flush=True)
    print("Generating training metrics plot...", flush=True)

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
    plt.savefig(f"federated_metrics.png")
