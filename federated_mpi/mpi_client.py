import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
from data_loader import load_and_preprocess_data
from model import build_cnn_model
from mpi_utils import serialize_weights, deserialize_weights

def run_client(comm, rank):
    print(f"Client {rank} initializing...")
    (x_train, y_train), _ = load_and_preprocess_data()

    # Split data among clients
    num_clients = comm.Get_size() - 1
    total_samples = x_train.shape[0]
    samples_per_client = total_samples // num_clients
    start = (rank - 1) * samples_per_client
    end = start + samples_per_client

    x_client = x_train[start:end]
    y_client = y_train[start:end]

    model = build_cnn_model()
    model.set_weights(comm.bcast(None, root=0))  # receive global weights

    model.fit(x_client, y_client, epochs=1, batch_size=32, verbose=0)

    # Send updated weights to server
    comm.send(serialize_weights(model.get_weights()), dest=0, tag=rank)
    print(f"Client {rank} finished and sent weights.")
