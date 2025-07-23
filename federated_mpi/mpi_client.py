import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from data_loader import load_and_preprocess_data
from model import build_cnn_model
from mpi_utils import serialize_weights, deserialize_weights
from mpi4py import MPI

def run_client(comm, rank):
    print(f"    [Client {rank}] Initializing...", flush=True)

    (x_train, y_train), _ = load_and_preprocess_data()

    # Split training data by client rank
    num_clients = comm.Get_size() - 1
    total_samples = x_train.shape[0]
    samples_per_client = total_samples // num_clients
    start = (rank - 1) * samples_per_client
    end = start + samples_per_client

    x_client = x_train[start:end]
    y_client = y_train[start:end]

    model = build_cnn_model()

    ### Change for number of rounds
    for round_num in range(1, 21):
        print(f"    [Client {rank}] Round {round_num} - Waiting for global weights...", flush=True)
        global_weights = comm.bcast(None, root=0)
        model.set_weights(global_weights)

        print(f"    [Client {rank}] Round {round_num} - Training on local data...", flush=True)
        model.fit(x_client, y_client, epochs=1, batch_size=32, verbose=0)

        updated_weights = serialize_weights(model.get_weights())
        print(f"    [Client {rank}] Round {round_num} - Sending updated weights to server...", flush=True)
        comm.send(updated_weights, dest=0, tag=rank)


    print(f"    [Client {rank}] Training complete. Waiting for others...", flush=True)
    comm.Barrier()
    MPI.Finalize()
    sys.exit(0)
    print(f"    [Client {rank}] Exiting.", flush=True)
