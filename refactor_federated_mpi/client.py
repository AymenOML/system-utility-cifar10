from mpi4py import MPI
from tensorflow.keras.utils import to_categorical
import numpy as np
from datetime import datetime

from model import build_cnn_model
from data_loader import load_and_preprocess_data
from config.config import NUM_ROUNDS
from mpi_utils import serialize_weights
from client_utils import (
    collect_system_metrics,
    compute_per_round_metrics,
    log_confusion_matrix,
    log_statistical_utility_tf
)


class FederatedClient:
    def __init__(self, comm, rank):
        """
        Initialize the federated learning client.
        :param comm: MPI communicator.
        :param rank: MPI rank of this client.
        """
        self.comm = comm
        self.rank = rank
        self.model = None
        self.x_client = None
        self.y_client = None

    def prepare_data(self):
        """
        Prepare client-specific data partition using Dirichlet distribution
        for unbalanced slicing across clients.
        """
        (x_train, y_train), _ = load_and_preprocess_data(mmap_mode='r', preprocess=False)
        num_clients = self.comm.Get_size() - 1  # Exclude server
        total_samples = x_train.shape[0]

        # Dirichlet-based unbalanced slicing
        if self.rank == 1:
            proportions = np.random.dirichlet(alpha=[0.5] * num_clients)
        else:
            proportions = None
        proportions = self.comm.bcast(proportions, root=1)

        cumulative = np.cumsum(proportions)
        start_idx = int(total_samples * cumulative[self.rank - 2]) if self.rank > 1 else 0
        end_idx = int(total_samples * cumulative[self.rank - 1])

        self.x_client = np.array(x_train[start_idx:end_idx], dtype='float32') / 255.0
        self.y_client = to_categorical(np.array(y_train[start_idx:end_idx]), 10)

    def run(self):
        """
        Main training loop for the client.
        Receives global weights, trains locally, sends back updated weights and system metrics.
        """
        self.prepare_data()

        for round_num in range(1, NUM_ROUNDS + 1):
            # Build model for each round
            self.model = build_cnn_model()

            # Receive the current global weights from the server
            global_weights = self.comm.bcast(None, root=0)
            self.model.set_weights(global_weights)

            # System metrics before training
            start_snapshot = collect_system_metrics(self.rank, round_num)

            # Local training
            self.model.fit(self.x_client, self.y_client, epochs=1, batch_size=32, verbose=0)

            # System metrics after training
            end_snapshot = collect_system_metrics(self.rank, round_num)

            # Compute metrics delta for the round
            metrics = compute_per_round_metrics(start_snapshot, end_snapshot)
            metrics.update({
                "client_rank": self.rank,
                "round": round_num,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Log statistical utility
            log_statistical_utility_tf(self.rank, round_num, self.x_client, self.y_client, self.model)

            # Log confusion matrix
            log_confusion_matrix(self.rank, round_num, self.model, self.x_client, self.y_client)

            # Send updated weights back to server
            serialized = serialize_weights(self.model.get_weights())
            self.comm.send(serialized, dest=0, tag=self.rank)

            # Send metrics to server
            self.comm.send(metrics, dest=0, tag=self.rank + 100)
