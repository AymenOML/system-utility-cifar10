from mpi4py import MPI
import pandas as pd
from model import build_cnn_model
from data_loader import load_and_preprocess_data
from config.config import NUM_ROUNDS
from mpi_utils import deserialize_weights
from server_utils import average_weights, save_client_metrics_csv, plot_training_curves


class FederatedServer:
    def __init__(self, comm):
        """
        Initialize the federated learning server.
        :param comm: MPI communicator.
        """
        self.comm = comm
        self.model = None
        self.global_weights = None

    def run(self):
        """
        Main federated learning coordination loop for the server.
        Broadcasts global weights, collects client updates, aggregates them,
        and evaluates the global model each round.
        """
        num_clients = self.comm.Get_size() - 1  # exclude server
        self.model = build_cnn_model()
        self.global_weights = self.model.get_weights()

        # Load test set for evaluation
        _, (x_test, y_test) = load_and_preprocess_data()

        acc_list, loss_list, rounds, all_client_metrics = [], [], [], []

        for round_num in range(1, NUM_ROUNDS + 1):
            # Send current global weights to clients
            self.comm.bcast(self.global_weights, root=0)

            # Receive updated weights from all clients
            client_weights = [
                deserialize_weights(self.comm.recv(source=i, tag=i))
                for i in range(1, num_clients + 1)
            ]

            # Receive system metrics from all clients
            metrics = [
                self.comm.recv(source=i, tag=i + 100)
                for i in range(1, num_clients + 1)
            ]
            for m in metrics:
                m['round'] = round_num
            all_client_metrics.extend(metrics)

            # Aggregate client weights
            self.global_weights = average_weights(client_weights)
            self.model.set_weights(self.global_weights)

            # Evaluate global model
            loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
            acc_list.append(acc)
            loss_list.append(loss)
            rounds.append(round_num)

            print(f"Round {round_num}: Loss={loss:.4f}, Accuracy={acc:.4f}")

        # Save metrics and plots
        save_client_metrics_csv(all_client_metrics, "client_system_metrics.csv")
        plot_training_curves(rounds, acc_list, loss_list, "federated_metrics.png")
