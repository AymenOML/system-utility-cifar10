import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def average_weights(weight_list):
    """
    Average the weights from multiple clients.
    :param weight_list: list of weight lists (one per client)
    """
    return [np.mean(w, axis=0) for w in zip(*weight_list)]


def save_client_metrics_csv(all_client_metrics, path="client_system_metrics.csv"):
    """
    Save collected client system metrics to a CSV file.
    """
    df = pd.DataFrame(all_client_metrics)
    df.to_csv(path, index=False)
    print(f"Client metrics saved to {path}")


def plot_training_curves(rounds, acc_list, loss_list, out="federated_metrics.png"):
    """
    Plot accuracy and loss curves over training rounds.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rounds, acc_list, label='Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, loss_list, label='Loss', color='red')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Training curves saved to {out}")
