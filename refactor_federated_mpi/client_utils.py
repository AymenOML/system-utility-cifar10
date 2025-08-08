import psutil
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def collect_system_metrics(rank, round_num):
    """Collect CPU, RAM, and other system metrics."""
    process = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_info": process.memory_info().rss,
    }


def compute_per_round_metrics(start_snapshot, end_snapshot):
    """Compute differences in system metrics between start and end."""
    metrics = {}
    for key in start_snapshot:
        metrics[key] = end_snapshot[key] - start_snapshot[key]
    return metrics


def log_confusion_matrix(rank, round_num, model, x, y_oh):
    """Generate and save confusion matrix for this round."""
    y_true = np.argmax(y_oh, axis=1)
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Client {rank} Round {round_num}")
    plt.colorbar()
    plt.savefig(f"confusion_matrix_client{rank}_round{round_num}.png")
    plt.close()


def log_statistical_utility_tf(rank, round_num, x, y_oh, model):
    """Log accuracy, loss, and other statistical utility metrics."""
    loss, acc = model.evaluate(x, y_oh, verbose=0)
    with open(f"statistical_utility_client{rank}.log", "a") as f:
        f.write(f"Round {round_num}: loss={loss}, acc={acc}\n")
