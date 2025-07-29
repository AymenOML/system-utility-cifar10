import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import psutil
import GPUtil
import tensorflow as tf
from data_loader import load_and_preprocess_data
from model import build_cnn_model
from mpi_utils import serialize_weights, deserialize_weights
from tensorflow.keras.utils import to_categorical
from mpi4py import MPI
from config.config import NUM_ROUNDS
from datetime import datetime
import csv

def evaluate_keras_model(model, x, y):
    results = model.evaluate(x, y, verbose=0)
    loss = results[0]
    accuracy = results[1] * 100  # percentage
    return loss, accuracy

def compute_data_variance_tf(x):
    x_flat = x.reshape(x.shape[0], -1)
    return float(np.var(x_flat))

def log_statistical_utility_tf(rank, round_num, x, y, model):
    csv_path = "all_clients_stats.csv"
    os.makedirs("client_logs", exist_ok=True)  # optional folder safety

    data_size = x.shape[0]
    data_variance = compute_data_variance_tf(x)
    local_loss, local_accuracy = evaluate_keras_model(model, x, y)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'client_rank', 'round', 'data_size',
                'data_variance', 'local_loss', 'local_accuracy'
            ])
        writer.writerow([
            rank, round_num, data_size,
            data_variance, local_loss, local_accuracy
        ])

    return {
        "client_rank": rank,
        "round": round_num,
        "data_size": data_size,
        "data_variance": data_variance,
        "local_loss": local_loss,
        "local_accuracy": local_accuracy
    }



def collect_system_metrics(rank, round_num):
    # This gets the current Python process
    p = psutil.Process(os.getpid())
    
    cpu_times = psutil.cpu_times()
    cpu_freq = psutil.cpu_freq().current
    # Get this process's memory usage (RSS: Resident Set Size)
    ram_used = p.memory_info().rss / 1e6
    net = psutil.net_io_counters()
    net_sent = net.bytes_sent
    net_recv = net.bytes_recv

    try:
        gpu = GPUtil.getGPUs()[0]
        gpu_mem_used = gpu.memoryUsed
        gpu_load = gpu.load
    except:
        gpu_mem_used = None
        gpu_load = None

    return {
        "client_rank": rank,
        "round": round_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_time": cpu_times.user + cpu_times.system,
        "cpu_freq_mhz": cpu_freq,
        "ram_used_mb": ram_used,
        "net_sent_bytes": net_sent,
        "net_recv_bytes": net_recv,
        "gpu_mem_used_mb": gpu_mem_used,
        "gpu_load": gpu_load
    }

def compute_per_round_metrics(start_snapshot, end_snapshot):
    return {
        "cpu_time": end_snapshot["cpu_time"] - start_snapshot["cpu_time"],
        "ram_used_mb": end_snapshot["ram_used_mb"],  # typically just take the latest
        "net_sent_bytes": end_snapshot["net_sent_bytes"] - start_snapshot["net_sent_bytes"],
        "net_recv_bytes": end_snapshot["net_recv_bytes"] - start_snapshot["net_recv_bytes"],
        "gpu_mem_used_mb": end_snapshot["gpu_mem_used_mb"],
        "gpu_load": end_snapshot["gpu_load"]
    }


def run_client(comm, rank):
    print(f"    [Client {rank}] Initializing...", flush=True)

    print(f"    [Client {rank}] Starting on host: {os.uname().nodename}", flush=True)

    #print(f"    [Client {rank}] Starting on host: {platform.node()}", flush=True)


    # 1. Load data in memory-map mode WITHOUT preprocessing
    (x_train_mmap, y_train_mmap), _ = load_and_preprocess_data(mmap_mode='r', preprocess=False)

    # 2. Calculate the data slice for this client
    num_clients = comm.Get_size() - 1
    total_samples = x_train_mmap.shape[0]
    samples_per_client = total_samples // num_clients
    start = (rank - 1) * samples_per_client
    end = start + samples_per_client

    # 3. Slice the data and load ONLY the slice into memory
    x_client = np.array(x_train_mmap[start:end])
    y_client = np.array(y_train_mmap[start:end])

    # 4. Preprocess ONLY the client's local data
    x_client = x_client.astype('float32') / 255.0
    y_client = to_categorical(y_client, 10)

    ### Change for number of rounds
    for round_num in range(1, NUM_ROUNDS + 1):

        model = build_cnn_model()

        print(f"    [Client {rank}] Round {round_num} - Waiting for global weights...", flush=True)
        global_weights = comm.bcast(None, root=0)
        model.set_weights(global_weights)

        print(f"    [Client {rank}] Round {round_num} - Training on local data...", flush=True)
        start_snapshot = collect_system_metrics(rank, round_num)
        model.fit(x_client, y_client, epochs=1, batch_size=32, verbose=0)

        end_snapshot = collect_system_metrics(rank, round_num)
        metrics = compute_per_round_metrics(start_snapshot, end_snapshot)
        metrics.update({
            "client_rank": rank,
            "round": round_num,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"    [Client {rank}] System Stats: {metrics}", flush=True)


        stats = log_statistical_utility_tf(rank, round_num, x_client, y_client, model)
        print(f"    [Client {rank}] Statistical Utility: {stats}", flush=True)


        # metrics = collect_system_metrics(rank, round_num)
        # print(f"    [Client {rank}] System Stats: {metrics}", flush=True)

        updated_weights = serialize_weights(model.get_weights())
        print(f"    [Client {rank}] Round {round_num} - Sending updated weights to server...", flush=True)

        try:
            comm.send(updated_weights, dest=0, tag=rank)
        except Exception as e:
            print(f"[Client {rank}] Failed to send: {e}", flush=True)

        try:
            comm.send(metrics, dest=0, tag=rank+100)
        except Exception as e:
            print(f"[Client {rank}] Failed to send: {e}", flush=True)

    print(f"    [Client {rank}] Training complete. Waiting for others...", flush=True)
    print(f"    [Client {rank}] Exiting.", flush=True)
    return

