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

def collect_system_metrics(rank, round_num):
    import time
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

def run_client(comm, rank):
    print(f"    [Client {rank}] Initializing...", flush=True)
    print(f"    [Client {rank}] Starting on host: {os.uname().nodename}", flush=True)


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

    model = build_cnn_model()
    
    ### Change for number of rounds
    for round_num in range(1, NUM_ROUNDS + 1):

        print(f"    [Client {rank}] Round {round_num} - Waiting for global weights...", flush=True)
        global_weights = comm.bcast(None, root=0)
        model.set_weights(global_weights)

        print(f"    [Client {rank}] Round {round_num} - Training on local data...", flush=True)
        model.fit(x_client, y_client, epochs=1, batch_size=32, verbose=0)

        metrics = collect_system_metrics(rank, round_num)
        print(f"    [Client {rank}] System Stats: {metrics}", flush=True)

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

