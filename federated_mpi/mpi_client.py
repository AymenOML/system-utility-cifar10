import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import psutil
import GPUtil
from data_loader import load_and_preprocess_data
from model import build_cnn_model
from mpi_utils import serialize_weights, deserialize_weights
from mpi4py import MPI
from config.config import NUM_ROUNDS
from datetime import datetime


def collect_system_metrics(rank, round_num):
    import time
    cpu_times = psutil.cpu_times()
    cpu_freq = psutil.cpu_freq().current
    ram_used = psutil.virtual_memory().used / 1e6
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
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"    [Client {rank}] Round {round_num} - Waiting for global weights...", flush=True)
        global_weights = comm.bcast(None, root=0)
        model.set_weights(global_weights)

        print(f"    [Client {rank}] Round {round_num} - Training on local data...", flush=True)
        model.fit(x_client, y_client, epochs=1, batch_size=32, verbose=0)

        metrics = collect_system_metrics(rank, round_num)
        print(f"    [Client {rank}] System Stats: {metrics}", flush=True)
        comm.send(metrics, dest=0, tag=rank + 100)



        updated_weights = serialize_weights(model.get_weights())
        print(f"    [Client {rank}] Round {round_num} - Sending updated weights to server...", flush=True)
        comm.send(updated_weights, dest=0, tag=rank)


    print(f"    [Client {rank}] Training complete. Waiting for others...", flush=True)
    print(f"    [Client {rank}] Exiting.", flush=True)
    return

