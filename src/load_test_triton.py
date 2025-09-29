import argparse
import json
import multiprocessing
import random
import signal
import time
from typing import List

import numpy as np
import requests

# --- Configuration ---
# These values should match the 'config.pbtxt' you generated.
MODEL_NAME = "graphsage_model"
URL = "http://triton-server:8000/v2/models"

# Input configuration
INPUT_NAMES = ["x__0", "edge_index__1"]
INPUT_DTYPES = ["FP32", "INT64"]

# The script will generate random data with these shapes.
# Adjust them to simulate your expected workload.
NUM_NODES = 50  # Number of nodes in the graph
NUM_FEATURES = 600 # Must match your model's in_channels
NUM_EDGES = 100 # Number of edges in the graph


def generate_random_input():
    """Generates a single random input payload for the GraphSAGE model."""
    node_features = np.random.rand(NUM_NODES, NUM_FEATURES).astype(np.float32)
    edge_index = np.random.randint(0, NUM_NODES, size=(2, NUM_EDGES)).astype(np.int64)

    payload = {
        "inputs": [
            {
                "name": INPUT_NAMES[0],
                "shape": node_features.shape,
                "datatype": INPUT_DTYPES[0],
                "data": node_features.tolist()
            },
            {
                "name": INPUT_NAMES[1],
                "shape": edge_index.shape,
                "datatype": INPUT_DTYPES[1],
                # Data must be flattened for the JSON payload
                "data": edge_index.flatten().tolist()
            }
        ]
    }
    return payload


def worker_func(worker_id: int, model_name: str, server_url: str):
    """
    The main function for each worker process.
    It continuously sends requests to the Triton server.
    """
    print(f"[Worker {worker_id}] Starting...")
    session = requests.Session()
    inference_url = f"{server_url}/{model_name}/infer"

    while True:
        try:
            payload = generate_random_input()
            start_time = time.time()
            response = session.post(inference_url, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            duration = time.time() - start_time

            print(
                f"[Worker {worker_id}] Success | "
                f"Status: {response.status_code} | "
                f"Latency: {duration:.4f}s"
            )

        except requests.exceptions.RequestException as e:
            print(f"[Worker {worker_id}] An error occurred: {e}")

        # Sleep for a short, random interval to stagger requests
        time.sleep(random.uniform(0.2, 1.0))


def main():
    """Parses arguments and launches worker processes."""
    parser = argparse.ArgumentParser(
        description="Load testing client for Triton Inference Server."
    )
    parser.add_argument(
        "-m", "--model", type=str, default=MODEL_NAME,
        help="Name of the model to send requests to."
    )
    parser.add_argument(
        "-u", "--url", type=str, default=URL,
        help="Base URL of the Triton server's v2 API."
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4,
        help="Number of concurrent worker processes to run."
    )
    args = parser.parse_args()

    print("--- Triton Load Tester ---")
    print(f"Server URL: {args.url}")
    print(f"Model Name: {args.model}")
    print(f"Concurrent Workers: {args.workers}")
    print("--------------------------")
    print("Press Ctrl+C to stop.")

    processes: List[multiprocessing.Process] = []

    # Gracefully handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nShutting down workers...")
        for p in processes:
            p.terminate()
            p.join()
        print("All workers stopped.")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for i in range(args.workers):
        process = multiprocessing.Process(
            target=worker_func, args=(i, args.model, args.url)
        )
        processes.append(process)
        process.start()
        # Stagger the start of workers slightly
        time.sleep(0.1)

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
