import argparse
import multiprocessing
import random
import signal
import time
from typing import List

import numpy as np
import requests

# --- Configuration ---
# Match these with your HDFS path /triton_models/recsys_model
MODEL_NAME = "recsys_model" 

# Adjust URL if your script runs outside the docker network (e.g., localhost:8000)
# URL = "http://localhost:8000/v2/models" 
# If running inside another container in the same network, use:
URL = "http://triton-server:8000/v2/models"

# --- Data Dimensions ---
# IMPORTANT: These must match the feature sizes in your META.json
USER_FEAT_DIM = 3   # e.g., if User x has 3 columns
QUIZ_FEAT_DIM = 3   # e.g., if Quiz x has 3 columns (or 600, change this!)

# Simulation Parameters
NUM_USERS = 100
NUM_QUIZZES = 50
NUM_MESSAGES = 200  # Number of edges in the graph structure
BATCH_SIZE = 10     # Number of pairs to predict scores for

def generate_random_input():
    """Generates inputs matching the recsys_model config."""
    
    # 1. User Features [Num_Users, User_Dim]
    x_user = np.random.randn(NUM_USERS, USER_FEAT_DIM).astype(np.float32)
    
    # 2. Quiz Features [Num_Quizzes, Quiz_Dim]
    x_quiz = np.random.randn(NUM_QUIZZES, QUIZ_FEAT_DIM).astype(np.float32)
    
    # 3. Edge Index (Graph Structure) [2, Num_Messages]
    # Row 0 = User Indices (0 to NUM_USERS-1)
    # Row 1 = Quiz Indices (0 to NUM_QUIZZES-1)
    edge_index = np.vstack([
        np.random.randint(0, NUM_USERS, NUM_MESSAGES),
        np.random.randint(0, NUM_QUIZZES, NUM_MESSAGES)
    ]).astype(np.int64)

    # 4. Edge Label Index (Target Pairs to Predict) [2, Batch_Size]
    edge_label_index = np.vstack([
        np.random.randint(0, NUM_USERS, BATCH_SIZE),
        np.random.randint(0, NUM_QUIZZES, BATCH_SIZE)
    ]).astype(np.int64)

    # Construct Triton JSON Payload
    payload = {
        "inputs": [
            {
                "name": "x_user",
                "shape": x_user.shape,
                "datatype": "FP32",
                "data": x_user.tolist()
            },
            {
                "name": "x_quiz",
                "shape": x_quiz.shape,
                "datatype": "FP32",
                "data": x_quiz.tolist()
            },
            {
                "name": "edge_index",
                "shape": edge_index.shape,
                "datatype": "INT64",
                "data": edge_index.flatten().tolist() # Flatten for JSON
            },
            {
                "name": "edge_label_index",
                "shape": edge_label_index.shape,
                "datatype": "INT64",
                "data": edge_label_index.flatten().tolist() # Flatten for JSON
            }
        ]
    }
    return payload

def worker_func(worker_id: int, model_name: str, server_url: str):
    """Worker loop sending requests."""
    print(f"[Worker {worker_id}] Starting...")
    session = requests.Session()
    # Ensure URL ends with /infer
    inference_url = f"{server_url}/{model_name}/infer"

    while True:
        try:
            payload = generate_random_input()
            start_time = time.time()
            
            response = session.post(inference_url, json=payload)
            response.raise_for_status()
            
            duration = time.time() - start_time
            result = response.json()
            
            # Extract output shape/data for verification
            outputs = result.get("outputs", [])
            output_data = outputs[0].get("data", []) if outputs else []

            print(
                f"[Worker {worker_id}] Success | "
                f"Time: {duration:.4f}s | "
                f"Predictions: {len(output_data)}"
            )

        except requests.exceptions.RequestException as e:
            print(f"[Worker {worker_id}] Request Error: {e}")
        except Exception as e:
            print(f"[Worker {worker_id}] General Error: {e}")

        # Sleep randomly between 0.2s and 1.0s
        time.sleep(random.uniform(0.2, 1.0))

def main():
    parser = argparse.ArgumentParser(description="Load testing client for RecSys Model.")
    parser.add_argument("-m", "--model", type=str, default=MODEL_NAME)
    parser.add_argument("-u", "--url", type=str, default=URL)
    parser.add_argument("-w", "--workers", type=int, default=2)
    args = parser.parse_args()

    print(f"--- Testing {args.model} at {args.url} ---")
    print(f"Workers: {args.workers}")
    print("------------------------------------------")

    processes = []

    def signal_handler(sig, frame):
        print("\nStopping workers...")
        for p in processes:
            p.terminate()
            p.join()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for i in range(args.workers):
        p = multiprocessing.Process(target=worker_func, args=(i, args.model, args.url))
        processes.append(p)
        p.start()
        time.sleep(0.1)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
