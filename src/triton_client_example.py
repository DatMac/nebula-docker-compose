import requests
import json
import numpy as np
import sys

# --- Configuration ---
# IMPORTANT: Use the Docker service name 'triton-server' as the hostname.
# The default HTTP port for Triton is 8000.
TRITON_HOSTNAME = "triton-server"
TRITON_HTTP_PORT = 8000
MODEL_NAME = "graphsage_model"

# Model input dimensions from your config.pbtxt
# This MUST match the model you are serving.
NUM_NODES = 10         # Number of nodes in our sample graph
IN_CHANNELS = 600      # Feature dimension per node
NUM_CLASSES = 2        # Output dimension per node

def main():
    """
    Creates sample graph data and sends an inference request to the
    Triton server running within the same Docker network.
    """
    print(f"--- Preparing inference request for model '{MODEL_NAME}' ---")

    # 1. Generate sample data that matches the model's expected input shapes.
    #    - x__0: Node feature matrix of shape [num_nodes, in_channels]
    #    - edge_index__1: Edge index matrix of shape [2, num_edges]
    node_features = np.random.rand(NUM_NODES, IN_CHANNELS).astype(np.float32)
    # Create 20 random edges for our sample graph
    edge_index = np.random.randint(0, NUM_NODES, size=(2, 20)).astype(np.int64)

    # 2. Construct the payload in the format Triton expects (Inference Protocol V2).
    payload = {
        "inputs": [
            {
                "name": "x__0",
                "shape": node_features.shape,
                "datatype": "FP32",
                "data": node_features.tolist()
            },
            {
                "name": "edge_index__1",
                "shape": edge_index.shape,
                "datatype": "INT64",
                # The data needs to be flattened for the JSON payload.
                "data": edge_index.flatten().tolist()
            }
        ]
    }

    # 3. Send the request to Triton's inference endpoint.
    url = f"http://{TRITON_HOSTNAME}:{TRITON_HTTP_PORT}/v2/models/{MODEL_NAME}/infer"
    print(f"Sending request to: {url}")

    try:
        response = requests.post(url, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Could not connect to Triton Inference Server.")
        print(f"  - Is the 'triton-server' container running?")
        print(f"  - Is the model '{MODEL_NAME}' loaded successfully? Check Triton's logs.")
        print(f"  - Details: {e}")
        sys.exit(1)

    # 4. Process the response.
    print("\n--- Inference successful! ---")
    result = response.json()
    
    # Extract the output tensor from the response
    output_tensor = result['outputs'][0]
    output_data = np.array(output_tensor['data'])
    output_shape = output_tensor['shape']

    print(f"Model Name: {result['model_name']}")
    print(f"Model Version: {result['model_version']}")
    print(f"Output Name: {output_tensor['name']}")
    print(f"Output Shape: {output_shape}")
    print(f"Output Datatype: {output_tensor['datatype']}")
    
    # Verify the output shape matches expectations: [num_nodes, num_classes]
    assert output_shape == [NUM_NODES, NUM_CLASSES]
    print("\nOutput shape is correct.")
    
    # Reshape the flattened output data back to its 2D shape
    final_output = output_data.reshape(output_shape)
    print("\nFirst 5 predictions (logits):")
    print(final_output[:5])


if __name__ == '__main__':
    main()
