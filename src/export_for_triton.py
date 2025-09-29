import argparse
import json
import os
import subprocess
import torch
from torch_geometric.nn import GraphSAGE, to_hetero

def get_next_version(hdfs_model_path):
    """Finds the next version number for a model on HDFS."""
    try:
        # List directories in the model path
        result = subprocess.run(
            ['hdfs', 'dfs', '-ls', hdfs_model_path],
            capture_output=True, text=True, check=True
        )
        existing_versions = [
            int(os.path.basename(line.split()[-1]))
            for line in result.stdout.strip().split('\n')[1:] # Skip the first 'Found X items' line
            if os.path.basename(line.split()[-1]).isdigit()
        ]
        return max(existing_versions) + 1 if existing_versions else 1
    except subprocess.CalledProcessError:
        # Directory likely doesn't exist yet
        return 1

def create_config_pbtxt(model_name, dataset_meta, hdfs_version_path):
    """Generates the config.pbtxt and uploads it to HDFS."""
    is_hetero = 'node_map' in dataset_meta
    target_node_type = dataset_meta['node_types'][0] if is_hetero else None

    # For this example, we'll create a simplified config for a homogeneous graph
    # as heterogeneous tracing is complex. A production setup might require
    # a model wrapper to flatten the dictionary inputs.
    if is_hetero:
        print("Warning: Heterogeneous model detected. Generating a simplified config.pbtxt. "
              "You may need a custom model wrapper for production inference.")
        # Simplified: assuming single feature tensor for the target node type
        input_dims = f'dims: [ -1, {dataset_meta["node_feat_schema"][target_node_type]["__feat__"]} ]'
    else:
        input_dims = f'dims: [ -1, {dataset_meta["node_feat_schema"]["__feat__"]} ]'

    config_content = f"""
name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 0

input [
  {{
    name: "x__0"
    data_type: TYPE_FP32
    {input_dims}
  }},
  {{
    name: "edge_index__1"
    data_type: TYPE_INT64
    dims: [ 2, -1 ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ -1, {dataset_meta["num_classes"]} ]
  }}
]
"""
    config_path = "/tmp/config.pbtxt"
    with open(config_path, 'w') as f:
        f.write(config_content)

    # Upload to the model's root directory (not the versioned one)
    subprocess.run(['hdfs', 'dfs', '-put', '-f', config_path, f"{os.path.dirname(hdfs_version_path)}/"], check=True)
    print(f"Uploaded config.pbtxt to {os.path.dirname(hdfs_version_path)}")

def main():
    parser = argparse.ArgumentParser(description="Export PyG model for Triton")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--dataset_meta_path', type=str, required=True)
    parser.add_argument('--hdfs_repo_path', type=str, required=True)
    args = parser.parse_args()

    # Load dataset metadata
    with open(args.dataset_meta_path, 'r') as f:
        meta = json.load(f)

    # --- Model Definition (must match training script) ---
    is_hetero = 'node_map' in meta
    if is_hetero:
        target_node_type = meta['node_types'][0]
        in_channels = meta['node_feat_schema'][target_node_type]['__feat__']
    else:
        in_channels = meta['node_feat_schema']['__feat__']
    out_channels = meta['num_classes']
    num_layers = 2 # Assuming 2 layers as per your DAG's num_neighbors '10,5'

    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=256,
        num_layers=num_layers,
        out_channels=out_channels,
    )
    if is_hetero:
        model_meta = (meta['node_types'], meta['edge_types'])
        model = to_hetero(model, model_meta, aggr='sum')

    # Load the trained weights
    # The checkpoint was saved from a DDP model, so we need to handle the 'module.' prefix
    state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # --- Trace the Model for Triton ---
    # For tracing, we need example inputs.
    # A simplified example for a homogeneous graph:
    x = torch.randn(10, in_channels) # 10 nodes, in_channels features
    edge_index = torch.randint(0, 10, (2, 20)) # 20 edges
    
    # NOTE: Tracing heterogeneous models is complex. This example traces the base model.
    # For a real heterogeneous setup, you'd create a wrapper class that accepts
    # tensors and reconstructs the x_dict/edge_index_dict inside its forward method.
    traced_model = torch.jit.trace(model.to('cpu'), (x, edge_index))
    
    traced_model_path = "/tmp/model.pt"
    traced_model.save(traced_model_path)
    print(f"Model traced and saved to {traced_model_path}")

    # --- Upload to HDFS ---
    hdfs_model_path = os.path.join(args.hdfs_repo_path, args.model_name)
    version = get_next_version(hdfs_model_path)
    hdfs_version_path = os.path.join(hdfs_model_path, str(version))

    subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', hdfs_version_path], check=True)
    subprocess.run(['hdfs', 'dfs', '-put', '-f', traced_model_path, f"{hdfs_version_path}/model.pt"], check=True)
    print(f"Uploaded model.pt to {hdfs_version_path}/")

    # --- Create and Upload config.pbtxt ---
    create_config_pbtxt(args.model_name, meta, hdfs_version_path)

    print(f"Successfully exported model '{args.model_name}' version {version} to HDFS.")

if __name__ == '__main__':
    main()
