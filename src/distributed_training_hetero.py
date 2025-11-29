import argparse
import json
import os
import os.path as osp
import re
import time
import subprocess
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.distributed import (
    DistContext,
    DistLinkNeighborLoader, # <--- CHANGED: For Link Prediction
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.nn import GraphSAGE, to_hetero
from typing import Union
from torch_geometric.typing import NodeType, EdgeType

# --- START MONKEY PATCH ---
def safe_set_global_id_to_index(self, group_name: Union[NodeType, EdgeType]):
    global_id = self.get_global_id(group_name)

    if global_id is None:
        return

    # PATCH: Check for empty tensor before calling .max()
    if global_id.numel() == 0:
        # Create an empty index map if there are no nodes/edges
        self._global_id_to_index[group_name] = global_id.new_full((0,), fill_value=-1)
        return

    # Original Logic
    global_id_to_index = global_id.new_full((int(global_id.max()) + 1, ),
                                            fill_value=-1)
    global_id_to_index[global_id] = torch.arange(global_id.numel())
    self._global_id_to_index[group_name] = global_id_to_index

# Apply the patch to the class
LocalFeatureStore._set_global_id_to_index = safe_set_global_id_to_index
print("Monkey patch applied: LocalFeatureStore handles empty partitions safely.")
# --- END MONKEY PATCH ---

# =========================================================================
# 1. DEFINE MODEL (Encoder + Dot Product Decoder)
# =========================================================================
class RecommenderModel(torch.nn.Module):
    def __init__(self, hetero_gnn):
        super().__init__()
        self.encoder = hetero_gnn

    def forward(self, x_dict, edge_index_dict, edge_label_index, target_edge_type):
        # 1. Get Node Embeddings for all nodes in the batch
        z_dict = self.encoder(x_dict, edge_index_dict)

        # 2. Decode: Calculate similarity for the specific edges we are predicting
        # edge_label_index contains pairs of (User_ID, Quiz_ID)
        src_type, _, dst_type = target_edge_type
        
        # Get indices
        row, col = edge_label_index
        
        # Get embeddings for source (User) and destination (Quiz)
        z_src = z_dict[src_type][row]
        z_dst = z_dict[dst_type][col]

        # 3. Dot Product Similarity
        return (z_src * z_dst).sum(dim=-1)

# =========================================================================
# 2. TRAIN & TEST LOOPS
# =========================================================================
def train(model, loader, optimizer, dist_context, device, epoch, target_edge_type):
    model.train()
    total_loss = 0
    total_examples = 0
    
    if dist_context.rank == 0:
        loader = tqdm(loader, desc=f'Train Epoch {epoch}')

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 1. POSITIVE SAMPLES (Actual Edges)
        # ----------------------------------
        # 'edge_label_index' contains the real User-Quiz pairs
        pos_edge_index = batch[target_edge_type].edge_label_index
        
        # Predict score for positives (should be close to 1)
        pos_pred = model(
            batch.x_dict, 
            batch.edge_index_dict, 
            pos_edge_index,
            target_edge_type
        )

        # 2. NEGATIVE SAMPLES (In-Batch Shuffling)
        # ----------------------------------------
        # We create negative pairs by keeping Users fixed and shuffling Quizzes.
        # This pairs User A with a random Quiz B from the same batch.
        # This is efficient because Quiz B's features are already in memory.
        
        # Clone the index to create negatives
        neg_edge_index = pos_edge_index.clone()
        # Shuffle the destination row (Quizzes)
        neg_edge_index[1] = neg_edge_index[1][torch.randperm(neg_edge_index.size(1))]
        
        # Predict score for negatives (should be close to 0)
        neg_pred = model(
            batch.x_dict, 
            batch.edge_index_dict, 
            neg_edge_index,
            target_edge_type
        )

        # 3. COMPUTE LOSS
        # ---------------
        # Concatenate predictions and create labels
        all_pred = torch.cat([pos_pred, neg_pred])
        all_labels = torch.cat([
            torch.ones(pos_pred.size(0), device=device), # Real edges = 1
            torch.zeros(neg_pred.size(0), device=device) # Fake edges = 0
        ])
        
        loss = F.binary_cross_entropy_with_logits(all_pred, all_labels)
        
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * all_pred.numel()
        total_examples += all_pred.numel()

    return total_loss / total_examples

@torch.no_grad()
def test(model, loader, dist_context, device, target_edge_type):
    model.eval()
    total_correct = 0
    total_examples = 0

    if dist_context.rank == 0:
        loader = tqdm(loader, desc='Test')

    for batch in loader:
        batch = batch.to(device)
        
        # 1. Positives
        pos_index = batch[target_edge_type].edge_label_index
        pos_pred = model(batch.x_dict, batch.edge_index_dict, pos_index, target_edge_type)
        
        # 2. Negatives (Random shuffle)
        neg_index = pos_index.clone()
        neg_index[1] = neg_index[1][torch.randperm(neg_index.size(1))]
        neg_pred = model(batch.x_dict, batch.edge_index_dict, neg_index, target_edge_type)
        
        # 3. Accuracy
        # Sigmoid > 0.5 means "Connected"
        pos_acc = (torch.sigmoid(pos_pred) > 0.5).sum().item()
        neg_acc = (torch.sigmoid(neg_pred) < 0.5).sum().item() # Correct if < 0.5
        
        total_correct += (pos_acc + neg_acc)
        total_examples += (pos_pred.numel() + neg_pred.numel())

    return total_correct / total_examples

# =========================================================================
# 3. MAIN RUNNER
# =========================================================================
def run_proc(
    node_rank, num_nodes, dataset_root_dir, master_addr,
    ddp_port, train_loader_port, test_loader_port,
    num_epochs, batch_size, num_neighbors, num_workers, model_save_path
):
    # --- 1. Load Data Stores ---
    print(f'[Rank {node_rank}] Loading partitions...')
    
    # Load META.json
    with open(osp.join(dataset_root_dir, 'META.json'), 'r') as f:
        meta = json.load(f)

    # Load Stores
    graph_store = LocalGraphStore.from_partition(dataset_root_dir, pid=node_rank)
    feature_store = LocalFeatureStore.from_partition(dataset_root_dir, pid=node_rank)
    
    # Define the Hetero Graph Metadata needed for to_hetero
    node_types = meta['node_types']
    # Convert list of lists to list of tuples for PyG
    edge_types = [tuple(x) for x in meta['edge_types']] 
    metadata = (node_types, edge_types)

    # --- 2. Setup Distributed Context ---
    dist_context = DistContext(
        world_size=num_nodes,
        rank=node_rank,
        global_world_size=num_nodes,
        global_rank=node_rank,
        group_name='pyg-dist'
    )
    
    torch.distributed.init_process_group(
        backend='gloo',
        rank=dist_context.rank,
        world_size=dist_context.world_size,
        init_method=f'tcp://{master_addr}:{ddp_port}',
    )
    
    # --- 3. Define Training Target ---
    # We want to predict which User TOOK which Quiz
    target_edge_type = ("User", "TOOK", "Quiz")
    
    # Get total number of edges for this type (to split train/test)
    # Note: In distributed settings, obtaining global edge counts requires synchronization
    # or reading from META. For simplicity, we assume we want to train on ALL edges present
    # in the graph store partitions.
    # In a real scenario, you might split edges by timestamp.
    
    # We pass 'None' to input_nodes, which tells DistLinkNeighborLoader 
    # to iterate over ALL edges of 'edge_type' distributed across partitions.
    
    # --- 4. Loaders (Link Prediction) ---
    neighbor_sizes = [int(i) for i in num_neighbors.split(',')]
    
    print(f'[Rank {node_rank}] Creating DistLinkNeighborLoader...')
    
    # We need to provide "edge_label_index" (the edges to train on).
    # Since we want to train on ALL edges in this partition, we can retrieve them 
    # from the local graph store.
    
    # 1. Get all local edges for ("User", "TOOK", "Quiz")
    # This retrieves the COOrdinate format (row, col) from the local store
    edge_attr = graph_store.get_all_edge_attrs()[0] # Assuming "TOOK" is the first/primary edge
    # Ideally, search for the specific edge type:
    target_attr = None
    for attr in graph_store.get_all_edge_attrs():
        if attr.edge_type == target_edge_type:
            target_attr = attr
            break
            
    if target_attr:
        # Get the edge_index (row, col)
        # We need to access the private attribute or use the API if available.
        # LocalGraphStore keeps it in _edge_index
        local_edge_index = graph_store._get_edge_index(target_attr)
    else:
        # Fallback if specific type not found (shouldn't happen if setup is correct)
        local_edge_index = torch.empty((2, 0), dtype=torch.long)

    train_loader = DistLinkNeighborLoader(
        data=(feature_store, graph_store),
        edge_label_index=(target_edge_type, local_edge_index), # Pass local edges here
        # input_nodes=None, # Removed this, we use edge_label_index instead
        neg_sampling_ratio=0.0, 
        num_neighbors=neighbor_sizes,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=train_loader_port,
        current_ctx=dist_context, # <--- FIXED NAME (was dist_context=...)
    )

    test_loader = DistLinkNeighborLoader(
        data=(feature_store, graph_store),
        edge_label_index=(target_edge_type, local_edge_index), # Using same edges for demo
        neg_sampling_ratio=0.0,
        num_neighbors=neighbor_sizes,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=test_loader_port,
        current_ctx=dist_context, # <--- FIXED NAME
    ) 

    # --- 5. Model Setup ---
    device = torch.device('cpu')
    
    # Calculate input dimension from User/Quiz features
    # Assuming standard feature size 3 from your Spark Job
    in_channels = -1
    
    base_model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=64,
        num_layers=len(neighbor_sizes),
        out_channels=64
    )
    
    # Convert to Hetero
    hetero_gnn = to_hetero(base_model, metadata, aggr='sum').to(device)
    
    # Wrap in Recommender (Decoder)
    model = RecommenderModel(hetero_gnn).to(device)
    
    # ==================== FIX START: INITIALIZE LAZY MODULES ====================
    # DDP needs parameters to exist, but they are currently empty (Lazy).
    # We feed one fake batch through the model to force weight initialization.
    print(f'[Rank {node_rank}] Running dummy forward pass to initialize lazy modules...')
    
    with torch.no_grad():
        # A. Create Dummy Node Features
        # Read dimensions from META.json to ensure shapes match
        dummy_x_dict = {}
        for ntype, schema in meta['node_feat_schema'].items():
            # schema['x'] is a list like [3]. Get the first element.
            feat_dim = schema['x'][0]
            # Create random tensor: [BatchSize=2, FeatDim]
            dummy_x_dict[ntype] = torch.randn(2, feat_dim, device=device)
            
        # B. Create Dummy Edge Indices (Empty is fine for initialization)
        dummy_edge_index_dict = {}
        for etype_list in meta['edge_types']:
            etype = tuple(etype_list) # Convert list to tuple
            # Empty edge index: [2, 0]
            dummy_edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long, device=device)
            
        # C. Define a dummy target for the Recommender
        # Use the "target_edge_type" defined later in your code
        init_target_edge = ("User", "TOOK", "Quiz") 
        
        # D. Dummy Label Index (Indices of User and Quiz to predict)
        # Shape [2, 2]: predicting for 2 pairs
        dummy_label_index = torch.zeros((2, 2), dtype=torch.long, device=device)
        
        # E. Run Forward (Results are ignored, side-effect initializes weights)
        model(dummy_x_dict, dummy_edge_index_dict, dummy_label_index, init_target_edge)
        
    print(f'[Rank {node_rank}] Initialization complete. Parameters materialized.')
    # ==================== FIX END ====================

    # Wrap in DDP
    model = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- 6. Training Loop ---
    print(f'[Rank {node_rank}] Starting Training...')
    
    for epoch in range(1, num_epochs + 1):
        loss = train(model, train_loader, optimizer, dist_context, device, epoch, target_edge_type)
        
        if node_rank == 0:
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
            
        if epoch % 5 == 0:
            acc = test(model, test_loader, dist_context, device, target_edge_type)
            if node_rank == 0:
                print(f'Epoch {epoch:02d}, Test Acc: {acc:.4f}')

        torch.distributed.barrier()

    # --- 7. Save Model ---
    if node_rank == 0:
        # --- PART 1: Save Locally (Your existing logic) ---
        os.makedirs(model_save_path, exist_ok=True)

        # 1. Find the next epoch number
        pattern = re.compile(r'model_epoch_(\d+)\.pt')
        existing_epochs = []
        for filename in os.listdir(model_save_path):
            match = pattern.match(filename)
            if match:
                existing_epochs.append(int(match.group(1)))
        
        next_epoch = max(existing_epochs) + 1 if existing_epochs else 1
            
        # 2. Save local file (e.g., ./saved_models/model_epoch_3.pt)
        local_filename = f'model_epoch_{next_epoch}.pt'
        save_full_path = osp.join(model_save_path, local_filename)
        
        torch.save(model.module.state_dict(), save_full_path)
        print(f"Model saved locally to {save_full_path}")

        # --- PART 2: Push to HDFS for Triton ---
        
        # Configuration
        hdfs_root = "/triton_models"      # Must match HDFS_MODEL_REPO in your syncer
        triton_model_name = "recsys_model" # The name of the model in Triton
        
        # Triton expects: /triton_models/recsys_model/1/model.pt
        # We use 'next_epoch' as the version number.
        hdfs_version_dir = f"{hdfs_root}/{triton_model_name}/{next_epoch}"
        hdfs_target_file = f"{hdfs_version_dir}/model.pt" # MUST be named model.pt for PyTorch backend

        print(f"Pushing to HDFS: {hdfs_target_file}...")

        try:
            # 1. Create the specific version directory in HDFS
            subprocess.run(
                ["hdfs", "dfs", "-mkdir", "-p", hdfs_version_dir], 
                check=True
            )

            # 2. Upload the local file to HDFS and rename it to 'model.pt'
            subprocess.run(
                ["hdfs", "dfs", "-put", "-f", save_full_path, hdfs_target_file], 
                check=True
            )
            print("Successfully pushed model to HDFS.")
            
        except subprocess.CalledProcessError as e:
            print(f"Error pushing to HDFS: {e}")

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    # Environment variables set by torchrun or kubernetes
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', type=str, default='/tmp/pyg_recsys_hetero')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_neighbors', type=str, default='5,3')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0, # <--- CHANGE THIS TO 0 for local testing
        help='Number of sampler sub-processes',
    )
    args = parser.parse_args()

    run_proc(
        node_rank=rank,
        num_nodes=world_size,
        dataset_root_dir=args.dataset_root_dir,
        master_addr=master_addr,
        ddp_port=11111,
        train_loader_port=11112,
        test_loader_port=11113,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbors,
        num_workers=args.num_workers,
        model_save_path=args.model_save_path
    )
