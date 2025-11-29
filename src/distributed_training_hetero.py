import argparse
import json
import os
import os.path as osp
import re
import time
import subprocess
from contextlib import nullcontext
from typing import Union, Tuple, Dict

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.distributed import (
    DistContext,
    DistLinkNeighborLoader,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.nn import GraphSAGE, to_hetero
from torch_geometric.typing import NodeType, EdgeType

# --- START MONKEY PATCH ---
# This fixes a known issue in PyG Distributed where empty partitions cause crashes
def safe_set_global_id_to_index(self, group_name: Union[NodeType, EdgeType]):
    global_id = self.get_global_id(group_name)

    if global_id is None:
        return

    if global_id.numel() == 0:
        self._global_id_to_index[group_name] = global_id.new_full((0,), fill_value=-1)
        return

    global_id_to_index = global_id.new_full((int(global_id.max()) + 1, ),
                                            fill_value=-1)
    global_id_to_index[global_id] = torch.arange(global_id.numel())
    self._global_id_to_index[group_name] = global_id_to_index

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

    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], 
                edge_label_index: torch.Tensor, 
                target_edge_type: Tuple[str, str, str]):
        
        # 1. Get Node Embeddings
        z_dict = self.encoder(x_dict, edge_index_dict)

        # 2. Decode
        src_type, _, dst_type = target_edge_type
        # row, col = edge_label_index
        row = edge_label_index[0]
        col = edge_label_index[1]
        
        z_src = z_dict[src_type][row]
        z_dst = z_dict[dst_type][col]

        # 3. Dot Product Similarity
        return (z_src * z_dst).sum(dim=-1)

# =========================================================================
# 2. INFERENCE WRAPPER (Required for Triton/JIT)
# =========================================================================
class InferenceWrapper(torch.nn.Module):
    """
    Wraps the RecommenderModel to accept flat Tensors instead of Dictionaries.
    Now handles filling in missing edge types with empty tensors to satisfy to_hetero.
    """
    def __init__(self, model, all_edge_types):
        super().__init__()
        self.model = model
        # We hardcode the edge type here because JIT struggles with passing Tuples as args from Triton
        self.target_edge_type = ("User", "TOOK", "Quiz") 
        # Store all known edge types to ensure dictionary is complete
        self.all_edge_types = all_edge_types

    def forward(self, x_user: torch.Tensor, x_quiz: torch.Tensor, 
                edge_index: torch.Tensor, edge_label_index: torch.Tensor):
        # 1. Reconstruct x_dict
        x_dict = {
            "User": x_user,
            "Quiz": x_quiz
        }
        
        # 2. Reconstruct edge_index_dict
        # logic: Use the input 'edge_index' for the target type, 
        # and create empty tensors [2, 0] for all other types defined in metadata.
        edge_index_dict = {}
        
        # We perform a loop here. Since self.all_edge_types is constant, 
        # torch.jit.trace will unroll this loop into static graph nodes.
        for etype in self.all_edge_types:
            if etype == self.target_edge_type:
                edge_index_dict[etype] = edge_index
            else:
                # Create empty edge index [2, 0] on the same device as input
                edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        # 3. Call the original model
        return self.model(
            x_dict, 
            edge_index_dict, 
            edge_label_index, 
            self.target_edge_type
        )

# =========================================================================
# 3. TRAIN & TEST LOOPS
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

        # 1. POSITIVE SAMPLES
        pos_edge_index = batch[target_edge_type].edge_label_index
        pos_pred = model(
            batch.x_dict, 
            batch.edge_index_dict, 
            pos_edge_index,
            target_edge_type
        )

        # 2. NEGATIVE SAMPLES (In-Batch Shuffling)
        neg_edge_index = pos_edge_index.clone()
        neg_edge_index[1] = neg_edge_index[1][torch.randperm(neg_edge_index.size(1))]
        
        neg_pred = model(
            batch.x_dict, 
            batch.edge_index_dict, 
            neg_edge_index,
            target_edge_type
        )

        # 3. LOSS
        all_pred = torch.cat([pos_pred, neg_pred])
        all_labels = torch.cat([
            torch.ones(pos_pred.size(0), device=device),
            torch.zeros(neg_pred.size(0), device=device)
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
        
        # 2. Negatives
        neg_index = pos_index.clone()
        neg_index[1] = neg_index[1][torch.randperm(neg_index.size(1))]
        neg_pred = model(batch.x_dict, batch.edge_index_dict, neg_index, target_edge_type)
        
        # 3. Accuracy
        pos_acc = (torch.sigmoid(pos_pred) > 0.5).sum().item()
        neg_acc = (torch.sigmoid(neg_pred) < 0.5).sum().item()
        
        total_correct += (pos_acc + neg_acc)
        total_examples += (pos_pred.numel() + neg_pred.numel())

    return total_correct / total_examples

# =========================================================================
# 4. MAIN RUNNER
# =========================================================================
def run_proc(
    node_rank, num_nodes, dataset_root_dir, master_addr,
    ddp_port, train_loader_port, test_loader_port,
    num_epochs, batch_size, num_neighbors, num_workers, model_save_path
):
    print(f'[Rank {node_rank}] Loading partitions...')
    
    # Load META.json
    with open(osp.join(dataset_root_dir, 'META.json'), 'r') as f:
        meta = json.load(f)

    graph_store = LocalGraphStore.from_partition(dataset_root_dir, pid=node_rank)
    feature_store = LocalFeatureStore.from_partition(dataset_root_dir, pid=node_rank)
    
    node_types = meta['node_types']
    edge_types = [tuple(x) for x in meta['edge_types']] 
    metadata = (node_types, edge_types)

    # Setup Distributed Context
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
    
    target_edge_type = ("User", "TOOK", "Quiz")
    
    # Retrieve local edges for training
    target_attr = None
    for attr in graph_store.get_all_edge_attrs():
        if attr.edge_type == target_edge_type:
            target_attr = attr
            break
            
    if target_attr:
        local_edge_index = graph_store._get_edge_index(target_attr)
    else:
        local_edge_index = torch.empty((2, 0), dtype=torch.long)

    neighbor_sizes = [int(i) for i in num_neighbors.split(',')]
    
    print(f'[Rank {node_rank}] Creating DistLinkNeighborLoader...')
    
    train_loader = DistLinkNeighborLoader(
        data=(feature_store, graph_store),
        edge_label_index=(target_edge_type, local_edge_index),
        neg_sampling_ratio=0.0, 
        num_neighbors=neighbor_sizes,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=train_loader_port,
        current_ctx=dist_context, 
    )

    test_loader = DistLinkNeighborLoader(
        data=(feature_store, graph_store),
        edge_label_index=(target_edge_type, local_edge_index), 
        neg_sampling_ratio=0.0,
        num_neighbors=neighbor_sizes,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=test_loader_port,
        current_ctx=dist_context,
    ) 

    # Model Setup
    device = torch.device('cpu')
    in_channels = -1 
    
    base_model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=64,
        num_layers=len(neighbor_sizes),
        out_channels=64
    )
    
    hetero_gnn = to_hetero(base_model, metadata, aggr='sum').to(device)
    model = RecommenderModel(hetero_gnn).to(device)
    
    # --- LAZY INIT FIX ---
    print(f'[Rank {node_rank}] Running dummy forward pass to initialize lazy modules...')
    with torch.no_grad():
        dummy_x_dict = {}
        # Read exact dims from META to ensure no mismatches
        for ntype, schema in meta['node_feat_schema'].items():
            feat_dim = schema['x'][0] 
            dummy_x_dict[ntype] = torch.randn(2, feat_dim, device=device)
            
        dummy_edge_index_dict = {}
        for etype_list in meta['edge_types']:
            etype = tuple(etype_list)
            dummy_edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long, device=device)
            
        dummy_label_index = torch.zeros((2, 2), dtype=torch.long, device=device)
        model(dummy_x_dict, dummy_edge_index_dict, dummy_label_index, target_edge_type)
    print(f'[Rank {node_rank}] Initialization complete.')

    model = DistributedDataParallel(model, find_unused_parameters=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f'[Rank {node_rank}] Starting Training...')
    
    for epoch in range(1, num_epochs + 1):
        loss = train(model, train_loader, optimizer, dist_context, device, epoch, target_edge_type)
        if node_rank == 0:
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
        torch.distributed.barrier()

    # =========================================================================
    # 5. SAVE MODEL LOGIC (JIT EXPORT + HDFS PUSH)
    # =========================================================================
    if node_rank == 0:
        os.makedirs(model_save_path, exist_ok=True)

        # A. Determine Version Number
        pattern = re.compile(r'model_epoch_(\d+)\.pt')
        existing_epochs = []
        for filename in os.listdir(model_save_path):
            match = pattern.match(filename)
            if match:
                existing_epochs.append(int(match.group(1)))
        
        next_epoch = max(existing_epochs) + 1 if existing_epochs else 1
        local_filename = f'model_epoch_{next_epoch}.pt'
        save_full_path = osp.join(model_save_path, local_filename)

        # B. EXPORT TO TORCHSCRIPT (Using TRACE)
        print(f"Preparing to JIT Trace model version {next_epoch}...")
        
        # 1. Unwrap DDP 
        model_to_save = model.module.cpu()
        
        # 2. Prepare Wrapper with ALL edge types (Critical Fix)
        # We rely on the 'edge_types' variable defined earlier in run_proc
        inference_model = InferenceWrapper(model_to_save, edge_types) 
        inference_model.eval()

        # 3. Create Dummy Inputs for Tracing
        user_dim = meta['node_feat_schema']['User']['x'][0]
        quiz_dim = meta['node_feat_schema']['Quiz']['x'][0]
        
        dummy_x_user = torch.randn(10, user_dim) 
        dummy_x_quiz = torch.randn(10, quiz_dim)
        dummy_edge_index = torch.randint(0, 5, (2, 20), dtype=torch.long)
        dummy_label_index = torch.randint(0, 5, (2, 5), dtype=torch.long)

        # 4. Trace and Save
        try:
            # Note: We use strict=False to allow dictionary construction with tuple keys 
            # inside the trace, although trace usually handles this fine by unrolling.
            traced_model = torch.jit.trace(
                inference_model, 
                (dummy_x_user, dummy_x_quiz, dummy_edge_index, dummy_label_index),
                strict=False
            )
            
            torch.jit.save(traced_model, save_full_path)
            print(f"Model successfully JIT TRACED and saved locally: {save_full_path}")
            
            # C. PUSH TO HDFS
            hdfs_root = "/triton_models"
            triton_model_name = "recsys_model"
            
            hdfs_version_dir = f"{hdfs_root}/{triton_model_name}/{next_epoch}"
            hdfs_target_file = f"{hdfs_version_dir}/model.pt"

            print(f"Pushing to HDFS: {hdfs_target_file}...")

            # 1. Create HDFS Directory
            subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_version_dir], check=True)

            # 2. Upload file
            subprocess.run(["hdfs", "dfs", "-put", "-f", save_full_path, hdfs_target_file], check=True)
            print("Successfully pushed model to HDFS.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAILED to save or push model: {e}")

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', type=str, default='/tmp/pyg_recsys_hetero')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_neighbors', type=str, default='5,3')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--num_workers', type=int, default=0)
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
