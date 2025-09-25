import os
import torch
import torch.nn.functional as F
from collections import defaultdict

# PyTorch Distributed
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel

# PyTorch Geometric
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.distributed import DistContext, LocalFeatureStore, LocalGraphStore, DistNeighborLoader

# Define a standard GNN model for node classification
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def run():
    """
    Main function to initialize and run the distributed training process.
    """
    # 1. Initialization
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"--- [Rank {rank}/{world_size}] SCRIPT STARTED ---")

    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')

    ddp_port = int(master_port)
    train_loader_port = int(master_port)
    val_loader_port = int(master_port)
    test_loader_port = int(master_port)

    # --- 1. Initialize DDP for gradient synchronization ---
    ddp_init_method = f'tcp://{master_addr}:{ddp_port}'
    print(f"--- [Rank {rank}] Initializing DDP Process Group... -> {ddp_init_method} ---")
    dist.init_process_group(
        backend="gloo",
        init_method=ddp_init_method,
        rank=rank,
        world_size=world_size
    )
    print(f"--- [Rank {dist.get_rank()}] DDP Process Group INITIALIZED! ---")

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='pyg-dist-training'
    )
    
    device = torch.device('cpu')

    if rank == 0:
        print("\n--- Distributed Training Environment Ready ---")
        print(f"World Size: {world_size}")
        print("--------------------------------------------\n")

    # 2. Data Loading
    data_path = os.environ.get('LOCAL_DATA_PATH', '/pyg_dataset')

    if rank == 0:
        print(f"Loading partition data for all ranks from base path: {data_path}")

    global_labels_path = os.path.join(data_path, 'labels.pt')

    graph_store = LocalGraphStore.from_partition(data_path, pid=rank)
    feature_store = LocalFeatureStore.from_partition(data_path, pid=rank)
    feature_store.labels = torch.load(global_labels_path, map_location='cpu')
    data = (feature_store, graph_store)

    # Get global indices for training, validation, and test sets for this partition
    train_mask = feature_store.get_tensor(group_name=None, attr_name='train_mask')
    train_idx = feature_store.get_global_id(group_name=None)[train_mask]
    
    val_mask = feature_store.get_tensor(group_name=None, attr_name='val_mask')
    val_idx = feature_store.get_global_id(group_name=None)[val_mask]
    
    test_mask = feature_store.get_tensor(group_name=None, attr_name='test_mask')
    test_idx = feature_store.get_global_id(group_name=None)[test_mask]
    
    print("--- Data Loaded ---")
    print(f"Rank {rank} has {len(train_idx)} training nodes, {len(val_idx)} validation nodes, and {len(test_idx)} test nodes.")
    print("-------------------")

    # 3. GNN Model
    num_features = feature_store.get_tensor_size(group_name=None, attr_name='x')[1]
    num_classes = int(feature_store.labels.max()) + 1

    model = GNN(
        in_channels=num_features,
        hidden_channels=128,
        out_channels=num_classes,
    )
    model = DistributedDataParallel(model, find_unused_parameters=True)

    # 4. Distributed Data Loaders
    train_loader = DistNeighborLoader(
        data=data,
        input_nodes=train_idx,
        batch_size=128,
        num_neighbors=[10, 5],
        shuffle=True,
        current_ctx=current_ctx,
        master_addr=master_addr, 
        master_port=train_loader_port,
        filter_per_worker=True,
    )
    
    if rank == 0:
        print("--- Setup Complete, Starting Training ---")

    # 5. Distributed Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        total_train_nodes = 0
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            
            # ========================== THE FIX: Part 2 ==========================
            # 1. Get the model's predictions for only the seed nodes.
            out_for_loss = out[:batch.batch_size]

            # 2. Get the corresponding labels directly from the batch object.
            # The loader has already fetched the correct labels for you.
            # Note: The loader might not slice .y, so we slice it to be safe.
            target_labels = batch.y[:batch.batch_size]
        
           # 3. Compute the loss.
            loss = F.cross_entropy(out_for_loss, target_labels)
            # ======================= END OF FIX: Part 2 ========================

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.batch_size
            total_train_nodes += batch.batch_size

        dist.barrier()

        # Aggregate training loss from all ranks
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_train_nodes_tensor = torch.tensor(total_train_nodes, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_train_nodes_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss_tensor.item() / total_train_nodes_tensor.item()
        
        if rank == 0:
            print(f'Epoch: {epoch:03d}, Avg Train Loss: {avg_loss:.4f}')
        
        dist.barrier()

    print(f"--- [Rank {rank}] Destroying DDP process group... ---")
    dist.destroy_process_group()
    
    print(f"--- [Rank {rank}] SCRIPT FINISHED ---")

if __name__ == "__main__":
    run()
