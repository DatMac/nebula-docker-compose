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
    rpc_port = int(master_port) + 1  

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
    global_labels = torch.load(global_labels_path, map_location='cpu')

    graph_store = LocalGraphStore.from_partition(data_path, pid=rank)
    feature_store = LocalFeatureStore.from_partition(data_path, pid=rank)
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
    num_classes = int(global_labels.max()) + 1

    model = GNN(
        in_channels=num_features,
        hidden_channels=128,
        out_channels=num_classes,
    )
    model = DistributedDataParallel(model)

    # 4. Distributed Data Loaders
    train_loader = DistNeighborLoader(
        data=data,
        input_nodes=train_idx,
        batch_size=128,
        num_neighbors=[10, 5],
        shuffle=True,
        current_ctx=current_ctx,
        master_addr=master_addr, 
        master_port=master_port,
        filter_per_worker=True,
        drop_last=True,
    )
    
    val_loader = DistNeighborLoader(
        data=data,
        input_nodes=val_idx,
        batch_size=128,
        num_neighbors=[10, 5],
        shuffle=False,
        current_ctx=current_ctx,
        master_addr=master_addr, 
        master_port=master_port,   
        filter_per_worker=True,
        drop_last=True,
    ) 

    test_loader = DistNeighborLoader(
        data=data,
        input_nodes=test_idx,
        batch_size=128,
        num_neighbors=[10, 5],
        shuffle=False,
        current_ctx=current_ctx,
        master_addr=master_addr,
        master_port=master_port,
        filter_per_worker=True,
        drop_last=True,
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
            # 1. Get the global IDs of the seed nodes for this batch.
            seed_node_global_ids = batch.n_id[:batch.batch_size]

            # 2. Use these IDs to look up the labels from our complete global_labels tensor.
            target_labels = global_labels[seed_node_global_ids]

            # 3. Get the model's predictions for only the seed nodes.
            out_for_loss = out[:batch.batch_size]
            
            # 4. Compute the loss.
            loss = F.cross_entropy(out_for_loss, target_labels)
            # ======================= END OF FIX: Part 2 ========================

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.batch_size
            total_train_nodes += batch.batch_size

        # Aggregate training loss from all ranks
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_train_nodes_tensor = torch.tensor(total_train_nodes, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_train_nodes_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss_tensor.item() / total_train_nodes_tensor.item()
        
        if rank == 0:
            print(f'Epoch: {epoch:03d}, Avg Train Loss: {avg_loss:.4f}')
        
        dist.barrier()

        model.eval()
        total_val_loss = 0
        total_correct = 0
        total_val_nodes = 0
        
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(out, batch.y)
                
                pred = out.argmax(dim=1)
                correct = (pred == batch.y).sum()
                
                total_val_loss += loss.item() * batch.num_nodes
                total_correct += correct.item()
                total_val_nodes += batch.num_nodes

        # Aggregate validation metrics from all ranks
        total_val_loss_tensor = torch.tensor(total_val_loss, device=device)
        total_correct_tensor = torch.tensor(total_correct, device=device)
        total_val_nodes_tensor = torch.tensor(total_val_nodes, device=device)
        
        dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_val_nodes_tensor, op=dist.ReduceOp.SUM)

        # Calculate global validation metrics
        global_val_loss = total_val_loss_tensor.item() / total_val_nodes_tensor.item()
        global_val_acc = total_correct_tensor.item() / total_val_nodes_tensor.item()

        if rank == 0:
            print(f'Epoch: {epoch:03d}, Val Loss: {global_val_loss:.4f}, Val Acc: {global_val_acc:.4f}')

        # Synchronize all processes at the end of the epoch
        dist.barrier()

        # 6. Model Checkpointing
        if rank == 0 and epoch % 5 == 0:
            checkpoint_dir = '/app/models'
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f'--- Model checkpoint saved to {checkpoint_path} ---')

    print(f"--- Rank {rank} finished training ---")
    dist.barrier()

    if rank == 0:
        print("--- Starting Final Test Evaluation ---")
        
    model.eval()
    total_test_correct = 0
    total_test_nodes = 0
    
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred == batch.y).sum()
            total_test_correct += correct.item()
            total_test_nodes += batch.num_nodes

    # Aggregate test metrics
    total_test_correct_tensor = torch.tensor(total_test_correct, device=device)
    total_test_nodes_tensor = torch.tensor(total_test_nodes, device=device)

    dist.all_reduce(total_test_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_test_nodes_tensor, op=dist.ReduceOp.SUM)
    
    # Calculate global test accuracy
    global_test_acc = total_test_correct_tensor.item() / total_test_nodes_tensor.item()

    if rank == 0:
        print(f'Final Test Accuracy: {global_test_acc:.4f}')
        print("--------------------------------------")

    print(f"--- [Rank {rank}] Shutting down RPC... ---")
    rpc.shutdown()
    
    print(f"--- [Rank {rank}] Destroying DDP process group... ---")
    dist.destroy_process_group()
    
    print(f"--- [Rank {rank}] SCRIPT FINISHED ---")

if __name__ == "__main__":
    run()
