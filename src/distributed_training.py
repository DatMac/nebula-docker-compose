import argparse
import json
import os
import os.path as osp
import time
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.distributed import (
    DistContext,
    DistNeighborLoader,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.nn import GraphSAGE, to_hetero

# Re-using the train and test functions from your example as they are generic
# and will work with the new data loading mechanism.
# ... (The train and test functions from your provided example go here) ...
@torch.no_grad()
def test(
    model,
    loader,
    dist_context,
    device,
    epoch,
    logfile=None,
    num_loader_threads=10,
    progress_bar=True,
):
    def test_homo(batch):
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y_pred = out.argmax(dim=-1)
        y_true = batch.y[:batch.batch_size]
        return y_pred, y_true

    def test_hetero(batch):
        # Assuming the target node type is 'paper' as in the original example.
        # This might need to be adjusted for your specific dataset.
        target_node_type = list(batch.y_dict.keys())[0]
        batch_size = batch[target_node_type].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out[target_node_type][:batch_size]
        y_pred = out.argmax(dim=-1)
        y_true = batch[target_node_type].y[:batch_size]
        return y_pred, y_true

    model.eval()
    total_examples = total_correct = 0

    if loader.num_workers > 0:
        context = loader.enable_multithreading(num_loader_threads)
    else:
        context = nullcontext()

    with context:
        if progress_bar:
            loader = tqdm(loader, desc=f'[Node {dist_context.rank}] Test')

        start_time = batch_time = time.time()
        for i, batch in enumerate(loader):
            batch = batch.to(device)

            if isinstance(batch, HeteroData):
                y_pred, y_true = test_hetero(batch)
            else:
                y_pred, y_true = test_homo(batch)

            total_correct += int((y_pred == y_true).sum())
            total_examples += y_pred.size(0)
            batch_acc = int((y_pred == y_true).sum()) / y_pred.size(0)

            result = (f'[Node {dist_context.rank}] Test: epoch={epoch}, '
                      f'it={i}, acc={batch_acc:.4f}, '
                      f'time={(time.time() - batch_time):.4f}')
            batch_time = time.time()

            if logfile:
                log = open(logfile, 'a+')
                log.write(f'{result}\n')
                log.close()

            if not progress_bar:
                print(result)

    total_acc = total_correct / total_examples
    print(f'[Node {dist_context.rank}] Test epoch {epoch} END: '
          f'acc={total_acc:.4f}, time={(time.time() - start_time):.2f}')
    torch.distributed.barrier()


def train(
    model,
    loader,
    optimizer,
    dist_context,
    device,
    epoch,
    logfile=None,
    num_loader_threads=10,
    progress_bar=True,
):
    def train_homo(batch):
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size])
        return loss, batch.batch_size

    def train_hetero(batch):
        # Assuming the target node type is 'paper' as in the original example.
        # This might need to be adjusted for your specific dataset.
        target_node_type = list(batch.y_dict.keys())[0]
        batch_size = batch[target_node_type].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out[target_node_type][:batch_size]
        target = batch[target_node_type].y[:batch_size]
        loss = F.cross_entropy(out, target)
        return loss, batch_size

    model.train()
    total_loss = total_examples = 0

    if loader.num_workers > 0:
        context = loader.enable_multithreading(num_loader_threads)
    else:
        context = nullcontext()

    with context:
        if progress_bar:
            loader = tqdm(loader, desc=f'[Node {dist_context.rank}] Train')

        start_time = batch_time = time.time()
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            if isinstance(batch, HeteroData):
                loss, batch_size = train_hetero(batch)
            else:
                loss, batch_size = train_homo(batch)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * batch_size
            total_examples += batch_size

            result = (f'[Node {dist_context.rank}] Train: epoch={epoch}, '
                      f'it={i}, loss={loss:.4f}, '
                      f'time={(time.time() - batch_time):.4f}')
            batch_time = time.time()

            if logfile:
                log = open(logfile, 'a+')
                log.write(f'{result}\n')
                log.close()

            if not progress_bar:
                print(result)

    print(f'[Node {dist_context.rank}] Train epoch {epoch} END: '
          f'loss={total_loss/total_examples:.4f}, '
          f'time={(time.time() - start_time):.2f}')
    torch.distributed.barrier()


def run_proc(
    local_proc_rank: int,
    node_rank: int,
    num_nodes: int,
    dataset_root_dir: str,
    master_addr: str,
    ddp_port: int,
    train_loader_port: int,
    test_loader_port: int,
    num_epochs: int,
    batch_size: int,
    num_neighbors: str,
    async_sampling: bool,
    concurrency: int,
    num_workers: int,
    num_loader_threads: int,
    progress_bar: bool,
    logfile: str,
    model_save_path: str,
):
    # ==================== MODIFIED PART 1: Data Loading ====================
    print('--- Loading data partition files ...')
    
    # Load metadata from the root directory
    with open(osp.join(dataset_root_dir, 'META.json'), 'r') as f:
        meta = json.load(f)
        
    is_hetero = 'node_map' in meta # A good indicator for heterogeneous graphs
    
    # Load partition into local graph store and feature store:
    graph = LocalGraphStore.from_partition(dataset_root_dir, pid=node_rank)
    feature = LocalFeatureStore.from_partition(dataset_root_dir, pid=node_rank)

    # The `labels.pt` file should contain tensors for labels, train_mask, val_mask, and test_mask
    labels_path = osp.join(dataset_root_dir, 'labels.pt')
    labels_data = torch.load(labels_path)
    
    # Get global indices for training and testing
    train_idx = torch.where(labels_data['train_mask'])[0]
    test_idx = torch.where(labels_data['test_mask'])[0]
    
    # Add labels to the feature store
    feature.labels = labels_data['y']

    partition_data = (feature, graph)

    if is_hetero:
        # Assuming the primary node type for training/testing is the first one in the list.
        # Adjust if your dataset has a different target node type.
        target_node_type = meta['node_types'][0]
        train_idx = (target_node_type, train_idx)
        test_idx = (target_node_type, test_idx)
        print(f"Heterogeneous graph detected. Target node type: '{target_node_type}'")
    
    print(f'Partition metadata: {graph.meta}')
    # ========================================================================

    # Initialize distributed context:
    current_ctx = DistContext(
        world_size=num_nodes,
        rank=node_rank,
        global_world_size=num_nodes,
        global_rank=node_rank,
        group_name='pyg-distributed-training',
    )
    current_device = torch.device('cpu')

    print('--- Initialize DDP training group ...')
    torch.distributed.init_process_group(
        backend='gloo',
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method=f'tcp://{master_addr}:{ddp_port}',
    )

    print('--- Initialize distributed loaders ...')
    num_neighbors = [int(i) for i in num_neighbors.split(',')]
    # Create distributed neighbor loader for training:
    train_loader = DistNeighborLoader(
        data=partition_data,
        input_nodes=train_idx,
        current_ctx=current_ctx,
        device=current_device,
        num_neighbors=num_neighbors,
        shuffle=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=train_loader_port,
        concurrency=concurrency,
        async_sampling=async_sampling,
    )
    # Create distributed neighbor loader for testing:
    test_loader = DistNeighborLoader(
        data=partition_data,
        input_nodes=test_idx,
        current_ctx=current_ctx,
        device=current_device,
        num_neighbors=num_neighbors,
        shuffle=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=test_loader_port,
        concurrency=concurrency,
        async_sampling=async_sampling,
    )

    if node_rank == 0:
        print("All loaders initialized. Synchronizing all nodes before training...")
    torch.distributed.barrier()
    if node_rank == 0:
        print("Synchronization complete. Starting training.")

    # ================== MODIFIED PART 2: Model Configuration ==================
    print('--- Initialize model ...')
    
    if is_hetero:
        # For heterogeneous graphs, feature sizes are in a dictionary.
        # We assume the target node type's feature size is the 'in_channels'.
        in_channels = meta['node_feat_schema'][target_node_type]['__feat__']
    else:
        # For homogeneous graphs, it's a single value.
        in_channels = meta['node_feat_schema']['__feat__']

    out_channels = meta['num_classes']

    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=256,
        num_layers=len(num_neighbors),
        out_channels=out_channels,
    ).to(current_device)

    if is_hetero:  # Turn model into a heterogeneous variant:
        # Metadata for to_hetero requires node_types and edge_types
        model_meta = (meta['node_types'], meta['edge_types'])
        model = to_hetero(model, model_meta, aggr='sum').to(current_device)
        torch.distributed.barrier()
    # ========================================================================

    # Enable DDP:
    model = DistributedDataParallel(model, find_unused_parameters=is_hetero)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.distributed.barrier()

    # Train and test:
    print(f'--- Start training for {num_epochs} epochs ...')
    for epoch in range(1, num_epochs + 1):
        print(f'Train epoch {epoch}/{num_epochs}:')
        train(
            model,
            train_loader,
            optimizer,
            current_ctx,
            current_device,
            epoch,
            logfile,
            num_loader_threads,
            progress_bar,
        )

        if epoch % 5 == 0:
            print(f'Test epoch {epoch}/{num_epochs}:')
            test(
                model,
                test_loader,
                current_ctx,
                current_device,
                epoch,
                logfile,
                num_loader_threads,
                progress_bar,
            )

        if epoch % 10 == 0 or epoch == num_epochs:
            # Ensure all processes are synchronized before saving.
            torch.distributed.barrier()
            
            # Only the main process (rank 0) should save the model.
            if node_rank == 0:
                # Create the directory if it doesn't exist.
                os.makedirs(model_save_path, exist_ok=True)
                
                checkpoint_path = osp.join(model_save_path, f'model_epoch_{epoch}.pt')
                print(f'[Node 0] Saving model checkpoint to: {checkpoint_path}')
                
                # When using DDP, the model is wrapped. We need to save `model.module.state_dict()`.
                torch.save(model.module.state_dict(), checkpoint_path)

    print(f'--- [Node {current_ctx.rank}] Closing ---')
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for distributed training')

    # ========== MODIFIED PART 3: Argument Parsing & Environment ==========
    # Removed dataset-specific args, using a generic root directory.
    parser.add_argument(
        '--dataset_root_dir',
        type=str,
        default='/pyg_dataset', # Default to the path in your container
        help='The root directory of the partitioned dataset',
    )
    # Distributed configuration will be read from environment variables
    parser.add_argument(
        '--num_neighbors',
        type=str,
        default='15,10,5',
        help='Number of node neighbors sampled at each layer',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='The number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size for training and testing',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2, # Adjusted default for containerized env
        help='Number of sampler sub-processes',
    )
    parser.add_argument(
        '--num_loader_threads',
        type=int,
        default=10,
        help='Number of threads used for each sampler sub-process',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=4,
        help='Number of maximum concurrent RPC for each sampler',
    )
    parser.add_argument(
        '--async_sampling',
        action='store_true',
        help='Whether sampler processes RPC requests asynchronously',
    )
    # Port configurations
    parser.add_argument(
        '--ddp_port',
        type=int,
        default=11111,
        help="The port used for PyTorch's DDP communication",
    )
    parser.add_argument(
        '--train_loader_port',
        type=int,
        default=11112,
        help='The port used for RPC communication across training samplers',
    )
    parser.add_argument(
        '--test_loader_port',
        type=int,
        default=11113,
        help='The port used for RPC communication across test samplers',
    )
    parser.add_argument('--logging', action='store_true')
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='./saved_models',  # Default path inside the container
        help='Directory to save the trained model checkpoints',
    )
    parser.add_argument('--progress_bar', action='store_true')

    args = parser.parse_args()

    # Get distributed configuration from environment variables
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    node_rank = int(os.environ.get('RANK', '0'))
    num_nodes = int(os.environ.get('WORLD_SIZE', '1'))

    print('--- Distributed training on custom dataset ---')
    print(f'* total nodes: {num_nodes}')
    print(f'* node rank: {node_rank}')
    print(f'* master addr: {master_addr}')
    print(f'* dataset root dir: {args.dataset_root_dir}')
    print(f'* epochs: {args.num_epochs}')
    print(f'* batch size: {args.batch_size}')

    if args.logging:
        logfile = f'dist_cpu-node{node_rank}.txt'
        with open(logfile, 'a+') as log:
            log.write(f'\n--- Inputs: {str(args)}')
    else:
        logfile = None

    print('--- Launching training process ...')
    
    torch.multiprocessing.spawn(
        run_proc,
        args=(
            node_rank,
            num_nodes,
            args.dataset_root_dir,
            master_addr,
            args.ddp_port,
            args.train_loader_port,
            args.test_loader_port,
            args.num_epochs,
            args.batch_size,
            args.num_neighbors,
            args.async_sampling,
            args.concurrency,
            args.num_workers,
            args.num_loader_threads,
            args.progress_bar,
            logfile,
            args.model_save_path
        ),
        join=True,
    )

    print('--- Finished training process ---')
    # ========================================================================
