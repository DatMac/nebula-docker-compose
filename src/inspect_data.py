import os
import json
import torch
import os.path as osp
import sys

# Configuration
DATASET_ROOT = "/pyg_recsys_hetero"  # Change this if your path is different

def print_pass(msg):
    print(f"\033[92m[PASS]\033[0m {msg}")

def print_fail(msg):
    print(f"\033[91m[FAIL]\033[0m {msg}")

def print_warn(msg):
    print(f"\033[93m[WARN]\033[0m {msg}")

def inspect_dataset(root_dir):
    print(f"🔍 Inspecting Dataset at: {root_dir}\n" + "="*50)

    if not osp.exists(root_dir):
        print_fail(f"Directory {root_dir} does not exist.")
        return

    # ---------------------------------------------------------
    # 1. META.json
    # ---------------------------------------------------------
    meta_path = osp.join(root_dir, 'META.json')
    if not osp.exists(meta_path):
        print_fail("META.json missing!")
        return
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print_pass("META.json loaded")
        print(f"   ├─ Node Types: {meta.get('node_types')}")
        print(f"   ├─ Edge Types: {meta.get('edge_types')}")
        print(f"   └─ Num Partitions: {meta.get('num_parts')}")
        num_parts = meta.get('num_parts', 0)
    except Exception as e:
        print_fail(f"Corrupt META.json: {e}")
        return

    # ---------------------------------------------------------
    # 2. Global Maps (node_map / edge_map)
    # ---------------------------------------------------------
    print("\n🔍 Checking Global Maps...")
    
    # Check Node Map
    nm_path = osp.join(root_dir, 'node_map')
    if osp.isdir(nm_path):
        print_pass("node_map is a directory (Correct for Hetero)")
        for f in os.listdir(nm_path):
            if f.endswith('.pt'):
                data = torch.load(osp.join(nm_path, f))
                print(f"   └─ {f}: {data.size(0)} entries")
    elif osp.isfile(nm_path + '.pt'):
        print_warn("node_map.pt is a FILE. (Likely incorrect for Hetero, run patch script)")
    else:
        print_fail("node_map missing")

    # Check Edge Map
    em_path = osp.join(root_dir, 'edge_map')
    if osp.isdir(em_path):
        print_pass("edge_map is a directory (Correct for Hetero)")
        for f in os.listdir(em_path):
            if f.endswith('.pt'):
                data = torch.load(osp.join(em_path, f))
                print(f"   └─ {f}: {data.size(0)} entries")
    elif osp.isfile(em_path + '.pt'):
        print_warn("edge_map.pt is a FILE. (Likely incorrect for Hetero, run patch script)")
    else:
        print_fail("edge_map missing")

    # ---------------------------------------------------------
    # 3. Partitions
    # ---------------------------------------------------------
    print("\n🔍 Checking Partitions...")
    
    for pid in range(num_parts):
        p_dir = osp.join(root_dir, f'part_{pid}')
        print(f"\n📂 Partition {pid} ({p_dir})")
        
        if not osp.exists(p_dir):
            print_fail(f"Partition {pid} directory missing!")
            continue

        # A. node_feats.pt
        nf_path = osp.join(p_dir, 'node_feats.pt')
        if osp.exists(nf_path):
            try:
                nf = torch.load(nf_path)
                print_pass(f"node_feats.pt loaded")
                for ntype, data in nf.items():
                    # Expected structure: {'global_id': ..., 'feats': {'x': ...}}
                    count = data['global_id'].size(0)
                    feat_dim = data['feats']['x'].size(1) if 'x' in data['feats'] and data['feats']['x'].numel() > 0 else 0
                    print(f"   ├─ {ntype}: {count} nodes, Feat Dim: {feat_dim}")
            except Exception as e:
                print_fail(f"node_feats.pt error: {e}")
        else:
            print_fail("node_feats.pt missing")

        # B. graph.pt (Topology)
        g_path = osp.join(p_dir, 'graph.pt')
        if osp.exists(g_path):
            try:
                g = torch.load(g_path)
                print_pass(f"graph.pt loaded")
                for etype, data in g.items():
                    # Expected: {'row': ..., 'col': ..., 'size': ...}
                    num_edges = data['row'].size(0)
                    print(f"   ├─ {etype}: {num_edges} edges")
                    
                    if num_edges == 0:
                        print_warn(f"   ⚠️  Zero edges for {etype} in this partition!")
            except Exception as e:
                print_fail(f"graph.pt error: {e}")
        else:
            print_fail("graph.pt missing")

        # C. edge_feats.pt
        ef_path = osp.join(p_dir, 'edge_feats.pt')
        if osp.exists(ef_path):
            print_pass("edge_feats.pt exists")
        else:
            print_fail("edge_feats.pt missing")

if __name__ == "__main__":
    # If passed as arg, use that, else default
    path = sys.argv[1] if len(sys.argv) > 1 else DATASET_ROOT
    inspect_dataset(path)
