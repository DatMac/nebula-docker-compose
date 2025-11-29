import torch
import os
import os.path as osp

root = "/pyg_recsys_hetero"  # Your data path

# 1. Fix Node Map
node_map_path = osp.join(root, "node_map.pt")
if osp.exists(node_map_path):
    print("Converting node_map.pt to directory...")
    # Load the dictionary: {'User': tensor, 'Quiz': tensor}
    data = torch.load(node_map_path)
    
    # Create directory
    os.makedirs(osp.join(root, "node_map"), exist_ok=True)
    
    for node_type, tensor in data.items():
        # Save individually: /pyg_recsys_hetero/node_map/User.pt
        save_path = osp.join(root, "node_map", f"{node_type}.pt")
        torch.save(tensor, save_path)
        print(f"  - Saved {save_path}")
    
    # Remove original file
    os.remove(node_map_path)
else:
    print("node_map.pt not found (already fixed?)")

# 2. Fix Edge Map
edge_map_path = osp.join(root, "edge_map.pt")
if osp.exists(edge_map_path):
    print("Converting edge_map.pt to directory...")
    # Load dictionary: {('User','TOOK','Quiz'): tensor, ...}
    data = torch.load(edge_map_path)
    
    os.makedirs(osp.join(root, "edge_map"), exist_ok=True)
    
    for edge_tuple, tensor in data.items():
        # PyG expects filename format: src__rel__dst.pt
        filename = "__".join(edge_tuple) + ".pt"
        save_path = osp.join(root, "edge_map", filename)
        torch.save(tensor, save_path)
        print(f"  - Saved {save_path}")
        
    os.remove(edge_map_path)
else:
    print("edge_map.pt not found (already fixed?)")

print("Fix complete. You can now run torchrun.")
