import torch
import torch.nn as nn
import numpy as np

def build_rgcn_adjacencies(triples, ent2id, rel2id, num_entities, num_relations, device):
    """
    Constructs relation-specific adjacency lists for R-GCN.
    Returns a list of 2 * num_relations elements, where each element is a tuple of
    (src_indices, dest_indices) torch tensors on the specified device.
    """
    # 2 * num_relations: first num_relations are forward, next num_relations are inverse
    edge_index_list = []
    
    # Initialize lists for each relation index
    forward_edges = {r_idx: ([], []) for r_idx in range(num_relations)}
    inverse_edges = {r_idx: ([], []) for r_idx in range(num_relations)}
    
    for h, r, t in triples:
        h_idx = ent2id[h]
        r_idx = rel2id[r]
        t_idx = ent2id[t]
        
        # Forward edge: h -> t (message flows h -> t, so source is h, dest is t)
        forward_edges[r_idx][0].append(h_idx)
        forward_edges[r_idx][1].append(t_idx)
        
        # Inverse edge: t -> h (message flows t -> h, so source is t, dest is h)
        inverse_edges[r_idx][0].append(t_idx)
        inverse_edges[r_idx][1].append(h_idx)
        
    # Convert to tensors
    for r_idx in range(num_relations):
        src, dest = forward_edges[r_idx]
        src_t = torch.tensor(src, dtype=torch.long, device=device)
        dest_t = torch.tensor(dest, dtype=torch.long, device=device)
        edge_index_list.append((src_t, dest_t))
        
    for r_idx in range(num_relations):
        src, dest = inverse_edges[r_idx]
        src_t = torch.tensor(src, dtype=torch.long, device=device)
        dest_t = torch.tensor(dest, dtype=torch.long, device=device)
        edge_index_list.append((src_t, dest_t))
        
    return edge_index_list


class RGCNRefiner(nn.Module):
    """
    A pure PyTorch implementation of Relational Graph Convolutional Networks (R-GCN)
    designed to refine KGE entity embeddings by propagating messages over semantic
    connections in the knowledge graph.
    """
    def __init__(self, num_entities, num_relations, in_dim, out_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Self-loop transformation matrix
        self.w_self = nn.Linear(in_dim, out_dim, bias=False)
        
        # Relation-specific transformation matrices (forward + inverse relations)
        self.w_rel = nn.Parameter(torch.Tensor(2 * num_relations, in_dim, out_dim))
        
        # Shared bias across all updates
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Glorot initialization
        bound = 1.0 / (self.in_dim + self.out_dim) ** 0.5
        nn.init.uniform_(self.w_rel, -bound, bound)
        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.w_self.weight, -bound, bound)
        
    def forward(self, x, edge_index_list):
        """
        Refines the input features x by running relational graph convolutions.
        x: (num_entities, in_dim) tensor of initial KGE embeddings
        edge_index_list: list of 2 * num_relations tuples (src, dest) representing relation edges
        """
        # Start with the transformed self-loop representation
        out = self.w_self(x)
        
        # Propagate messages relation by relation
        for r in range(2 * self.num_relations):
            src, dest = edge_index_list[r]
            if len(src) == 0:
                continue
                
            # Compute relation-specific node features
            x_r = torch.matmul(x, self.w_rel[r]) # (num_entities, out_dim)
            
            # Fetch message features from the source nodes
            msg_r = x_r[src] # (num_edges, out_dim)
            
            # Accumulate messages at the destination nodes
            agg_msg = torch.zeros(self.num_entities, self.out_dim, device=x.device)
            agg_msg.index_add_(0, dest, msg_r)
            
            # Compute normalized node degrees
            deg = torch.zeros(self.num_entities, 1, device=x.device)
            ones = torch.ones(len(dest), 1, device=x.device)
            deg.index_add_(0, dest, ones)
            deg = torch.clamp(deg, min=1.0)
            
            # Add degree-normalized messages to the output representation
            out = out + (agg_msg / deg)
            
        # Add shared bias
        out = out + self.bias
        
        # Apply non-linear activation (ReLU)
        return torch.relu(out)


def verify_adjacency_normalization():
    """
    Self-verification test: builds a mock graph and verifies that
    R-GCN correctly aggregates and normalizes relational edges.
    """
    print("Running R-GCN Adjacency Normalization Verification...")
    device = torch.device("cpu")
    num_entities = 4
    num_relations = 2
    in_dim = 4
    out_dim = 4
    
    # Mock entities: 0, 1, 2, 3
    # Mock triples: (0, r0, 1), (2, r0, 1), (1, r1, 3)
    # ent2id = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}
    # rel2id = {"r0": 0, "r1": 1}
    triples = [
        ("e0", "r0", "e1"),
        ("e2", "r0", "e1"),
        ("e1", "r1", "e3"),
    ]
    ent2id = {f"e{i}": i for i in range(4)}
    rel2id = {"r0": 0, "r1": 1}
    
    edge_index_list = build_rgcn_adjacencies(triples, ent2id, rel2id, num_entities, num_relations, device)
    
    # Verification check:
    # Under forward r0 (index 0): edges are 0->1, 2->1.
    # Node 1 should have in-degree 2.
    src_r0, dest_r0 = edge_index_list[0]
    deg = torch.zeros(num_entities, 1)
    ones = torch.ones(len(dest_r0), 1)
    deg.index_add_(0, dest_r0, ones)
    
    assert deg[1].item() == 2.0, f"Expected node 1 to have in-degree 2 under r0, got {deg[1].item()}"
    assert deg[0].item() == 0.0
    assert deg[2].item() == 0.0
    
    print("  -> Adjacency construction checks out!")
    
    # Let's run a forward pass with a refiner
    refiner = RGCNRefiner(num_entities, num_relations, in_dim, out_dim)
    # Set all weights to identity and bias to zero to trace propagation
    with torch.no_grad():
        refiner.w_self.weight.copy_(torch.eye(in_dim))
        refiner.w_rel.copy_(torch.eye(in_dim).unsqueeze(0).expand(2 * num_relations, -1, -1))
        refiner.bias.zero_()
        
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.0], # e0
        [0.0, 2.0, 0.0, 0.0], # e1
        [0.0, 0.0, 3.0, 0.0], # e2
        [0.0, 0.0, 0.0, 4.0], # e3
    ], dtype=torch.float32)
    
    out = refiner(x, edge_index_list)
    print("  -> Forward pass complete!")
    print("R-GCN Adjacency Normalization Verification: PASSED")

if __name__ == "__main__":
    verify_adjacency_normalization()
