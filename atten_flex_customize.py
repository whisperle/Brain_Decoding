import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attn_gym import visualize_attention_scores
class NearestNeighborAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, num_neighbors, visual_cortex_mask):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.num_neighbors = num_neighbors
        
        # Initialize projection layers
        self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # Process the mask
        self.mask_3d = visual_cortex_mask
        self.mask_flat = visual_cortex_mask.flatten()
        self.active_voxels = torch.nonzero(self.mask_flat).squeeze()
        
        # Compute nearest neighbors
        self.nearest_voxels = self._compute_nearest_neighbors()
        
        # Create the attention mask
        self.attention_mask = self._create_attention_mask()

    def _compute_nearest_neighbors(self):
        device = self.mask_3d.device
        shape = self.mask_3d.shape
        indices = torch.arange(self.mask_flat.numel(), device=device).view(shape)
        active_indices = indices[self.mask_3d]
        
        z, y, x = torch.meshgrid(torch.arange(shape[0], device=device),
                                 torch.arange(shape[1], device=device),
                                 torch.arange(shape[2], device=device))
        all_coords = torch.stack((z[self.mask_3d], y[self.mask_3d], x[self.mask_3d]), dim=1)
        
        distances = torch.cdist(all_coords.float(), all_coords.float())
        _, nearest = torch.topk(distances, k=self.num_neighbors + 1, largest=False)
        return active_indices[nearest[:, 1:]]  # Exclude self

    def _neighborhood_mask(self, b, h, q_idx, kv_idx):
        q_idx_clamped = torch.clamp(q_idx, 0, self.nearest_voxels.size(0) - 1)
        kv_idx_clamped = torch.clamp(kv_idx, 0, self.nearest_voxels.size(0) - 1)
        return torch.any(self.nearest_voxels[q_idx_clamped] == self.active_voxels[kv_idx_clamped].unsqueeze(-1), dim=-1)

    def _create_attention_mask(self):
        return create_block_mask(
            self._neighborhood_mask,
            B=None,
            H=None,
            Q_LEN=self.active_voxels.shape[0],
            KV_LEN=self.active_voxels.shape[0],
            device=self.mask_3d.device
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply flex attention
        output = flex_attention(query, key, value, block_mask=self.attention_mask)
        
        # Reshape output and return
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
    
def visualize_custom_attention(query, key, custom_mask_fn, device="cpu", name="custom_attention_mask"):
    from attn_gym import visualize_attention_scores
    visualize_attention_scores(
        query,
        key,
        mask_mod=custom_mask_fn,
        device=device,
        name=name,
    )

# # Example usage
# if __name__ == "__main__":
#     import numpy as np
    
#     # Set up example data
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     voxel_grid_size = (10, 10, 10)
    
#     # Create a sample visual cortex mask
#     visual_cortex_mask = torch.zeros(voxel_grid_size, dtype=torch.bool, device=device)
#     visual_cortex_mask[3:7, 3:7, 3:7] = True  # Example: central region is the visual cortex
    
#     # Initialize the attention module
#     feature_dim = 32
#     num_heads = 4
#     num_neighbors = 5
#     attention = NearestNeighborAttention(feature_dim, num_heads, num_neighbors, visual_cortex_mask)
    
#     # Create dummy input (only for masked voxels)
#     batch_size = 1
#     num_masked_voxels = visual_cortex_mask.sum().item()
#     x = torch.randn(batch_size, num_masked_voxels, feature_dim, device=device)
    
#     # Forward pass
#     output = attention(x)
#     print("Output shape:", output.shape)
    
#     # Visualize attention
#     B, H, SEQ_LEN, HEAD_DIM = 1, 1, num_masked_voxels, feature_dim // num_heads
#     query = key = torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)
#     visualize_custom_attention(
#         query,
#         key,
#         custom_mask_fn=attention._neighborhood_mask,
#         device=device,
#         name="neighborhood_attention_mask"
#     )