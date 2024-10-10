import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attn_gym import visualize_attention_scores
class NearestNeighborAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, num_neighbors):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.num_neighbors = num_neighbors
        
        # Initialize projection layers
        self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, feature_dim, bias=False)

    def _compute_nearest_neighbors(self, mask_3d, active_voxels):
        device = mask_3d.device
        shape = mask_3d.shape
        indices = torch.arange(mask_3d.numel(), device=device).view(shape)
        active_indices = indices[mask_3d]
        
        z, y, x = torch.meshgrid(
            torch.arange(shape[0], device=device),
            torch.arange(shape[1], device=device),
            torch.arange(shape[2], device=device),
            indexing='ij'  # Ensure correct indexing
        )
        all_coords = torch.stack((z[mask_3d], y[mask_3d], x[mask_3d]), dim=1)
        
        distances = torch.cdist(all_coords.float(), all_coords.float())
        _, nearest = torch.topk(distances, k=self.num_neighbors + 1, largest=False)
        return active_indices[nearest[:, 1:]]  # Exclude self

    def _create_attention_mask(self, mask_3d, active_voxels, nearest_voxels):
        def _neighborhood_mask(b, h, q_idx, kv_idx):
            q_idx_clamped = torch.clamp(q_idx, 0, nearest_voxels.size(0) - 1)
            kv_idx_clamped = torch.clamp(kv_idx, 0, nearest_voxels.size(0) - 1)
            return torch.any(
                nearest_voxels[q_idx_clamped] == active_voxels[kv_idx_clamped].unsqueeze(-1),
                dim=-1
            )
        return create_block_mask(
            _neighborhood_mask,
            B=None,
            H=None,
            Q_LEN=active_voxels.shape[0],
            KV_LEN=active_voxels.shape[0],
            device=mask_3d.device
        )

    def forward(self, x, visual_cortex_mask):
        batch_size, seq_len, _ = x.shape

        # Process the mask
        mask_3d = visual_cortex_mask
        active_voxels = torch.nonzero(mask_3d.flatten()).squeeze()
        print("active_voxels",active_voxels)
        # Compute nearest neighbors
        nearest_voxels = self._compute_nearest_neighbors(mask_3d, active_voxels)
        
        # Create the attention mask
        attention_mask = self._create_attention_mask(mask_3d, active_voxels, nearest_voxels)
    
        # Project input to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply flex attention
        output = flex_attention(query, key, value, block_mask=attention_mask)

        # Compute the metric: the mean of the key vectors over the heads
        metric = key.mean(1)  # Averaging over heads
        
        # Reshape output and return
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim), metric
    
def visualize_custom_attention(query, key, custom_mask_fn, device="cpu", name="custom_attention_mask"):
    from attn_gym import visualize_attention_scores
    visualize_attention_scores(
        query,
        key,
        mask_mod=custom_mask_fn,
        device=device,
        name=name,
    )
