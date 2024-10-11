import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attn_gym import visualize_attention_scores

def visualize_custom_attention(query, key, custom_mask_fn, device="cpu", name="custom_attention_mask"):
    from attn_gym import visualize_attention_scores
    visualize_attention_scores(
        query,
        key,
        mask_mod=custom_mask_fn,
        device=device,
        name=name,
    )

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

    def _compute_nearest_neighbors(self, voxel_coords):
        """
        Efficiently compute the nearest neighbors for each voxel using broadcasting and matrix operations.
        
        Args:
            voxel_coords (torch.Tensor): Tensor of shape [batch_size, seq_len, coord_dim]
            
        Returns:
            torch.Tensor: Nearest neighbors indices, shape [batch_size, seq_len, num_neighbors]
        """
        batch_size, seq_len, _ = voxel_coords.shape

        # Compute pairwise distances in a batched manner
        # voxel_coords: [batch_size, seq_len, coord_dim]
        voxel_coords_expanded = voxel_coords.unsqueeze(1)  # [batch_size, 1, seq_len, coord_dim]
        voxel_coords_repeated = voxel_coords.unsqueeze(2)  # [batch_size, seq_len, 1, coord_dim]

        # Compute squared distances
        dists = torch.norm(voxel_coords_expanded - voxel_coords_repeated, dim=-1)  # [batch_size, seq_len, seq_len]

        # Sort the distances to find the nearest neighbors
        # The first closest point is always the point itself, so we exclude it by taking indices [1:num_neighbors+1]
        nearest_voxels = torch.argsort(dists, dim=-1)[:, :, 1:self.num_neighbors+1]  # [batch_size, seq_len, num_neighbors]

        return nearest_voxels
    
    def _create_attention_mask(self, nearest_voxels):
        # nearest_voxels: [batch_size, seq_len, num_neighbors]
        batch_size, seq_len, _ = nearest_voxels.shape
        def _neighborhood_mask(b, h, q_idx, kv_idx):
            # Ensure indices are within bounds
            q_idx = torch.clamp(q_idx, 0, seq_len - 1)
            kv_idx = torch.clamp(kv_idx, 0, seq_len - 1)
            # Check if kv_idx is among the nearest neighbors of q_idx
            return torch.any(
                nearest_voxels[b, q_idx] == kv_idx.unsqueeze(-1),
                dim=-1
            )
        return create_block_mask(
            _neighborhood_mask,
            B=batch_size,
            H=self.num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=nearest_voxels.device
        )

    def forward(self, x, coords):
        # x: [batch_size, seq_len, feature_dim]
        # coords: [batch_size, seq_len, coord_dim]
        batch_size, seq_len, _ = x.shape
        
        # Compute nearest neighbors
        nearest_voxels = self._compute_nearest_neighbors(coords)
        # Create the attention mask
        attention_mask = self._create_attention_mask(nearest_voxels)
        
        # Project input to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, self.num_heads,seq_len,  self.head_dim).transpose(1, 2)
        key = key.view(batch_size, self.num_heads,seq_len, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, self.num_heads,seq_len, self.head_dim).transpose(1, 2)
        
        # Apply flex attention
        output = flex_attention(query, key, value, block_mask=attention_mask)

        # Compute the metric: the mean of the key vectors over the heads
        metric = key.mean(2)  # Averaging over heads
        
        # Reshape output and return
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim), metric