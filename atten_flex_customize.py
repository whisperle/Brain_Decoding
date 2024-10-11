import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

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

    def _compute_nearest_neighbors(self, voxel_coords, lens):
        """
        Efficiently compute the nearest neighbors for each voxel using broadcasting and matrix operations,
        taking sequence lengths into account.
        
        Args:
            voxel_coords (torch.Tensor): Tensor of shape [batch_size, seq_len, coord_dim]
            lens (torch.Tensor): Tensor of shape [batch_size], containing lengths of each sequence
            
        Returns:
            torch.Tensor: Nearest neighbors indices, shape [batch_size, seq_len, num_neighbors]
        """
        batch_size, seq_len, _ = voxel_coords.shape

        # Mask for valid sequences based on lengths
        valid_mask = torch.arange(seq_len, device=lens.device).expand(batch_size, seq_len) < lens.unsqueeze(1)
        
        # Set out-of-bound coordinates to a large value to exclude them from nearest neighbor calculations
        masked_coords = voxel_coords.clone()
        masked_coords[~valid_mask] = float('inf')

        # Compute pairwise distances in a batched manner
        voxel_coords_expanded = masked_coords.unsqueeze(1)  # [batch_size, 1, seq_len, coord_dim]
        voxel_coords_repeated = masked_coords.unsqueeze(2)  # [batch_size, seq_len, 1, coord_dim]

        # Compute squared distances
        dists = torch.norm(voxel_coords_expanded - voxel_coords_repeated, dim=-1)  # [batch_size, seq_len, seq_len]

        # Sort the distances to find the nearest neighbors
        # The first closest point is always the point itself, so we exclude it by taking indices [1:num_neighbors+1]
        nearest_voxels = torch.argsort(dists, dim=-1)[:, :, 1:self.num_neighbors+1]  # [batch_size, seq_len, num_neighbors]

        return nearest_voxels

    def _create_attention_mask(self, nearest_voxels, lens):
        # nearest_voxels: [batch_size, seq_len, num_neighbors]
        batch_size, seq_len, _ = nearest_voxels.shape
        
        # Create a mask for the valid lengths
        valid_mask = torch.arange(seq_len, device=lens.device).expand(batch_size, seq_len) < lens.unsqueeze(1)

        def _neighborhood_mask(b, h, q_idx, kv_idx):
            # Ensure indices are within bounds and valid length
            q_idx = torch.clamp(q_idx, 0, seq_len - 1)
            kv_idx = torch.clamp(kv_idx, 0, seq_len - 1)
            valid_kv_mask = valid_mask[b, kv_idx]  # Only allow attention to valid indices
            return valid_kv_mask & torch.any(nearest_voxels[b, q_idx] == kv_idx.unsqueeze(-1), dim=-1)

        return create_block_mask(
            _neighborhood_mask,
            B=batch_size,
            H=self.num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=nearest_voxels.device
        )

    def forward(self, x, coords, lens):
        # x: [batch_size, seq_len, feature_dim]
        # coords: [batch_size, seq_len, coord_dim]
        # lens: [batch_size], the lengths of the valid sequences

        batch_size, seq_len, _ = x.shape
        
        # Compute nearest neighbors, respecting the valid sequence lengths
        nearest_voxels = self._compute_nearest_neighbors(coords, lens)

        # Create the attention mask
        attention_mask = self._create_attention_mask(nearest_voxels, lens)
        
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