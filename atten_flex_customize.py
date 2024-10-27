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

    def _compute_nearest_neighbors(self, voxel_coords):
        """
        Efficiently compute the nearest neighbors for each voxel using iterative distance computation.
        
        Args:
            voxel_coords (torch.Tensor): Tensor of shape [batch_size, seq_len, coord_dim]
            
        Returns:
            torch.Tensor: Nearest neighbors indices, shape [batch_size, seq_len, num_neighbors]
        """
        batch_size, seq_len, coord_dim = voxel_coords.shape
        device = voxel_coords.device
        
        # Initialize tensor to store nearest neighbors
        nearest_voxels = torch.zeros(batch_size, seq_len, self.num_neighbors, dtype=torch.long, device=device)

        # Iterate over each voxel in the sequence
        for i in range(seq_len):
            # Extract the current voxel's coordinates
            voxel_i_coords = voxel_coords[:, i, :]  # [batch_size, coord_dim]
            
            # Compute distances between the current voxel and all other voxels
            dists = torch.norm(voxel_coords - voxel_i_coords.unsqueeze(1), dim=-1)  # [batch_size, seq_len]
            
            # Get the indices of the top n nearest neighbors (excluding the voxel itself)
            _, top_k_indices = torch.topk(dists, k=self.num_neighbors+1, dim=-1, largest=False)  # [batch_size, num_neighbors+1]
            
            # Store the nearest neighbors (excluding the voxel itself, which is always the first one)
            nearest_voxels[:, i, :] = top_k_indices[:, 1:self.num_neighbors+1]  # Exclude self (first index)

        return nearest_voxels

    def _create_attention_mask(self, nearest_voxels):
        # nearest_voxels: [batch_size, seq_len, num_neighbors]
        batch_size, seq_len, _ = nearest_voxels.shape

        def _neighborhood_mask(b, h, q_idx, kv_idx):
            # Ensure indices are within bounds
            q_idx = torch.clamp(q_idx, 0, seq_len - 1)
            kv_idx = torch.clamp(kv_idx, 0, seq_len - 1)
            return torch.any(nearest_voxels[b, q_idx] == kv_idx.unsqueeze(-1), dim=-1)

        return create_block_mask(
            _neighborhood_mask,
            B=batch_size,
            H=self.num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=nearest_voxels.device
        )

    def forward(self, x, coords,**kwargs):
        with torch.amp.autocast('cuda'):
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
            query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Apply flex attention
            output = flex_attention(query, key, value, block_mask=attention_mask)
            # Compute the metric: the mean of the key vectors over the heads
            metric = key.mean(1)  # Averaging over heads
        
            # Reshape output and return
            return output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim), metric
