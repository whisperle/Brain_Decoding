import torch
import torch.nn as nn
from functools import lru_cache
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

class NearestNeighborAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, num_neighbors, full_attention=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.num_neighbors = num_neighbors
        self.dtype = torch.bfloat16
        self.scale = self.head_dim ** -0.5
        # Initialize projection layers
        self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.full_attention = full_attention

    @lru_cache
    def _compute_nearest_neighbors(self, voxel_coords):
        batch_size, seq_len, coord_dim = voxel_coords.shape
        device = voxel_coords.device
        x = voxel_coords  # (batch_size, seq_len, coord_dim)
        # Compute squared norms
        x_norm = (x ** 2).sum(-1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        # Compute pairwise squared distances
        dists_sq = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        # Exclude self by setting diagonal to large value
        diag_inf = torch.full((seq_len,), float('inf'), device=device)
        dists_sq += torch.diag(diag_inf).unsqueeze(0)
        # Get top k indices
        _, top_k_indices = torch.topk(dists_sq, k=self.num_neighbors, dim=-1, largest=False)  # (batch_size, seq_len, num_neighbors)
        return top_k_indices

    def forward(self, x, coords, **kwargs):
        with torch.amp.autocast('cuda'):
            batch_size, seq_len, _ = x.shape
            query = self.query_proj(x)
            key = self.key_proj(x)
            value = self.value_proj(x)
            # Reshape and transpose to get shapes: (batch_size, num_heads, seq_len, head_dim)
            query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            if not self.full_attention:
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    output = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
                # TODO: Implement the rest of the code for nearest neighbor attention
                
            else:
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    output = F.scaled_dot_product_attention(query, key, value, scale=self.scale)

            metric = key.mean(1)  # Averaging over heads
            return output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim), metric