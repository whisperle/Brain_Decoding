import torch
import torch.nn as nn
from functools import lru_cache
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention import SDPBackend, sdpa_kernel

# flex_attention = torch.compile(flex_attention)
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
        nearest_voxels = torch.zeros(batch_size, seq_len, self.num_neighbors, dtype=torch.int, device=device)

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
        print("nearest_voxels shape:", nearest_voxels.shape)
        print("nearest_voxels:", nearest_voxels)
        print("seq_len:", seq_len)
        def _neighborhood_mask(b, h, q_idx, kv_idx):
            print("b:", b)
            # import pdb; pdb.set_trace()
            print("nearest_voxels[b, q_idx]:", nearest_voxels[b, q_idx])
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
            
            if not self.full_attention:
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
            
            # Apply attention based on full_attention flag
            if self.full_attention:
                # Use PyTorch's scaled_dot_product_attention function
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    output = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value,
                        scale=self.scale
                    )
            else:
                # Apply flex attention for sparse attention
                output = flex_attention(query, key, value, block_mask=attention_mask)
                
            # Compute the metric: the mean of the key vectors over the heads
            metric = key.mean(1)  # Averaging over heads
        
            # Reshape output and return
            return output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim), metric

# Test case
if __name__ == "__main__":
    import pdb;
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define test parameters
    batch_size = 2
    seq_len = 16
    feature_dim = 64
    coord_dim = 3
    num_heads = 8
    num_neighbors = 4
    
    # Create test inputs
    x = torch.randn(batch_size, seq_len, feature_dim).cuda()
    coords = torch.randn(batch_size, seq_len, coord_dim).cuda()
    
    # Initialize model
    model = NearestNeighborAttention(
        feature_dim=feature_dim,
        num_heads=num_heads,
        num_neighbors=num_neighbors
    ).cuda()
    
    # Forward pass
    try:
        output, metric = model(x, coords)
        
        # Print shapes for verification
        print("\nTest Results:")
        print(f"Input shape: {x.shape}")
        print(f"Coordinates shape: {coords.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Metric shape: {metric.shape}")
        
        # Basic assertions
        assert output.shape == (batch_size, seq_len, feature_dim), "Output shape mismatch"
        assert metric.shape == (batch_size, seq_len, feature_dim // num_heads), "Metric shape mismatch"
        print("\nAll shape tests passed!")
        
        # Check if output contains valid values
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"
        print("All value tests passed!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
