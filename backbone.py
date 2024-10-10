import torch
import torch.nn as nn
from timm.models.layers import DropPath
from atten_flex_customize import NearestNeighborAttention
from tome_customize import TokenMerging

class ConvTokenizer1D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=1, stride=1, padding=0),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BrainNATLayer(nn.Module):
    def __init__(self, dim, num_heads, num_neighbors, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, tome_r=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = NearestNeighborAttention(dim, num_heads, num_neighbors)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # Token merging module
        self.token_merging = TokenMerging(r=tome_r)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, visual_cortex_mask):
        x_attn, metric = self.attn(self.norm1(x), visual_cortex_mask)
        x = x + self.drop_path(x_attn)

        # Token merging module
        x = self.token_merging(x, metric)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BrainNATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, num_neighbors, mlp_ratio=4., qkv_bias=True, drop=0., 
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,tome_r=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            BrainNATLayer(
                dim=dim, num_heads=num_heads, num_neighbors=num_neighbors,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, tome_r=tome_r)
            for i in range(depth)])

    def forward(self, x, visual_cortex_mask):
        for blk in self.blocks:
            x = blk(x, visual_cortex_mask)
        return x

class BrainNAT(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, depth=4, num_heads=8, num_neighbors=5, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm,tome_r=0):
        super().__init__()
        # Ensure embed_dim is a power of 2
        self.embed_dim = 2 ** int(torch.log2(torch.tensor(embed_dim)).ceil().item())
        self.num_features = self.embed_dim

        self.embed_layer = ConvTokenizer1D(in_chans=in_chans, embed_dim=self.embed_dim, norm_layer=norm_layer)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = BrainNATBlock(
            dim=self.embed_dim, depth=depth, num_heads=num_heads, num_neighbors=num_neighbors,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, tome_r=tome_r)

        self.norm = norm_layer(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, visual_cortex_mask):
        x = self.embed_layer(x)
        x = self.pos_drop(x)
        x = self.blocks(x, visual_cortex_mask)
        x = self.norm(x)
        return x

    def forward(self, x, visual_cortex_mask):
        x = self.forward_features(x, visual_cortex_mask)
        x = self.head(x)
        return x

# Example usage
if __name__ == "__main__":
    import numpy as np
    import nibabel as nib
    
    # Set up example data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subj = 1
    fmri_scan = nib.load(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj0{subj}/func1pt8mm/valid_session01.nii.gz')
    nsdgeneral_roi_mask = nib.load(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj0{subj}/func1pt8mm/roi/nsdgeneral.nii.gz').get_fdata() == 1
    input_scan = fmri_scan.get_fdata()[nsdgeneral_roi_mask].reshape(-1,1)
    num_voxels = np.prod(fmri_scan.shape)
    
    visual_cortex_mask = torch.tensor(nsdgeneral_roi_mask, dtype=torch.bool, device=device)
    
    # Initialize the model
    model = BrainNAT(
        in_chans=1,
        embed_dim=128,
        depth=1,
        num_heads=8,
        num_neighbors=3,
        tome_r=2000,
    ).to(device)
    
    model.train()
    # Create dummy input
    batch_size = 2
    sequence_length = visual_cortex_mask.sum().item()
    print("Sequence length:", sequence_length)
    x = torch.randn(batch_size, 1, sequence_length, device=device)
    print("Input shape:", x.shape)
    
    # Forward pass
    output = model(x, visual_cortex_mask)
    print("Output shape:", output.shape)
    
    # Create a virtual target
    y = torch.randn(output.shape, device=device)
    
    # Compute the loss
    criterion = nn.MSELoss()
    loss = criterion(output, y)
    print("Loss:", loss.item())
    loss.backward()