"""
    MoCo v3 architecture for BrainNAT
    Author: Steven Zhang
"""
import torch
import torch.nn as nn
import os
import torch.distributed as dist
from backbone import BrainNAT

class MoCo_attention(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim= 6724* 64, T=1.0, device='cuda'):
        """
        dim: input feature dimension (see from 1D)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo_attention, self).__init__()
        self.T = T
        # build encoders & momentum encoder (same structure)
        self.base_encoder = BrainNAT(
            in_chans=1,
            embed_dim=64,
            # pos_embed_dim=pos_embed_dim,
            depth=9,
            num_heads=8,
            num_neighbors=8,
            tome_r=1000,
            layer_scale_init_value=1e-6,
            coord_dim=3,
            omega_0=30,
            last_n_features=16,  # Use the last 16 features for neighborhood attention
            full_attention=True
        ).to(device)
        self.momentum_encoder = BrainNAT(
            in_chans=1,
            embed_dim=64,
            # pos_embed_dim=pos_embed_dim,
            depth=9,
            num_heads=8,
            num_neighbors=8,
            tome_r=1000,
            layer_scale_init_value=1e-6,
            coord_dim=3,
            omega_0=30,
            last_n_features=16,  # Use the last 16 features for neighborhood attention
            full_attention=True
        ).to(device)
        # predictor(MLP), consider not using when backbone output dim is large
        self.predictor = self._build_mlp(2, dim, 4* 2928, dim, False)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, v1, c1, v2, c2, m):
        """
        Input:
            v1,c1: first views of data
            v2,c2: second views of data
            m: moco momentum
        Output:
            loss
        """
        # query encoder
        q1 = self.base_encoder(v1, c1)
        q2 = self.base_encoder(v2, c2)
        dim0, dim1, dim2 = q1.shape
        q1 = q1.view(dim0, dim1 * dim2)
        q2 = q2.view(dim0, dim1 * dim2)
        # print("q1 shape"+ str(q1.shape))
        # q & k shape: [b, 6724, 64]

        # compute features through mlp layers (deleted in this version because backbone's output is too large)
        # q1 = self.predictor(q1)
        # q2 = self.predictor(q2)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1 = self.momentum_encoder(v1, c1)
            k2 = self.momentum_encoder(v2, c2)
            k1 = k1.view(dim0, dim1 * dim2)
            k2 = k2.view(dim0, dim1 * dim2)
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    # initialize the default process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model = MoCo_attention()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    v1 = torch.randn(9, 1, 15724).cuda()
    c1 = torch.randn(9, 15724, 3).cuda()

    v2 = torch.randn(9, 1, 15724).cuda()
    c2 = torch.randn(9, 15724, 3).cuda()
    loss = model(v1, c1, v2, c2, 1)
    print(loss)