import sys
sys.path.append('/scratch/cc6946/projects/Brain_Decoding/Downstream')
sys.path.append('/scratch/cc6946/projects/Brain_Decoding')

import torch
import torch.nn as nn
import argparse
import copy
import numpy as np
import wandb
import os

from backbone import BrainNAT

from dataset import MindEye2Dataset, SubjectBatchSampler, custom_collate_fn

from IPython import embed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="BrainNAT-pre",
        help="Name of the model, used for checkpoint saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--multi_subject",type=lambda x: [int(i) for i in x.split(',')],
        default="1",
        help="List of subjects to use for multi-subject training",
    )
    parser.add_argument(
        "--ema",type=lambda x: [float(i) for i in x.split(',')],
        default="0.998,1"
    )
    parser.add_argument(
        "--data_path", type=str, default='/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2',
        help="Path to where NSD data is stored or where to download it",
    )
    parser.add_argument(
        "--cache_dir", type=str, default='/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2',
        help="Path to where miscellaneous files downloaded from huggingface are stored. Defaults to current directory.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size can be increased by 10x if only training retrieval submodule and not diffusion prior",
    )
    parser.add_argument(
        "--wandb_log", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to log to wandb",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="BRAIN_NAT",
        help="wandb project name",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=150,
        help="Number of epochs of training",
    )
    parser.add_argument(
        "--n_blocks", type=int, default=4,
        help="Number of blocks in the model",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=1024, #todo Try 512
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to save checkpoints",
    )
    parser.add_argument(
        "--ckpt_interval", type=int, default=10,
        help="Save backup checkpoint and reconstruct every x epochs",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="/scratch/cc6946/projects/Brain_Decoding/ckpt"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4,
        help="Number of attention heads in BrainNAT",
    )
    parser.add_argument(
        "--tome_r", type=int, default=2000,
        help="Token merging ratio for BrainNAT",
    )
    parser.add_argument(
        "--last_n_features", type=int, default=16,
        help="Number of features in the last layer of BrainNAT",
    )
    parser.add_argument(
        "--nat_depth", type=int, default=2,
        help="Depth of the BrainNAT model",
    )
    parser.add_argument(
        "--nat_num_neighbors", type=int, default=8,
        help="Number of neighbors in BrainNAT",
    )    
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'],
        help="Type of learning rate scheduler",
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4,
        help="Maximum learning rate",
    )
    parser.add_argument(
        "--ipe_scale", type=float, default=1.0
    )
    parser.add_argument(
        "--num_masks", type=int, default=50
    )
    parser.add_argument(
        "--mask_size", type=int, default=100
    )  
    args = parser.parse_args()
    return args

####################################### Training Setup #######################################
args = parse_arguments()

# -- set device
if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

# build model
hidden_dim_nat = args.hidden_dim//4
encoder = BrainNAT(
    in_chans=1,
    embed_dim=hidden_dim_nat,
    depth=args.nat_depth,
    num_heads=args.num_heads,
    num_neighbors=args.nat_num_neighbors,
    tome_r=args.tome_r,
    layer_scale_init_value=1e-6,
    coord_dim=3,
    omega_0=30,
    last_n_features=args.last_n_features,
    full_attention=True
).to(device)
target_encoder = copy.deepcopy(encoder).to(device)
for p in target_encoder.parameters():
    p.requires_grad = False

# mask token
embed_dim = 2 ** int(torch.log2(torch.tensor(hidden_dim_nat)).ceil().item())
mask_token = nn.Parameter(torch.randn(1, embed_dim, device=device))

# create dataloader
data_type = torch.float16
train_data = MindEye2Dataset(args, data_type, 'train')
train_sampler = SubjectBatchSampler(train_data, args.batch_size)
train_dl = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)

num_iterations_per_epoch = len(train_data) // args.batch_size

# set optimizer
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
max_lr = args.max_lr
opt_grouped_parameters = [
    {'params': [p for n, p in encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-2},
    {'params': [p for n, p in encoder.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0},
    {'params': [mask_token],
            'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if args.lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(args.num_epochs * num_iterations_per_epoch)),
        last_epoch=-1
    )
elif args.lr_scheduler_type == 'cycle':
    total_steps = int(np.floor(args.num_epochs * num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2 / args.num_epochs
    )

# momentum schedule
momentum_scheduler = (args.ema[0] + i*(args.ema[1]-args.ema[0])/(num_iterations_per_epoch*args.num_epochs*args.ipe_scale)
                          for i in range(int(num_iterations_per_epoch*args.num_epochs*args.ipe_scale)+1))

####################################### Saving Checkpoint #######################################
def save_checkpoint(epoch, path):
    save_dict = {
        'encoder': encoder.state_dict(),
        'opt': optimizer.state_dict(),
        'target_encoder': target_encoder.state_dict()
    }
    try:
        torch.save(save_dict, os.path.join(path, f'{args.model_name}_epoch_{epoch}.pth'))
    except Exception as e:
        logger.info(f'Encountered exception when saving checkpoint: {e}')

####################################### Training Loop #######################################
if args.wandb_log:
    os.environ['SSL_CERT_FILE'] = '/scratch/cc6946/cacert.pem'
    wandb.init(
        id=args.model_name,
        project=args.wandb_project,
        name=args.model_name,
        config=vars(args),
        resume="auto",
    )

for epoch in range(args.num_epochs):
    running_loss = 0.0 

    for iteration, (images, voxels, subj_idx, coords, image_idx) in enumerate(train_dl):
        voxels = voxels.to(device)
        coords = coords.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            z = encoder(voxels, coords, True, args.num_masks, args.mask_size, mask_token)
            h = target_encoder(voxels, coords)

            criterion = nn.MSELoss()
            loss = criterion(z, h)
        
        loss.backward()
        optimizer.step()

        if args.lr_scheduler_type:
            lr_scheduler.step()

        m = next(momentum_scheduler)
        with torch.no_grad():
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

        if args.wandb_log:
            wandb.log({"Loss": loss.item()})

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dl)
    if args.wandb_log:
        wandb.log({"Average Loss per Epoch": avg_loss})

    if args.ckpt_saving and (epoch % args.ckpt_interval == 0 or epoch == (args.num_epochs - 1)):
        save_checkpoint(epoch + 1, args.ckpt_path)