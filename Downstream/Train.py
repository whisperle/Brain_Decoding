import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
import kornia
from kornia.augmentation.container import AugmentationSequential
# Add the path for SDXL unCLIP requirements
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # bigG embedder

# Enable tf32 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True

# Custom utility functions
import utils
from utils import save_ckpt

import re

from IPython import embed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="Name of the model, used for checkpoint saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--data_path", type=str, default='/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2',#os.getcwd(),
        help="Path to where NSD data is stored or where to download it",
    )
    parser.add_argument(
        "--cache_dir", type=str, default='/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2',#os.getcwd(),
        help="Path to where miscellaneous files downloaded from huggingface are stored. Defaults to current directory.",
    )
    #TODO: We gonna validate on all the subjects in the ideal case since we are doing multi-subject stuff.
    # held-out subject
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--multisubject_ckpt", type=str, default=None,
        help="Path to pre-trained multisubject model to finetune a single subject from. multisubject must be False.",
    )
    parser.add_argument(
        "--num_sessions", type=int, default=1,
        help="Number of training sessions to include",
    )
    parser.add_argument(
        "--use_prior", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size can be increased by 10x if only training retrieval submodule and not diffusion prior",
    )
    parser.add_argument(
        "--wandb_log", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to log to wandb",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="stability",
        help="wandb project name",
    )
    parser.add_argument(
        "--mixup_pct", type=float, default=.33,
        help="Proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--blurry_recon", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--blur_scale", type=float, default=.5,
        help="Multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--clip_scale", type=float, default=1.,
        help="Multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--prior_scale", type=float, default=30,
        help="Multiply diffusion prior loss by this",
    )
    parser.add_argument(
        "--use_image_aug", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to use image augmentation",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=150,
        help="Number of epochs of training",
    )
    parser.add_argument(
        "--multi_subject", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to use multi-subject training",
    )
    parser.add_argument(
        "--new_test", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to use the new test set",
    )
    parser.add_argument(
        "--n_blocks", type=int, default=4,
        help="Number of blocks in the model",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=1024,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'],
        help="Type of learning rate scheduler",
    )
    parser.add_argument(
        "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to save checkpoints",
    )
    parser.add_argument(
        "--ckpt_interval", type=int, default=5,
        help="Save backup checkpoint and reconstruct every x epochs",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4,
        help="Maximum learning rate",
    )
    args = parser.parse_args()
    return args



def prepare_data(args, data_type):
    # TODO: nsdgeneral masks are also needed for each subject. We need them as the guidance for the NAT.
    def my_split_by_node(urls): return urls
    num_voxels_list = []

    if args.multi_subject:
        subj_list = np.arange(1,9)
        subj_list = subj_list[subj_list != subj]
        nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
        num_samples_per_epoch = (750 * 40 * 7) // args.batch_size
    else:
        subj_list = [args.subj]
        num_samples_per_epoch = (750 * args.num_sessions) // args.batch_size

    num_iterations_per_epoch = num_samples_per_epoch // args.batch_size
    print("batch_size =", args.batch_size, "num_iterations_per_epoch =", num_iterations_per_epoch,
        "num_samples_per_epoch =", num_samples_per_epoch)

    train_data_list = []
    num_voxels = {}
    voxels = {}

    if args.multi_subject:
        train_urls = [
            f"{args.data_path}/wds/subj0{s}/train/{i}.tar"
            for s in subj_list
            for i in range(nsessions_allsubj[s - 1])
        ]
        random.shuffle(train_urls)
    else:
        train_urls = f"{args.data_path}/wds/subj0{args.subj}/train/" + "{0.." + f"{args.num_sessions - 1}" + "}.tar"

    subject_pattern = re.compile(r"/subj0(\d+)/")

    def add_subject(sample):
        match = subject_pattern.search(sample["__url__"])
        if match:
            sample["subject_id"] = int(match.group(1))
        return sample

    train_data = (
        wds.WebDataset(train_urls, resampled=False, nodesplitter=my_split_by_node)
        .shuffle(1500, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .map(add_subject)
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav", "subject_id")
    )

    train_dl = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    for s in subj_list:
        f = h5py.File(f'{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
        betas = f['betas'][:]
        betas = torch.Tensor(betas).to("cpu").to(data_type)
        num_voxels_list.append(betas[0].shape[-1])
        num_voxels[f'subj0{s}'] = betas[0].shape[-1]
        voxels[f'subj0{s}'] = betas
        print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

    print("Loaded all subject train loaders and betas!\n")

    # Validate only on one subject
    subj = args.subj
    if not args.new_test:  # using old test set
        num_test_mapping = {3: 2113, 4: 1985, 6: 2113, 8: 1985}
        num_test = num_test_mapping.get(subj, 2770)
        test_url = f"{args.data_path}/wds/subj0{subj}/test/" + "0.tar"
    else:  # using larger test set
        num_test_mapping = {3: 2371, 4: 2188, 6: 2371, 8: 2188}
        num_test = num_test_mapping.get(subj, 3000)
        test_url = f"{args.data_path}/wds/subj0{subj}/new_test/" + "0.tar"

    print(test_url)
    test_data = (
        wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
        .shuffle(1500, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .map(add_subject)
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav", "subject_id")
    )
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True,
                                        pin_memory=True)
    print(f"Loaded test loader for subj0{subj}!\n")

    # Load all 73k NSD images
    f = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
    images = f['images']
    print("Loaded all 73k possible NSD images to CPU!", images.shape)

    return train_dl, test_dl, images, num_voxels_list, num_iterations_per_epoch, num_samples_per_epoch, num_test, voxels, num_voxels




def build_model(args, device, data_type, num_voxels_list):
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)

    clip_seq_dim = 256
    clip_emb_dim = 1664

    if args.blurry_recon:
        from diffusers import AutoencoderKL
        autoenc = AutoencoderKL(
            down_block_types=['DownEncoderBlock2D'] * 4,
            up_block_types=['UpDecoderBlock2D'] * 4,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            sample_size=256,
        )
        ckpt = torch.load(f'{args.cache_dir}/sd_image_var_autoenc.pth')
        autoenc.load_state_dict(ckpt)
        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(device)

        from autoencoder.convnext import ConvnextXL
        cnx = ConvnextXL(f'{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
        cnx.requires_grad_(False)
        cnx.eval()
        cnx.to(device)

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1, 3, 1, 1)

        blur_augs = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.1),
            kornia.augmentation.RandomSolarize(p=0.1),
            kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
            data_keys=["input"],
        )
    else:
        autoenc = None
        cnx = None
        mean = None
        std = None
        blur_augs = None

    class MindEyeModule(nn.Module):
        def __init__(self):
            super(MindEyeModule, self).__init__()

        def forward(self, x):
            return x

    model = MindEyeModule()
    # ------------------ Backbone Part ------------------
    # TODO: Replace RidgeRegression+MLP to our Backbone
    class RidgeRegression(torch.nn.Module):
        # Make sure to add weight_decay when initializing optimizer to enable regularization
        def __init__(self, input_sizes, out_features):
            super(RidgeRegression, self).__init__()
            self.out_features = out_features
            self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])

        def forward(self, x, subj_idx):
            out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
            return out

    hidden_dim = args.hidden_dim
    model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)
    utils.count_params(model.ridge)
    utils.count_params(model)

    from models import BrainNetwork
    model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=args.n_blocks,
                                  clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
                                  blurry_recon=args.blurry_recon, clip_scale=args.clip_scale)
    utils.count_params(model.backbone)
    utils.count_params(model)
    # ------------------ Prior Part ------------------
    if args.use_prior:
        from models import PriorNetwork, BrainDiffusionPrior
        out_dim = clip_emb_dim
        depth = 6
        dim_head = 52
        heads = clip_emb_dim // 52
        timesteps = 100

        prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens=clip_seq_dim,
            learned_query_mode="pos_emb"
        )

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

        utils.count_params(model.diffusion_prior)
        utils.count_params(model)
    else:
        model.diffusion_prior = None

    return clip_img_embedder, model, autoenc, cnx, mean, std, blur_augs

def setup_optimizer(args, model, num_iterations_per_epoch):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    max_lr = args.max_lr

    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    if args.use_prior:
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ])

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
    return optimizer, lr_scheduler


def setup_wandb(args, num_params, train_url, test_url):
    local_rank = os.getenv('RANK', 0)
    wandb_log = args.wandb_log

    if int(local_rank) == 0 and wandb_log:
        import wandb
        print(f"wandb {args.wandb_project} run {args.model_name}")
        wandb_config = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "num_sessions": args.num_sessions,
            "num_params": num_params,
            "clip_scale": args.clip_scale,
            "prior_scale": args.prior_scale,
            "blur_scale": args.blur_scale,
            "use_image_aug": args.use_image_aug,
            "max_lr": args.max_lr,
            "mixup_pct": args.mixup_pct,
            "ckpt_interval": args.ckpt_interval,
            "ckpt_saving": args.ckpt_saving,
            "seed": args.seed,
            "train_url": train_url,
            "test_url": test_url,
        }
        wandb.init(
            id=args.model_name,
            project=args.wandb_project,
            name=args.model_name,
            config=wandb_config,
            resume="allow",
        )
    else:
        wandb_log = False
    return wandb_log


def train(args, model, train_dl, test_dl, images, accelerator, device, data_type, num_iterations_per_epoch,
          num_test, subj_list, clip_img_embedder, optimizer, lr_scheduler, wandb_log, autoenc, cnx, mean, std,
          blur_augs, local_rank, voxels, num_voxels):
    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test_loss = 1e9
    torch.cuda.empty_cache()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    ckpt_interval = args.ckpt_interval
    ckpt_saving = args.ckpt_saving
    mixup_pct = args.mixup_pct
    blur_scale = args.blur_scale
    clip_scale = args.clip_scale
    prior_scale = args.prior_scale
    use_image_aug = args.use_image_aug
    blurry_recon = args.blurry_recon
    use_prior = args.use_prior
    model_name = args.model_name

    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)

    print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
    progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
    test_image, test_voxel = None, None
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    for epoch in progress_bar:
        model.train()

        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        test_fwd_percent_correct = 0.
        test_bwd_percent_correct = 0.
        
        recon_cossim = 0.
        test_recon_cossim = 0.
        recon_mse = 0.
        test_recon_mse = 0.

        loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_blurry_cont_total = 0.
        test_loss_clip_total = 0.
        
        loss_prior_total = 0.
        test_loss_prior_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

        # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
        voxel_iters = [] # empty dict because diff subjects have differing # of voxels
        image_iters = torch.zeros(num_iterations_per_epoch, batch_size, 3, 224, 224).float()
        annot_iters = []
        perm_iters, betas_iters, select_iters = [], [], []
        subj_idx_iters = []
        with torch.cuda.amp.autocast(dtype=data_type):
            iter = -1
            for behav0, past_behav0, future_behav0, old_behav0, subj_idx in train_dl: 
                # Load images to cpu from hdf5 (requires sorted indexing)
                image_idx = behav0[:,0,0].cpu().long().numpy()
                image0, image_sorted_idx = np.unique(image_idx, return_index=True)                
                if len(image0) != len(image_idx): # hdf5 cant handle duplicate indexing
                    continue
                iter += 1
                image0 = torch.tensor(images[image0], dtype=data_type)
                image_iters[iter] = image0
                
                # Load voxels for current batch, matching above indexing
                voxel_idx = behav0[:,0,5].cpu().long().numpy()
                voxel_sorted_idx = voxel_idx[image_sorted_idx]

                subj_idx = subj_idx.cpu().long().numpy()
                subj_sorted_idx = subj_idx[image_sorted_idx]
                subj_idx_iters.append(subj_sorted_idx)

                max_voxel_len = np.max([num_voxels[f'subj0{s}'] for s in np.unique(subj_idx)])
                voxel0 = torch.zeros((batch_size,1,max_voxel_len), dtype=data_type)

                for i,s in enumerate(subj_sorted_idx):
                    voxel = voxels[f'subj0{s}'][voxel_sorted_idx[i]]
                    voxel0[i,0,:len(voxel)] = voxel

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    perm_iters.append(perm)
                    betas_iters.append(betas)
                    select_iters.append(select)

                voxel_iters.append(voxel0)

                if iter >= num_iterations_per_epoch-1:
                    break

        # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
        for train_i in range(num_iterations_per_epoch):
            with torch.cuda.amp.autocast(dtype=data_type):
                optimizer.zero_grad()
                loss=0.

                voxel = voxel_iters[train_i].to(device)
                image = image_iters[train_i].detach().to(device)
                subj_idx = subj_idx_iters[train_i]

                if use_image_aug: 
                    image = img_augment(image)

                clip_target = clip_img_embedder(image)
                assert not torch.any(torch.isnan(clip_target))

                if epoch < int(mixup_pct * num_epochs):
                    perm = perm_iters[train_i].detach().to(device)
                    betas = betas_iters[train_i].detach().to(device)
                    select = select_iters[train_i].detach().to(device)

                # TODO: replace model
                # voxel_nat = model(voxel, subj_idx)
                
                voxel_ridge = torch.zeros((batch_size,1,args.hidden_dim)).float().to(device)

                backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)
                if clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if use_prior:
                    loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                    loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior

                    recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                    recon_mse += mse(prior_out, clip_target).item()

                if clip_scale>0:
                    if epoch < int(mixup_pct * num_epochs):                
                        loss_clip = utils.mixco_nce(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006,
                            perm=perm, betas=betas, select=select)
                    else:
                        epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=epoch_temp)

                    loss_clip_total += loss_clip.item()
                    loss_clip *= clip_scale
                    loss += loss_clip

                if blurry_recon:     
                    image_enc_pred, transformer_feats = blurry_image_enc_

                    image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_total += loss_blurry.item()

                    if epoch < int(mixup_pct * num_epochs):
                        image_enc_shuf = image_enc[perm]
                        betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                        image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                            image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                    image_norm = (image - mean)/std
                    image_aug = (blur_augs(image) - mean)/std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)

                    cont_loss = utils.soft_cont_loss(
                        nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        temp=0.2)
                    loss_blurry_cont_total += cont_loss.item()

                    loss += (loss_blurry + 0.1*cont_loss) * blur_scale #/.18215

                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                    fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                if blurry_recon:
                    with torch.no_grad():
                        # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                        random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        blurry_pixcorr += pixcorr.item()

                utils.check_loss(loss)
                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()

        model.eval()
        if local_rank==0:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
                for test_i, (behav, past_behav, future_behav, old_behav, subj_idx) in enumerate(test_dl):  
                    # all test samples should be loaded per batch such that test_i should never exceed 0
                    assert len(behav) == num_test

                    ## Average same-image repeats ##
                    if test_image is None:
                        max_voxel_len = np.max([num_voxels[f'subj0{s}'] for s in np.unique(subj_idx)])
                        voxel = torch.zeros((num_test,1,max_voxel_len), dtype=data_type)
                        for i,s in enumerate(subj_idx):
                            voxel_ = voxels[f'subj0{s}'][behav[:,0,5].cpu().long()[i]]
                            voxel[i,0,:len(voxel_)] = voxel_
                        # voxel = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()].unsqueeze(1)
                        
                        image = behav[:,0,0].cpu().long()

                        unique_image, sort_indices = torch.unique(image, return_inverse=True)
                        for im in unique_image:
                            locs = torch.where(im == image)[0]
                            if len(locs)==1:
                                locs = locs.repeat(3)
                            elif len(locs)==2:
                                locs = locs.repeat(2)[:3]
                            assert len(locs)==3
                            if test_image is None:
                                test_image = torch.Tensor(images[im][None])
                                test_voxel = voxel[locs][None]
                            else:
                                test_image = torch.vstack((test_image, torch.Tensor(images[im][None])))
                                test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                    loss=0.
                                
                    test_indices = torch.arange(len(test_voxel))[:300]
                    voxel = test_voxel[test_indices].to(device)
                    image = test_image[test_indices].to(device)
                    assert len(image) == 300

                    clip_target = clip_img_embedder(image.float())

                    for rep in range(3):
                        # TODO: replace model
                        voxel_ridge = torch.zeros((num_test,1,args.hidden_dim)).float().to(device)
                        
                        backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
                        if rep==0:
                            clip_voxels = clip_voxels0
                            backbone = backbone0
                        else:
                            clip_voxels += clip_voxels0
                            backbone += backbone0
                    clip_voxels /= 3
                    backbone /= 3

                    if clip_scale>0:
                        clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                    
                    # for some evals, only doing a subset of the samples per batch because of computational cost
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                    
                    if use_prior:
                        loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                        test_loss_prior_total += loss_prior.item()
                        loss_prior *= prior_scale
                        loss += loss_prior
                        
                    if clip_scale>0:
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006)

                        test_loss_clip_total += loss_clip.item()
                        loss_clip = loss_clip * clip_scale
                        loss += loss_clip

                    if blurry_recon:
                        image_enc_pred, _ = blurry_image_enc_
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        test_blurry_pixcorr += pixcorr.item()

                    if clip_scale>0:
                        # forward and backward top 1 accuracy        
                        labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                        test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                        test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                    
                    utils.check_loss(loss)                
                    test_losses.append(loss.item())

                assert (test_i+1) == 1
                logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "test/loss": np.mean(test_losses[-(test_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "test/num_steps": len(test_losses),
                    "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                    "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                    "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                    "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                    "train/loss_clip_total": loss_clip_total / (train_i + 1),
                    "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                    "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                    "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                    "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                    "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                    "train/recon_cossim": recon_cossim / (train_i + 1),
                    "test/recon_cossim": test_recon_cossim / (test_i + 1),
                    "train/recon_mse": recon_mse / (train_i + 1),
                    "test/recon_mse": test_recon_mse / (test_i + 1),
                    "train/loss_prior": loss_prior_total / (train_i + 1),
                    "test/loss_prior": test_loss_prior_total / (test_i + 1),
                    }

                # if finished training, save jpg recons if they exist
                if (epoch == num_epochs-1) or (epoch % ckpt_interval == 0):
                    if blurry_recon:    
                        image_enc = autoenc.encode(2*image[:4]-1).latent_dist.mode() * 0.18215
                        # transform blurry recon latents to images and plot it
                        fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                        jj=-1
                        for j in [0,1,2,3]:
                            jj+=1
                            axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                            axes[jj].axis('off')
                            jj+=1
                            axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc_pred[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                            axes[jj].axis('off')

                        if wandb_log:
                            logs[f"test/blur_recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                            plt.close()
                        else:
                            plt.show()

                progress_bar.set_postfix(**logs)

                if wandb_log: wandb.log(logs)
                
        # Save model checkpoint and reconstruct
        if (ckpt_saving) and (epoch % ckpt_interval == 0):
            save_ckpt(f'last',args,model,optimizer,lr_scheduler,epoch, losses, test_losses, lrs, accelerator, ckpt_saving=True)

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

    print("\n===Finished!===\n")
    if ckpt_saving:
        save_ckpt(f'last',args,model,optimizer,lr_scheduler,epoch, losses, test_losses, lrs, accelerator, ckpt_saving=True)

def main():
    args = parse_arguments()
    utils.seed_everything(args.seed)
    data_type = torch.float16  # Change depending on your mixed_precision

    local_rank = int(os.getenv('RANK', 0))
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        num_devices = 1
    args.num_devices = num_devices
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device

    # if utils.is_interactive():
    #     batch_size = args.batch_size = 8
    # else:
    batch_size = args.batch_size

    if args.use_image_aug or args.blurry_recon:
        import kornia
        from kornia.augmentation.container import AugmentationSequential

    if args.use_image_aug:
        img_augment = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
            same_on_batch=False,
            data_keys=["input"],
        )
    else:
        img_augment = None

    train_dl, test_dl, images, num_voxels_list, num_iterations_per_epoch, num_samples_per_epoch, num_test, voxels, num_voxels = prepare_data(
        args, data_type)

    clip_img_embedder, model, autoenc, cnx, mean, std, blur_augs = build_model(args, device, data_type,
                                                                               num_voxels_list)

    optimizer, lr_scheduler = setup_optimizer(args, model, num_iterations_per_epoch)

    num_params = utils.count_params(model)

    wandb_log = setup_wandb(args, num_params, train_url="", test_url="")  # Update with actual URLs if needed

    train(args, model, train_dl, test_dl, images, accelerator, device, data_type, num_iterations_per_epoch,
          num_test, [args.subj], clip_img_embedder, optimizer, lr_scheduler, wandb_log, autoenc, cnx, mean, std,
          blur_augs, local_rank, voxels, num_voxels)


if __name__ == "__main__":
    main()