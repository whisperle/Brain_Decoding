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
import wandb

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import kornia
from kornia.augmentation.container import AugmentationSequential
# from models import NAT_BrainNe
from attention_decoders import NAT_BrainNet
# Add the path for SDXL unCLIP requirements
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # bigG embedder

sys.path.append('../')
from pretraining.utils import load_checkpoint

# Enable tf32 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True

# Custom utility functions
import utils
from utils import save_ckpt
from dataset import MindEye2Dataset, SubjectBatchSampler, custom_collate_fn
import re
from models import PriorNetwork, BrainDiffusionPrior

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
        "--use_prior", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
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
        "--mixup_pct", type=float, default=.33,
        help="Proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--blurry_recon", action=argparse.BooleanOptionalAction, default=False,
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
        "--multi_subject",type=lambda x: [int(i) for i in x.split(',')],
        default="1,2,5,7",#[1,2,3,4,5,6,7,8],
        help="List of subjects to use for multi-subject training",
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
        "--decoder_hidden_dim", type=int, default=1024, #todo Try 512
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--encoder_hidden_dim", type=int, default=1024, #todo Try 512
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'],
        help="Type of learning rate scheduler",
    )
    parser.add_argument(
        "--ckpt_saving", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to save checkpoints",
    )
    parser.add_argument(
        "--ckpt_interval", type=int, default=5,
        help="Save backup checkpoint and reconstruct every x epochs",
    )
    parser.add_argument(
        "--ckpt_iter", type=int, default=15000,
        help="Save backup checkpoint and reconstruct every x iterations",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4,
        help="Maximum learning rate",
    )
    parser.add_argument(
        "--use_mixer", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to use the mixer",
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
        "--full_attention", action="store_true",
        help="Whether to use full attention in BrainNAT",
    )
    parser.add_argument(
        "--n_blocks_decoder", type=int, default=4,
        help="Number of blocks in the decoder",
    )
    parser.add_argument(
        "--drop", type=float, default=0.1,
        help="Dropout rate for the model",
    )
    parser.add_argument(
        "--progressive_dims", action="store_true",
        help="Whether to use progressive dimension scaling",
    )
    parser.add_argument(
        "--initial_tokens", type=int, default=15000,
        help="Initial number of tokens for progressive dimension scaling",
    )
    parser.add_argument(
        "--dim_scale_factor", type=float, default=1.0,
        help="Power factor for dimension scaling (0.5 = square root scaling)",
    )
    parser.add_argument(
        "--finetuning", action=argparse.BooleanOptionalAction, default=False,
        help="Finetune on pretrain encoder",
    )
    parser.add_argument(
        "--pretrained_ckpt", type=str, default=None,
        help="Path to pre-trained multisubject model to finetune a single subject from. ",
    )
    parser.add_argument(
        "--freeze_pretrained_weights", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to freeze pretrain encoder",
    )
    args = parser.parse_args()
    return args



def prepare_data(args, data_type):
    train_data = MindEye2Dataset(args, data_type, 'train')
    train_sampler = SubjectBatchSampler(train_data, args.batch_size)
    train_dl = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True)

    test_data = MindEye2Dataset(args, data_type, 'test')
    test_sampler = SubjectBatchSampler(test_data, args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_data, batch_sampler=test_sampler, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True)

    num_iterations_per_epoch = len(train_data) // args.batch_size
    return train_dl, test_dl, len(test_data), num_iterations_per_epoch

def build_model(args, device, data_type):
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    ).to(device)
    print("clip_img_embedder")
    utils.count_params(clip_img_embedder)

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
        ).to(device)
        ckpt = torch.load(f'{args.cache_dir}/sd_image_var_autoenc.pth', map_location=device)
        autoenc.load_state_dict(ckpt)
        autoenc.eval()
        autoenc.requires_grad_(False)

        from autoencoder.convnext import ConvnextXL
        cnx = ConvnextXL(f'{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth').to(device)
        cnx.requires_grad_(False)
        cnx.eval()

        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        std = torch.tensor([0.228, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

        blur_augs = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.1),
            kornia.augmentation.RandomSolarize(p=0.1),
            kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
            data_keys=["input"],
        ).to(device)
    else:
        autoenc = None
        cnx = None
        mean = None
        std = None
        blur_augs = None

    model = NAT_BrainNet(args, clip_emb_dim, clip_seq_dim).to(device)

    if args.finetuning:
        load_checkpoint(args.pretrained_ckpt, model.brain_nat, args.freeze_pretrained_weights)

    # Optional Prior Network
    if args.use_prior:
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
        
        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )
    else:
        diffusion_prior = None
    

    # model = torch.compile(model)
    print("model parameters:")
    Model_param = utils.count_params(model)
    print("model.brain_nat")
    Brain_nat_param = utils.count_params(model.brain_nat)
    print("model.backbone")
    Backbone_param = utils.count_params(model.backbone)
    param_count_dict = {"Model_param": Model_param, "Brain_nat_param": Brain_nat_param, "Backbone_param": Backbone_param}
    return (
        clip_img_embedder,
        model,
        diffusion_prior,
        autoenc,
        cnx,
        mean,
        std,
        blur_augs,
        param_count_dict
    )

def setup_optimizer(args, model, diffusion_prior, num_iterations_per_epoch):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    max_lr = args.max_lr

    # Group parameters for NAT backbone
    opt_grouped_parameters = [
        {'params': [p for n, p in model.brain_nat.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2},
        {'params': [p for n, p in model.brain_nat.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    if hasattr(model, 'feature_mapper'):
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.feature_mapper.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.feature_mapper.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ])

    # Add voxel adaptor and embed linear parameters if they exist
    if hasattr(model, 'voxel_adaptor'):
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.voxel_adaptor.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.voxel_adaptor.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ])
    
    if hasattr(model, 'embed_linear'):
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.embed_linear.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2}, 
            {'params': [p for n, p in model.embed_linear.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ])

    # Add backbone parameters
    opt_grouped_parameters.extend([
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ])

    # Add blurry recon parameters if enabled
    if args.blurry_recon:
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.backbone.blin1.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.backbone.blin1.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.backbone.b_maps_projector.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.backbone.b_maps_projector.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.backbone.bupsampler.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.backbone.bupsampler.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ])

    # Add prior network parameters if enabled
    if args.use_prior:
        opt_grouped_parameters.extend([
            {'params': [p for n, p in diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)],
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


def setup_wandb(args,train_url="", test_url=""):
    local_rank = os.getenv('RANK', 0)
    wandb_log = args.wandb_log

    if int(local_rank) == 0 and wandb_log:
        import wandb
        print(f"wandb {args.wandb_project} run {args.model_name}")
        wandb.init(
            entity='nyu_brain_decoding',
            id=args.model_name,
            project=args.wandb_project,
            name=args.model_name,
            config=vars(args),
            resume="auto",
        )
    else:
        wandb_log = False
    return wandb_log


def train(args, model, diffusion_prior, train_dl, test_dl, accelerator, data_type, num_iterations_per_epoch,
          num_test, subj_list, clip_img_embedder, optimizer, lr_scheduler, wandb_log, autoenc, cnx, mean, std,
          blur_augs, epoch_start=0, losses=None, test_losses=None, lrs=None):
    
    device = accelerator.device
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

    accelerator.print(f"{model_name} starting with epoch {epoch_start} / {num_epochs}")
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    # Initialize tracking lists if not provided
    losses = losses if losses is not None else []
    test_losses = test_losses if test_losses is not None else []
    lrs = lrs if lrs is not None else []

    # Training loop
    epoch_progress = tqdm(
        range(epoch_start, num_epochs), 
        disable=not accelerator.is_local_main_process
    )
    
    for epoch in epoch_progress:
        model.train()
        iteration = 0
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
        
        iter_progress = tqdm(train_dl, desc=f'Epoch {epoch}', leave=False, disable=not accelerator.is_local_main_process)
        for train_i, (images, voxels, subj_idx, coords, image_idx) in enumerate(iter_progress):
            
            with torch.amp.autocast('cuda'):
                batch_size = voxels.shape[0]
                if batch_size != args.batch_size:
                    print(f"Warning: Batch size mismatch. Expected {args.batch_size}, got {batch_size}")
                    continue
                optimizer.zero_grad()
                loss=0.

                lens = torch.ones(voxels.shape[0], dtype=torch.long)*voxels.shape[-1]

                image_idx = image_idx.cpu().long().numpy()
                _, img_sorted_idx = np.unique(image_idx, return_index=True)
                voxel0 = voxels[img_sorted_idx]
                image = images[img_sorted_idx]
                coords = coords[img_sorted_idx]

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)

                if use_image_aug: 
                    image = img_augment(image)

                clip_target = clip_img_embedder(image)
                assert not torch.any(torch.isnan(clip_target))
                backbone, clip_voxels, blurry_image_enc_ = model(voxel0, coords)
                    
                if clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if use_prior:
                    loss_prior, prior_out = diffusion_prior(text_embed=backbone, image_embed=clip_target)

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
                            accelerator,
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

                    image_norm = ((image - mean)/std)
                    image_aug = (blur_augs(image - mean))/std
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
                        samp_size = len(image)//5 if len(image)>5 else len(image)
                        random_samps = np.random.choice(np.arange(len(image)), size=samp_size, replace=False)
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

                if accelerator.is_main_process and wandb_log:
                    wandb.log({
                        "train/loss_per_iter": loss.item(),
                        "train/blurry_pixcorr_per_iter": blurry_pixcorr / args.batch_size,
                        "train/recon_cossim_per_iter": recon_cossim / args.batch_size,
                        "train/recon_mse_per_iter": recon_mse / args.batch_size,
                        "train/loss_prior_per_iter": loss_prior_total / args.batch_size,
                        "train/loss_clip_per_iter": loss_clip_total / args.batch_size,
                        "train/loss_blurry_per_iter": loss_blurry_total / args.batch_size,
                        "train/loss_blurry_cont_per_iter": loss_blurry_cont_total / args.batch_size,
                    })

                iteration += 1
                if accelerator.is_main_process:
                    if iteration % args.ckpt_iter == 0 and ckpt_saving:
                        save_ckpt(f'iter_{iteration}', args, accelerator.unwrap_model(model), optimizer, lr_scheduler, epoch, losses, test_losses, lrs, accelerator, ckpt_saving=True)

        model.eval()
        test_image, test_voxel, test_coords, test_lens = None, None, None, None
        
        # if accelerator.is_main_process:
        with torch.no_grad(), torch.amp.autocast('cuda'): 
            # Add progress bar for test dataloader
            test_progress = tqdm(test_dl, desc=f'Testing epoch {epoch}', leave=False, 
                                disable=not accelerator.is_local_main_process)
            for test_i, (images, voxels, subj_idx, coords, image_idx) in enumerate(test_progress):
                images = images.to(device)
                voxels = voxels.to(device)
                coords = coords.to(device)
                image_idx = image_idx.to(device)
                # all test samples should be loaded per batch such that test_i should never exceed 0
                if len(images) != args.batch_size:
                    print(f"Warning: Batch size mismatch. Expected {args.batch_size}, got {len(images)}")
                    continue

                # Update progress bar description with current metrics
                if test_i > 0:  # Only update if we have accumulated some metrics
                    test_progress.set_postfix({
                        'loss': f"{np.mean(test_losses[-(test_i+1):]):.4f}",
                        'fwd_acc': f"{test_fwd_percent_correct/(test_i+1):.4f}",
                        'bwd_acc': f"{test_bwd_percent_correct/(test_i+1):.4f}"
                    })

                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels
                    image = image_idx
                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    for im in unique_image:
                        locs = torch.where(im == image_idx)[0]
                        if len(locs)==1:
                            locs = locs.repeat(3)
                        elif len(locs)==2:
                            locs = locs.repeat(2)[:3]
                        assert len(locs)==3
                        if test_image is None:
                            test_image = torch.Tensor(images[locs,0][None])
                            test_voxel = voxels[locs][None]
                            test_coords = coords[locs][None]
                        else:
                            test_image = torch.vstack((test_image, torch.Tensor(images[locs,0][None])))
                            test_voxel = torch.vstack((test_voxel, voxels[locs][None]))
                            test_coords = torch.vstack((test_coords, coords[locs][None]))
                loss=0.
                            
                test_indices = torch.arange(len(test_voxel))
                voxel = test_voxel[test_indices]
                coords = test_coords[test_indices]
                image = test_image[test_indices]

                clip_target = clip_img_embedder(image)
                for rep in range(3):
                    backbone0, clip_voxels0, blurry_image_enc_ = model(voxel[:,rep], coords[:,rep])
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
                    diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior
                    # TODO: this two line was not tested
                    recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                    recon_mse += mse(prior_out, clip_target).item()
                
                if clip_scale>0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        accelerator,
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
                if wandb_log:
                    wandb.log({
                        "test/loss_per_iter": loss.item(),
                        "test/blurry_pixcorr_per_iter": pixcorr.item() if blurry_recon else 0.0,
                        "test/fwd_acc_per_iter": utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item() if clip_scale > 0 else 0.0,
                        "test/bwd_acc_per_iter": utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item() if clip_scale > 0 else 0.0,
                        "test/loss_clip_per_iter": loss_clip.item() if clip_scale > 0 else 0.0,
                        "test/loss_prior_per_iter": loss_prior.item() if use_prior else 0.0,
                        "global_step": iteration
                    })

            # assert (test_i+1) == 1
            logs = {
                "epoch": epoch,
                "train/loss": np.mean(losses[-num_iterations_per_epoch:]),  # Only average losses from current epoch
                "test/loss": np.mean(test_losses[-len(test_dl):]),  # Only average losses from current test run
                "train/lr": lrs[-1],
                "train/fwd_acc": fwd_percent_correct / (train_i + 1),
                "train/bwd_acc": bwd_percent_correct / (train_i + 1),
                "test/fwd_acc": test_fwd_percent_correct / (test_i + 1),
                "test/bwd_acc": test_bwd_percent_correct / (test_i + 1),
            }

            if clip_scale > 0:
                logs.update({
                    "train/loss_clip": loss_clip_total / (train_i + 1),
                    "test/loss_clip": test_loss_clip_total / (test_i + 1),
                })

            if blurry_recon:
                logs.update({
                    "train/loss_blurry": loss_blurry_total / (train_i + 1),
                    "train/loss_blurry_cont": loss_blurry_cont_total / (train_i + 1),
                    "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                    "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                })

            if use_prior:
                logs.update({
                    "train/loss_prior": loss_prior_total / (train_i + 1),
                    "test/loss_prior": test_loss_prior_total / (test_i + 1),
                    "train/recon_cossim": recon_cossim / (train_i + 1),
                    "test/recon_cossim": test_recon_cossim / (test_i + 1),
                    "train/recon_mse": recon_mse / (train_i + 1),
                    "test/recon_mse": test_recon_mse / (test_i + 1),
                })

            # if finished training or checkpoint interval, save blurry reconstructions
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
                        logs["test/blur_recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()

            if wandb_log:
                wandb.log(logs)
                
        # Save model checkpoint and reconstruct
        if (ckpt_saving) and (epoch % ckpt_interval == 0):
            save_ckpt(f'last',args,accelerator.unwrap_model(model),optimizer,lr_scheduler,epoch, losses, test_losses, lrs, accelerator, ckpt_saving=True)

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()

    accelerator.print("\n===Finished!===\n")
    if ckpt_saving:
        save_ckpt(f'last',args,accelerator.unwrap_model(model),optimizer,lr_scheduler,epoch, losses, test_losses, lrs, accelerator, ckpt_saving=True)

def main():
    torch._dynamo.config.optimize_ddp=False
    args = parse_arguments()
    
    # Initialize accelerator first
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        device_placement=True, 
        split_batches=False, 
        mixed_precision="fp16",
        dynamo_backend="no",
        kwargs_handlers=[kwargs]
    )
    
    # Set up wandb only on main process
    if accelerator.is_main_process:
        if args.wandb_log:
            import wandb
            # Try to resume wandb run if it exists
            try:
                wandb.init(
                    entity='nyu_brain_decoding',
                    project=args.wandb_project,
                    name=args.model_name,
                    id=args.model_name,
                    resume="allow",
                    config=vars(args)
                )
                print(f"Resumed wandb run: {wandb.run.path}")
            except wandb.errors.UsageError:
                # If run doesn't exist, start new one
                wandb.init(
                    entity='nyu_brain_decoding',
                    project=args.wandb_project,
                    name=args.model_name,
                    config=vars(args)
                )
                print(f"Started new wandb run: {wandb.run.path}")
    
    utils.seed_everything(args.seed)
    data_type = torch.float16  # Change depending on your mixed_precision

    # Setup multi-GPU training
    local_rank = int(os.getenv('RANK', 0))
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        num_devices = 1
    args.num_devices = num_devices
    device = accelerator.device
    batch_size = args.batch_size

    # Data preparation
    train_dl, test_dl, num_test, num_iterations_per_epoch = prepare_data(args, data_type)

    # Model initialization
    clip_img_embedder, model, diffusion_prior, autoenc, cnx, mean, std, blur_augs, param_count_dict = build_model(args, device, data_type)
    if args.wandb_log and accelerator.is_main_process:
        wandb.log(param_count_dict)
    optimizer, lr_scheduler = setup_optimizer(args, model, diffusion_prior, num_iterations_per_epoch)

    # Load checkpoint if exists
    epoch_start, losses, test_losses, lrs, resumed = utils.load_ckpt(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        tag='last'
    )

    # Prepare for distributed training
    model, diffusion_prior, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        model, diffusion_prior, optimizer, train_dl, lr_scheduler
    )

    # Print training status
    if resumed:
        accelerator.print(f"Resuming training from epoch {epoch_start}")
    else:
        accelerator.print("Starting new training run")
        epoch_start = 0

    # Training loop
    train(
        args=args,
        model=model,
        diffusion_prior=diffusion_prior,
        train_dl=train_dl,
        test_dl=test_dl,
        accelerator=accelerator,
        data_type=data_type,
        num_iterations_per_epoch=num_iterations_per_epoch,
        num_test=num_test,
        subj_list=[args.subj],
        clip_img_embedder=clip_img_embedder,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wandb_log=args.wandb_log and accelerator.is_main_process,
        autoenc=autoenc,
        cnx=cnx,
        mean=mean,
        std=std,
        blur_augs=blur_augs,
        epoch_start=epoch_start,
        losses=losses,
        test_losses=test_losses,
        lrs=lrs
    )


if __name__ == "__main__":
    main()