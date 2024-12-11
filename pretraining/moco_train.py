import random
import math
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import os

import moco.optimizer
import torchvision.models as torchvision_models
import argparse
from dataset import MindEye2Dataset, SubjectBatchSampler, custom_collate_fn
from mask import MaskCollator
import MoCo_BrainNAT
import data_augmentation
import torch.nn.functional as F
import wandb

# using multi-gpus settings
torch.distributed.init_process_group(backend="nccl")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
RANK = dist.get_rank()
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
MASTER_RANK = 0
device = torch.device(f"cuda:{LOCAL_RANK}")
device_id = RANK % torch.cuda.device_count()

"""Parameters for training"""
def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="Name of the model, used for checkpoint saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--data_path", type=str, default='/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2',  # os.getcwd(),
        help="Path to where NSD data is stored or where to download it",
    )
    parser.add_argument(
        "--cache_dir", type=str, default='/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2',  # os.getcwd(),
        help="Path to where miscellaneous files downloaded from huggingface are stored. Defaults to current directory.",
    )
    # TODO: We gonna validate on all the subjects in the ideal case since we are doing multi-subject stuff.
    # held-out subject
    parser.add_argument(
        "--subj", type=int, default=2, choices=[1, 2, 3, 4, 5, 6, 7, 8],
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
        "--batch_size", type=int, default=16,
        help="Batch size can be increased by 10x if only training retrieval submodule and not diffusion prior",
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
        "--num_epochs", type=int, default=15,
        help="Number of epochs of training",
    )
    parser.add_argument(
        "--multi_subject", type=lambda x: [int(i) for i in x.split(',')],
        default="2",  # [1,2,3,4,5,6,7,8],
        help="List of subjects to use for multi-subject training",
    )
    parser.add_argument(
        "--new_test", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to use the new test set",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=1024,  # todo Try 512
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'],
        help="Type of learning rate scheduler",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    # moco specific configs:
    parser.add_argument('--moco-dim', default=256, type=int,
                        help='feature dimension (default: 256)')
    parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                        help='hidden dimension in MLPs (default: 4096)')
    parser.add_argument('--moco-m', default=0.99, type=float,
                        help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco-m-cos', action='store_true',
                        help='gradually increase moco momentum to 1 with a '
                             'half-cycle cosine schedule')
    parser.add_argument('--moco-t', default=1.0, type=float,
                        help='softmax temperature (default: 1.0)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-6)',
                        dest='weight_decay')
    parser.add_argument('--optimizer', default='lars', type=str,
                        choices=['lars', 'adamw'],
                        help='optimizer used (default: lars)')
    args = parser.parse_args()
    return args


def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


def prepare_data(args, data_type):
    # masking
    mask_token = torch.nn.Parameter(torch.randn(1, 192, device=device))
    mask_collator = MaskCollator(0.6, 1)

    train_data = MindEye2Dataset(args, data_type, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dl = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size,
                                           collate_fn=mask_collator,
                                           num_workers=4, pin_memory=True)

    test_data = MindEye2Dataset(args, data_type, 'test')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size,
                                          collate_fn=custom_collate_fn,
                                          num_workers=4, pin_memory=True)

    num_iterations_per_epoch = len(train_data) // args.batch_size
    print(f"Train length: {len(train_data)}")
    print(f"Test length: {len(test_data)}")
    return train_dl, test_dl, len(test_data), num_iterations_per_epoch, mask_token


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


# main train loop
def train(args, train_dl, test_dl, model, optimizer, epoch, data_type, scaler, scheduler, mask_token):
    print(f"Total Training Epoches: {epoch}")
    print(f"Training Dataset Size: {len(train_dl)}")
    # setting current device
    for i in range(epoch):  # train epoches
        if RANK == MASTER_RANK:
            print(f"Epoch {i + 1} / {epoch}:")

        iters_per_epoch = len(train_dl)
        moco_m = args.moco_m

        """Training"""
        loss_total = 0.0
        model.train()
        tot_samples = 0
        for j, (images, voxels, subj_idx, coords, image_idx, mask_indices) in enumerate(train_dl):
            optimizer.zero_grad()
            # put train data on gpu
            voxels = voxels.cuda(device)
            coords = coords.cuda(device)
            images = images.cuda(device)
            mask_indices = mask_indices.cuda(device)

            if voxels.shape[2] != 15724:
                if voxels.shape[2] < 15724:
                    padding_size = 15724 - voxels.shape[2]
                    voxels = F.pad(voxels, (0, padding_size), mode='constant', value=0)
                else:
                    voxels = voxels[:, :, 0:15724]
            if coords.shape[1] != 15724:
                if coords.shape[1] < 15724:
                    padding_size = 15724 - coords.shape[1]
                    coords = F.pad(coords, (0, 0, 0, padding_size), mode='constant', value=0)
                else:
                    coords = coords[:, 0:15724]

            with torch.amp.autocast('cuda'):
                if args.moco_m_cos:
                    moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)
                # print(voxels.shape)  # 4,1,15724
                # print(coords.shape)  # 4,15724,3

                loss = model(voxels, coords, moco_m, True, mask_indices, mask_token)
                loss_total += float(loss) * voxels.shape[0]
                tot_samples += voxels.shape[0]

                # compute gradient and do SGD step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if RANK == MASTER_RANK:
                    wandb.log({"loss": round(float(loss), 4)})
                    print_progress_bar(j + 1, iters_per_epoch, prefix='Progress', suffix='Complete', length=50)

        if RANK == MASTER_RANK:
            print(f"Epoch {i + 1} loss = {loss_total / tot_samples:.5f}")

        """Testing on downstream task"""
        # model.eval()
        # for images, voxels, subj_idx, coords, image_idx in test_dl:
        scheduler.step()
        # save model's encoder parameters
        # torch.save(model.base_encoder.state_dict(), f"/home/yz7212/fMRI/save_models/model1.pth")

    # destroy
    if RANK == MASTER_RANK:
        wandb.finish()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    torch.cuda.set_device(device)
    torch.distributed.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

    args = parse_arguments()
    seed_everything(args.seed)
    data_type = torch.float16  # Change depending on your mixed_precision

    # recording
    if RANK == MASTER_RANK:
        record_name = f"batch{args.batch_size}_lr{args.lr}_sub_{args.multi_subject}_momen{args.momentum}"
        wandb.init(
            project="MOCO-project",
            name="batch32_lr001_subj2_maskr06",
            config={
                "learning_rate": args.lr,
                "architecture": "MOCO_attention",
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
            }
        )

    """Prepare model"""
    model = MoCo_BrainNAT.MoCo_attention()
    model.to(device_id)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    """Prepare data"""
    train_dl, test_dl, num_test, num_iterations_per_epoch, mask_token = prepare_data(args, data_type)

    """Prepare Optimizer"""
    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    else:
        print("Optimizer AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)
    # scaler
    scaler = torch.amp.GradScaler('cuda')
    scheduler = MultiStepLR(optimizer, milestones=[4, 7], gamma=0.1)
    train(args, train_dl, test_dl, model, optimizer, args.num_epochs, data_type, scaler, scheduler, mask_token)

