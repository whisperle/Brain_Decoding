#!/bin/bash
#SBATCH --job-name=JEPA
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00 
#SBATCH --constraint="a100|h100"
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --array=0-2

export SSL_CERT_FILE=/scratch/cc6946/cacert.pem

NUM_MASKS_ARRAY=(4 6 8)

NUM_MASKS=${NUM_MASKS_ARRAY[$SLURM_ARRAY_TASK_ID]}

MODEL_NAME="num_masks_${NUM_MASKS}_mask_ratio_0.95"
PRETRAINED_CKPT="/scratch/cc6946/projects/Brain_Decoding/pretraining/ckpt/num_masks_${NUM_MASKS}_mask_ratio_0.95/last.pth"

singularity exec --nv \
    --overlay /scratch/cc6946/envs/neural-decoding/fMRI.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c " \
    source /ext3/env.sh; \
    cd /scratch/cc6946/projects/Brain_Decoding/Downstream; \
    python Train.py --wandb_log --ckpt_saving --model_name=${MODEL_NAME} --num_session=40 \
    --batch_size=8 --wandb_project=fMRI_Finetune --multisubject=1 --decoder_hidden_dim=512 \
    --encoder_hidden_dim=256 --num_heads=8 --tome_r=1000 --nat_depth=8 --drop=0.1 \
    --finetuning --pretrained_ckpt=${PRETRAINED_CKPT} --freeze_pretrained_weights \
    --wandb_entity=chuyangchen-new-york-university"
