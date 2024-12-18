#!/bin/bash
#SBATCH --job-name=JEPA
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00 
#SBATCH --constraint="a100|h100"
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --array=0-2

NUM_MASKS_ARRAY=(4 6 8)

NUM_MASKS=${NUM_MASKS_ARRAY[$SLURM_ARRAY_TASK_ID]}

MODEL_NAME="num_masks_${NUM_MASKS}_mask_ratio_0.95"

singularity exec --nv \
    --overlay /scratch/cc6946/envs/neural-decoding/fMRI.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c " \
    source /ext3/env.sh; \
    cd /scratch/cc6946/projects/Brain_Decoding/pretraining; \
    python jepa.py --wandb_log --wandb_project=fMRI_JEPA --model_name=${MODEL_NAME} \
    --num_masks=${NUM_MASKS} --mask_ratio=0.95 --num_epochs=20 --ckpt_interval=1 \
    --batch_size=24 --hidden_dim=256"