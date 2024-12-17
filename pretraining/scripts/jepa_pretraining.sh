#!/bin/bash
#SBATCH --job-name=JEPA
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00 
#SBATCH --constraint="a100|h100"
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --array=0-3

NUM_MASKS_ARRAY=(1 2 3 4)

NUM_MASKS=${NUM_MASKS_ARRAY[$SLURM_ARRAY_TASK_ID]}

MODEL_NAME="num_masks_${NUM_MASKS}_mask_ratio_0.8"

singularity exec --nv \
    --overlay /scratch/cc6946/envs/neural-decoding/fMRI.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c " \
    source /ext3/env.sh; \
    cd /scratch/cc6946/projects/Brain_Decoding/pretraining; \
    python jepa.py --wandb_log --wandb_project=JEPA --model_name=${MODEL_NAME} \
    --num_masks=${NUM_MASKS} --mask_ratio=0.8 --num_epochs=20 --ckpt_interval=5 \
    --batch_size=24"