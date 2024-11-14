#!/bin/bash
#SBATCH --job-name=JEPA
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00 
#SBATCH --constraint="a100|h100"
#SBATCH --account=pr_60_tandon_advanced

singularity exec --nv \
    --overlay /scratch/cc6946/envs/neural-decoding/fMRI.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c " \
    source /ext3/env.sh; \
    cd /scratch/cl6707/Projects/fmri/Brain_Decoding/pretraining; \
    python jepa.py --wandb_log --model_name=v1 --num_masks=150 --mask_size=50 "