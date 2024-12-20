#!/bin/bash
#SBATCH --account=topfmri
#SBATCH --job-name=brainnat
#SBATCH --nodes=1           
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint="a100|h100"
#SBATCH --account=pr_60_tandon_advanced

overlay_ext3=/scratch/cl6707/dl-env/fmri.ext3
singularity exec --nv \
    --overlay fMRI.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
cd /scratch/cl6707/Projects/fmri/Brain_Decoding/Downstream

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=2 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

# multisubject pretraining
model_name="multisubject_excludingsubj01_40sess"
echo model_name=${model_name}
accelerate launch --num_processes=${NUM_GPUS} --mixed_precision=fp16 Train.py \
    --data_path=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    --cache_dir=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    --model_name=${model_name} \
    --no-multi_subject \
    --subj=1 \
    --batch_size=${BATCH_SIZE} \
    --max_lr=3e-4 \
    --mixup_pct=0.33 \
    --num_epochs=150 \
    --use_prior \
    --prior_scale=30 \
    --clip_scale=1 \
    --no-blurry_recon \
    --blur_scale=0.5 \
    --no-use_image_aug \
    --n_blocks=4 \
    --hidden_dim=512 \
    --num_sessions=40 \
    --ckpt_interval=999 \
    --ckpt_saving \
    --wandb_log \
    --num_heads=4 \
    --tome_r=2000 \
    --last_n_features=16 \
    --nat_depth=2

# singlesubject finetuning
#model_name="finetuned_subj01_40sess"
#echo model_name=${model_name}
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset --cache_dir=/weka/proj-fmri/shared/cache --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/multisubject_excludingsubj01_40sess
