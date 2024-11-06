#!/bin/bash
# # Make sure you activate your fmri environment created from src/setup.sh
cd /scratch/cl6707/Projects/fmri/Brain_Decoding/Downstream
source /ext3/env.sh

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=20 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
# export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
# export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
# echo MASTER_ADDR=${MASTER_ADDR}
# echo MASTER_PORT=${MASTER_PORT}
# echo WORLD_SIZE=${COUNT_NODE}

# multisubject pretraining
model_name="multisubject_excludingsubj01_40sess"
echo model_name=${model_name}
accelerate launch --num_processes=${NUM_GPUS} --mixed_precision=fp16 Train.py \
    --data_path=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    --cache_dir=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    --model_name=${model_name} \
    --multi_subject=1,2,5,7 \
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
    --num_sessions=1 \
    --ckpt_interval=999 \
    --ckpt_saving \
    --no-wandb_log \
    --full_attention \
    --num_heads=8 \
    --tome_r=2000 \
    --last_n_features=16 \
    --nat_depth=2 \
    --nat_num_neighbors=8

# singlesubject finetuning
#model_name="finetuned_subj01_40sess"
#echo model_name=${model_name}
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset --cache_dir=/weka/proj-fmri/shared/cache --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/multisubject_excludingsubj01_40sess
