#!/bin/bash

HOME=$(pwd)

NQ_DATA="${HOME}/data/preprocessed_data/nq"

# Copy the entire data folder to avoid corrupting the original data files (an issue in Colab)
TMP_DATA=$(mktemp -d -t data-XXXXXXXXXX)
cp -r $NQ_DATA/* $TMP_DATA
echo "Finished copying data from ${NQ_DATA} to ${TMP_DATA}"
NQ_DATA=$TMP_DATA
#md5sum $NQ_DATA/*

DATA="nq"
#SIZE="base"
SIZE=$3
NOW=$(date '+%Y%m%d-%H-%M-%S')
CKPT="${HOME}/checkpoints"
MODEL="${HOME}/pretrained_models/nq_${SIZE}_dpr/"

N_CTX=$1
POOL=$2
NAME="acfid-${SIZE}-${DATA}_${POOL}_${NOW}"

python FiD/train_ac.py \
  --model_path $MODEL \
  --train_data_path $NQ_DATA/nq_dpr_train.json \
  --dev_data_path $NQ_DATA/nq_dpr_dev.json \
  --dev_data_size 2000 \
  --model_size $SIZE \
  --per_gpu_batch_size $4 \
  --gradient_accumulation_steps $5 \
  --n_context $N_CTX \
  --name "${NAME}" \
  --checkpoint_dir $CKPT \
  --lr 1e-4 \
  --log_freq 50 \
  --eval_freq 5000 \
  --save_freq 5000 \
  --total_step 20000 \
  --is_master \
  --freeze_fid_params \
  --has_answer_pool_type $POOL
# --fp16
# --checkpointing_encoder

rm -fr $NQ_DATA
