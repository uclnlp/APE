#!/bin/bash

HOME=$(pwd)

NQ_DATA="${HOME}/data/preprocessed_data/nq"

# Copy the entire data folder to avoid corrupting the original data files (an issue in Colab)
TMP_DATA=$(mktemp -d -t data-XXXXXXXXXX)
cp -r $NQ_DATA/* $TMP_DATA
echo "Finished copying data from ${NQ_DATA} to ${TMP_DATA}"
NQ_DATA=$TMP_DATA
md5sum $NQ_DATA/*

DATA="nq"
SIZE="base"
NOW=$(date '+%Y%m%d-%H-%M-%S')
NAME="fid-${SIZE}-${DATA}_${NOW}"
CKPT="${HOME}/checkpoints/${NAME}"

python FiD/train.py \
  --train_data_path $NQ_DATA/nq_dpr_train.json \
  --dev_data_path $NQ_DATA/nq_dpr_dev.json \
  --model_size $SIZE \
  --per_gpu_batch_size 2 \
  --n_context 10 \
  --name "${NAME}" \
  --checkpoint_dir $CKPT \
  --lr 1e-4 \
  --eval_freq 1000 \
  --eval_print_freq 1000 \
  --save_freq 1000 \
  --total_step 50000 \
  --is_master
# --checkpointing_encoder

rm -fr $NQ_DATA
