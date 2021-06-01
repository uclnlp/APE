#!/bin/bash

HOME=$(pwd)

TRIVIA_DATA="${HOME}/data/preprocessed_data/trivia"

## Copy the entire data folder to avoid corrupting the original data files (an issue in Colab)
#TMP_DATA=$(mktemp -d -t data-XXXXXXXXXX)
#cp -r $TRIVIA_DATA/* $TMP_DATA
#echo "Finished copying data from ${TRIVIA_DATA} to ${TMP_DATA}"
#TRIVIA_DATA=$TMP_DATA
##md5sum $TRIVIA_DATA/*

DATA="trivia"
N_CTX=$1
POOL=$2
SIZE=$3
BSZ=$4
ACC=$5

NOW=$(date '+%Y%m%d-%H-%M-%S')
CKPT="${HOME}/checkpoints"
MODEL="${HOME}/pretrained_models/triviaqa_${SIZE}_dpr/"

NAME="acfid-${SIZE}-${DATA}_${POOL}_${NOW}"

python FiD/train_ac.py \
  --model_path $MODEL \
  --train_data_path $TRIVIA_DATA/trivia_dpr_train.json \
  --dev_data_path $TRIVIA_DATA/trivia_dpr_dev.json \
  --dev_data_size 2000 \
  --model_size $SIZE \
  --per_gpu_batch_size $BSZ \
  --gradient_accumulation_steps $ACC \
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

#rm -fr $TRIVIA_DATA
