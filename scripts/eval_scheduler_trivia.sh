#!/bin/bash

HOME=$(pwd)

TRIVIA_DATA="${HOME}/data/preprocessed_data/trivia/trivia_dpr_test.json"
DATA="trivia"

N_CTX=$1
BSZ=$2
SIZE=$3
MODEL=$4
BUDGET=$5
TOPK=$6

NAME="${DATA}_${SIZE}_nctx=${N_CTX}_budget=${BUDGET}_topk=${TOPK}"

python FiD/test_ac_scheduler.py \
  --model_path $MODEL \
  --checkpoint_dir "$MODEL/eval" \
  --name $NAME \
  --test_data_path $TRIVIA_DATA \
  --model_size $SIZE \
  --per_gpu_batch_size $BSZ \
  --n_context $N_CTX \
  --budget $BUDGET \
  --num_passages_retained $TOPK \
  --write_results \
  --is_master

#  --checkpoint_dir $MODEL \
