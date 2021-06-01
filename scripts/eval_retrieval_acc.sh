#!/bin/bash

HOME=$(pwd)

DATA=$1
DATA_PATH="${HOME}/data/preprocessed_data/${DATA}/${DATA}_dpr_test.json"

N_CTX=$2
BSZ=$3
SIZE=$4
MODEL=$5
BUDGET=$6
TOPK=$7

NAME="${DATA}_${SIZE}_nctx=${N_CTX}_budget=${BUDGET}_topk=${TOPK}"

python FiD/test_retrieval_acc.py \
  --model_path $MODEL \
  --checkpoint_dir "$MODEL/eval" \
  --name $NAME \
  --test_data_path $DATA_PATH \
  --model_size $SIZE \
  --per_gpu_batch_size $BSZ \
  --n_context $N_CTX \
  --budget $BUDGET \
  --num_passages_retained $TOPK \
  --write_results \
  --is_master

#  --checkpoint_dir $MODEL \
