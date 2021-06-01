#!/bin/bash

HOME=$(pwd)

NQ_DATA="${HOME}/data/preprocessed_data/nq/nq_dpr_test.json"
DATA="nq"

N_CTX=$1
BSZ=$2
SIZE=$3
MODEL=$4
BUDGET=$5
TOPK=$6

NAME="${DATA}_${SIZE}_nctx=${N_CTX}_budget=${BUDGET}_topk=${TOPK}"

python FiD/test_ac.py \
  --model_path $MODEL \
  --test_data_path $NQ_DATA \
  --model_size $SIZE \
  --per_gpu_batch_size $BSZ \
  --n_context $N_CTX \
  --name $NAME \
  --checkpoint_dir "/tmp/${NAME}" \
  --budget $BUDGET \
  --num_passages_retained $TOPK
