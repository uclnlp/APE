#!/bin/bash

HOME=$(pwd)

NQ_DATA="${HOME}/data/preprocessed_data/nq/nq_dpr_test.json"
DATA="nq"

N_CTX=$1
BSZ=$2
SIZE=$3
#MODEL="${HOME}/pretrained_models/nq_${SIZE}_dpr/"
MODEL=$4

NAME="${DATA}_${SIZE}_nctx=${N_CTX}"
python FiD/test.py \
  --model_path $MODEL \
  --test_data_path $NQ_DATA \
  --model_size $SIZE \
  --per_gpu_batch_size $BSZ \
  --n_context $N_CTX \
  --name $NAME \
  --checkpoint_dir "$MODEL/eval" \
  --write_results \
  --is_master
