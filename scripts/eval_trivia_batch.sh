#!/bin/bash

conda activate FiD
cd FiD

TRIVIA_DATA="../data/preprocessed_data/trivia/trivia_dpr_test.json"
DATA="trivia"
SIZE=$1
MODEL="../pretrained_models/triviaqa_${SIZE}_dpr/"

BSZ=2

for N_CTX in 6 12 22 26 30 35 45; do
  echo "${SIZE}  ${DATA}  ${N_CTX}"
  python test.py --model_path $MODEL --test_data_path $TRIVIA_DATA --model_size $SIZE --per_gpu_batch_size $BSZ --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
done
