#!/bin/bash

conda activate FiD
cd FiD

NQ_DATA="../data/preprocessed_data/nq/nq_dpr_test.json"
DATA="nq"
SIZE=$1
MODEL="../pretrained_models/nq_${SIZE}_dpr/"

BSZ=2

for N_CTX in 6 12 22 26 30 35 45; do
  python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
done
