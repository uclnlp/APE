#!/bin/bash

conda activate FiD
cd FiD

NQ_DATA="../data/preprocessed_data/trivia/trivia_dpr_test.json"
DATA="trivia"
SIZE="base"
MODEL="../pretrained_models/triviaqa_base_dpr/"

BSZ=10
N_CTX=5
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
N_CTX=10
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
N_CTX=20
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
BSZ=5
N_CTX=40
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
N_CTX=50
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
BSZ=2
N_CTX=80
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
N_CTX=100
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/

SIZE="large"
MODEL="../pretrained_models/triviaqa_large_dpr/"

BSZ=8
N_CTX=5
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
N_CTX=10
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
BSZ=4
N_CTX=20
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
BSZ=2
N_CTX=40
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
N_CTX=50
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
BSZ=1
N_CTX=80
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
N_CTX=100
python test.py --model_path $MODEL --test_data_path $NQ_DATA --model_size $SIZE --per_gpu_batch_size $BSZ  --n_context $N_CTX --name "${DATA}_${SIZE}_nctx=${N_CTX}" --checkpoint_dir ../checkpoint/
