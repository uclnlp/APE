#!/bin/bash

HOME="$(pwd)"

NQ_DATA="${HOME}/data/preprocessed_data/nq/nq_dpr_test.json"
DATA="nq"

MODEL=$1
SIZE=$2
N_CTX=$3
BSZ=4
#BUDGET=$5
#TOPK=$6

for k in 5 10 20 30 40; do
  TOPK=$k
  if [ "$SIZE" = "base" ]; then
    BUDGET=$(($TOPK * 12))
  else
    BUDGET=$(($TOPK * 24))
  fi
  echo "budget $BUDGET topk $TOPK"

  ./scripts/eval_scheduler_nq.sh $N_CTX $BSZ $SIZE $MODEL $BUDGET $TOPK
  cd "$HOME"
done
