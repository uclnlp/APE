#!/bin/bash

HOME="$(pwd)"

DATA=$1
MODEL=$2
SIZE=$3
N_CTX=$4

BSZ=4
#BUDGET=$5
#TOPK=$6

for k in 5 10 20; do
  TOPK=$k
  if [ "$SIZE" = "base" ]; then
    BUDGET=$(($TOPK * 12))
  else
    BUDGET=$(($TOPK * 24))
  fi
  echo "budget $BUDGET topk $TOPK"

  ./scripts/eval_retrieval_acc.sh $DATA $N_CTX $BSZ $SIZE $MODEL $BUDGET $TOPK
  cd "$HOME"
done
