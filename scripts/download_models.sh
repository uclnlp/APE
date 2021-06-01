#!/bin/bash

ROOT_DIR=$PWD

MODEL_DIR=$ROOT_DIR/pretrained_models
mkdir -p "$MODEL_DIR"

for NAME in "nq_base_dpr" "nq_large_dpr" "triviaqa_base_dpr" "triviaqa_large_dpr"; do
  mkdir -p "$MODEL_DIR"/${NAME}
  cd $MODEL_DIR/${NAME}

  if [[ ! -f pytorch_model.bin ]]; then
    wget http://dl.fbaipublicfiles.com/FiD/pretrained_models/${NAME}/pytorch_model.bin
  fi
  if [[ ! -f config.json ]]; then
    wget http://dl.fbaipublicfiles.com/FiD/pretrained_models/${NAME}/config.json
  fi

  cd $ROOT_DIR
done
