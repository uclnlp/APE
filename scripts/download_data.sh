#!/bin/bash

ROOT_DIR=$(pwd)

DATA_DIR="$ROOT_DIR/data"
mkdir -p "$DATA_DIR"
PROC_DATA="$DATA_DIR"/preprocessed_data
mkdir -p "$PROC_DATA"

# Download retrieved passages for NaturalQuestions
cd $ROOT_DIR
mkdir -p "$PROC_DATA"/nq
cd $PROC_DATA/nq

if [[ ! -f nq_dpr_train.json ]]; then
  wget http://dl.fbaipublicfiles.com/FiD/preprocessed_data/nq/nq_dpr_train.json.xz
  xz --decompress nq_dpr_train.json.xz
fi
#6e86173809c6b2f8390f9dd20631c001  nq/nq_dpr_train.json

if [[ ! -f nq_dpr_dev.json ]]; then
  wget http://dl.fbaipublicfiles.com/FiD/preprocessed_data/nq/nq_dpr_dev.json.xz
  xz --decompress nq_dpr_dev.json.xz
fi
#9e09fba3a450bebf86c7706b181a7491  nq/nq_dpr_dev.json

if [[ ! -f nq_dpr_test.json ]]; then
  wget http://dl.fbaipublicfiles.com/FiD/preprocessed_data/nq/nq_dpr_test.json.xz
  xz --decompress nq_dpr_test.json.xz
fi
#06850dd776b473818129665344470664  nq/nq_dpr_test.json

# Download retrieved passages for TriviaQA
mkdir -p "$PROC_DATA"/trivia
cd "$PROC_DATA"/trivia

if [[ ! -f trivia_dpr_train.json ]]; then
  wget http://dl.fbaipublicfiles.com/FiD/preprocessed_data/trivia/trivia_dpr_train.json.xz
  xz --decompress trivia_dpr_train.json.xz
fi
#dd12dddd006ec9c35894e0a2d188f9d6  trivia/trivia_dpr_train.json

if [[ ! -f trivia_dpr_dev.json ]]; then
  wget http://dl.fbaipublicfiles.com/FiD/preprocessed_data/trivia/trivia_dpr_dev.json.xz
  xz --decompress trivia_dpr_dev.json.xz
fi
#2128d1f3aafb35c61c62855a37d63d0f  trivia/trivia_dpr_dev.json

if [[ ! -f trivia_dpr_test.json ]]; then
  wget http://dl.fbaipublicfiles.com/FiD/preprocessed_data/trivia/trivia_dpr_test.json.xz
  xz --decompress trivia_dpr_test.json.xz
fi
#22c74e02c429b1fb777c4f62967e6756  trivia/trivia_dpr_test.json

cd $ROOT_DIR