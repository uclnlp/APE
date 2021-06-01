#!/bin/bash

ROOT_DIR=$(pwd)
echo $ROOT_DIR

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install transformers
#pip install transformers==3.0.2
#pip install transformers==3.1.0

cd $ROOT_DIR/etc/transformers-v3.0.2
#cd $ROOT_DIR/etc/transformers-v3.1.0
pip install -e .
cd $ROOT_DIR

## Install Apex
#cd $ROOT_DIR/etc/apex
#pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install wandb
pip install wandb

cd $ROOT_DIR
pip install -r requirements.txt
