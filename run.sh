#!/usr/bin/env bash

#python main.py PATH_OF_THE_CONFIG_FILE

# Experiments of FCN8s
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=1
#python3 main.py configs/newfcn8s_exp_0.json
python main.py configs/erfnet_exp_0.json
