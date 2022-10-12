#!/bin/bash

EXPERIMENT="train_codebook_16_224_8192"

python main.py \
--train_data_folder /root/dataset/segmentation/train \
--test_data_folder /root/dataset/segmentation/val \
--output_folder ./result/${EXPERIMENT}/weight \
--log_folder ./result/${EXPERIMENT}/log \
--test_res_path ./result/${EXPERIMENT}/test_res > ${EXPERIMENT}.out 2>&1 &