#!/bin/bash

EXPERIMENT="train_vit_debug_02"
python main.py \
--train_data_folder /root/dataset/segmentation/train \
--test_data_folder /root/dataset/segmentation/val \
--output_folder ./result/${EXPERIMENT}/weight \
--log_folder ./result/${EXPERIMENT}/log \
--test_res_path ./result/${EXPERIMENT}/test_res