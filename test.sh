EXPERIMENT="test_vit_debug_02"

python main.py \
--train_data_folder /root/dataset/segmentation/train \
--test_data_folder /root/dataset/segmentation/val \
--output_folder ./result/${EXPERIMENT}/weight \
--log_folder ./result/${EXPERIMENT}/log \
--test_model_path ./result/train_vit_debug_02/weight/best.pt \
--test_res_path ./result/train_vit_debug_02/test_res