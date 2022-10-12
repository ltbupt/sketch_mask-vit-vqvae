EXPERIMENT="test_codebook"
MODEL="train_codebook_16_224_8192"
python main.py \
--train_data_folder /root/dataset/segmentation/train \
--test_data_folder /root/dataset/segmentation/val \
--output_folder ./result/${EXPERIMENT}/weight \
--log_folder ./result/${EXPERIMENT}/log \
--test_model_path ./result/${MODEL}/weight/best.pt \
--test_res_path ./result/${MODEL}/test_res