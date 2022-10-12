##vit-vqvae test_vit_patch4_01
第一次尝试使用vit结构，在sketch数据集上，有像素块问题
##第一次debug test_vit_debug_01
使用cifar10数据集，发现在普通的image上未出现像素块问题
##第二次debug test_vit_debug_02
使用sketch数据集，将reconstruction loss * 10，保证三个loss间数量级一致，结果仍存在之前的像素块问题，但相比于初次这一问题有所缓解