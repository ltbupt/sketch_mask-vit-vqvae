import argparse
import os
import multiprocessing as mp
def parse_opts():
    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--train_data_folder', type=str,
                        default="/root/dataset/cifar10",
                        help='name of the data folder')
    parser.add_argument('--test_data_folder', type=str,
                        default="/root/dataset/cifar10",
                        help='name of the data folder')

    # Latent space
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors (default: 512)')
    parser.add_argument('--num_channels', type=int, default=3,
                        help=' ')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--log_folder', type=str, default='models/log',
                        help=' ')
    parser.add_argument('--output_folder', type=str, default='models/vqvae',
                        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='set the device (cpu or cuda, default: cpu)')

    # test
    parser.add_argument('--test_model_path', type=str, default='',
                        help='')
    parser.add_argument('--test_res_path', type=str, default='',
                        help='')
    

    args = parser.parse_args()

    args.steps = 0
    return args