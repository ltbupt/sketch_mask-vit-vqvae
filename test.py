#encoding=utf8
import pdb

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToPILImage
import cv2

def generate_samples(images, model, args):# images.shape([16, 3, 32, 32])
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)# x_tilde.shape([16, 3, 32, 32]

    return x_tilde# x_tilde.shape([16, 3, 32, 32])


def test(test_loader, model, args, writer):
    fixed_images, _ = next(iter(test_loader))# fixed_images.Size([16, 3, 32, 32])
    save_image(fixed_images, args.test_res_path+'/original.jpg', nrow=8, normalize=True)
    print('save at {}'.format(args.test_res_path+'/original.jpg'))
    # fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    # writer.add_image('original', fixed_grid, 0)
    
    # Generate the samples first once           # fixed_images.Size([16, 3, 32, 32])
    reconstruction = generate_samples(fixed_images, model, args)# reconstruction.shape([16, 3, 32, 32])

    save_image(reconstruction.cpu(), args.test_res_path+'/reconstruction.jpg', nrow=8, normalize=True)
    print('save at {}'.format(args.test_res_path+'/reconstruction.jpg'))
    # grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)# grid.shape([3, 70, 274])
    # writer.add_image('reconstruction', grid, 0)