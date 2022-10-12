import os
import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from model import VectorQuantizedVAE
from tensorboardX import SummaryWriter

from dataset import get_dataloader
from train import train_one_epoch, val_one_epoch
from opts import parse_opts
from test import test


def main(args):
    # Create logs and models folder if they don't exist
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if args.test_res_path!=''and not os.path.exists(args.test_res_path):
        os.makedirs(args.test_res_path)
        
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(args.log_folder)
    save_filename = args.output_folder
    
    print('')
    print('step 1 : Define the data loaders...')
    train_loader, valid_loader, test_loader = get_dataloader(args)

    print('')
    print('step 2 : Define the model...')
    model = VectorQuantizedVAE(args.num_channels, 
                               args.hidden_size, 
                               args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.test_model_path != '':
        print('')
        print('step 3 : test...')
        model.load_state_dict(torch.load(args.test_model_path, map_location=args.device))
        print('load weight at {}'.format(args.test_model_path))
        test(test_loader, model, args, writer)

    else:
        print('')
        print('step 3 : train...')
        best_loss = -1.
        for epoch in range(args.num_epochs):
            print('epoch : {}'.format(epoch))
            torch.cuda.empty_cache()

            train_one_epoch(train_loader, model, optimizer, args, writer)

            loss, _ = val_one_epoch(valid_loader, model, args, writer)

            if (epoch == 0) or (loss < best_loss):
                best_loss = loss
                with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                    torch.save(model.state_dict(), f)
            with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
                torch.save(model.state_dict(), f)

    

if __name__ == '__main__':
    args = parse_opts()
    torch.cuda.empty_cache()
    main(args)
