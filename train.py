import pdb

from tqdm import tqdm
import torch
import torch.nn.functional as F



def train_one_epoch(data_loader, model, optimizer, args, writer):
    for images, _ in tqdm(data_loader):
        images = images.to(args.device)# images.shape([128, 3, 32, 32])
        optimizer.zero_grad()
        x_recon, z_e_x, z_q_x = model(images)
        # model.py 115行
        # x_recon.shape([B, 3, 32, 32]) z_e_x.shape([B, 256, 8, 8]) z_q_x.shape([B, 256, 8, 8])
        
        # Reconstruction loss
        loss_recons = F.mse_loss(x_recon, images)  # mse_loss均方误差
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())  # mse_loss均方误差
        # Commitment objective
        # https://ml.berkeley.edu/blog/posts/vq-vae/
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())  # mse_loss均方误差

        loss = 10*loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)

        optimizer.step()
        args.steps += 1

def val_one_epoch(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in tqdm(data_loader):
            images = images.to(args.device)
            x_recon, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_recon, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar(args.log_folder, loss_recons.item(), args.steps)
    writer.add_scalar(args.log_folder, loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()