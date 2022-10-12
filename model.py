#encoding=utf8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from torch.distributions.normal import Normal

from torchvision import datasets, transforms
from torchvision.utils import save_image
from functions import vq, vq_st


from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
import pdb


class AutoencoderViT(nn.Module):


    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)  # 224, 16, 3, 768
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # embed_dim=768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding #num_patches=196 embed_dim=768

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)  # embed_dim=768 decoder_embed_dim=512


        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))  # self.pos_embed.shape=(197, 768)

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))  # self.decoder_pos_embed.shape=(197, 512)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data  # w.shape([768, 3, 16, 16])
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs



    def forward_encoder(self, x):  # x.shape = [B, C, H, W]
        # embed patches

        x = self.patch_embed(x)  # x.shape = [B, num_patch, encoder_embed] [B, 64, 768]num_patch根据img和patch_size 自动算出来

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # x.shape [B, 64, 768]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # cls_token.shape=torch.Size([1, 1, batch_dim])
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # x.shape=[B, num_patch+1, batch_dim]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


    def forward_decoder(self, x):
        # x.shape([B, num_patches+1,768]
        # embed tokens
        x = self.decoder_embed(x) # [B,num_patches+1,512]
        #x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = x[:,1:,:]  # no cls token x_.shape[B,num_patches,512]
        #x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token x.shape[B,num_patches+1,512]

        # add pos embed
        x = x + self.decoder_pos_embed  # x.shape[B,num_patches+1,512]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)  # x.shape[B,num_patches+1,512]

        # remove cls token
        x = x[:, 1:, :]  # x.shape[B,num_patches,patch_size ** 2 * in_chans]

        return x

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        return loss

    def forward(self, imgs):  # imgs.shape[B, C, H, W]
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred)
        return pred,loss

def vit_base_patch4_dec512d8b(**kwargs):
    model = AutoencoderViT(
        patch_size=4, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def vit_base_patch16_dec512d8b(**kwargs):
    model = AutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):  # x.Size([16, 256, 8, 8])
        return x + self.block(x)

class VQEmbedding(nn.Module):
    '''
    K : int
        num of code
    D : int
        dim of a code
    '''
    def __init__(self, K, D):  # K = 512 D = 256
        super().__init__()
        self.embedding = nn.Embedding(K, D)# 这就是codebook记录的code 表征
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):# z_e_x.shape([16, 256, 8, 8])
        '''
        将encoder编码后的z_e_x 与当前的codebook最近的表征输出
        '''

        #z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()  # z_q_x_.shape([16, 8, 8, 256])
        latents = vq(z_e_x, self.embedding.weight)#
        return latents

    def straight_through(self, z_e_x):
        '''
        将encoder编码后的z_e_x 得到codebook中最近的 z_q_x 和响应的 idx
        '''

        # step 1 : 得到codebook中最近的 与 z_e_x(被编码的) 最近的 z_q_x(来自于codebook)
        z_e_x_ = z_e_x.contiguous()  # z_e_x.shape([B,num_patches+1,768])
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())# z_q_x_.shape([B,num_patches+1,768]) indices.shape([B*num_patches+1])
        z_q_x = z_q_x_.contiguous()  # z_q_x.shape([B,num_patches+1,768])

        # # step 2 : 得到codebook中最近的 z_q_x 和响应的 idx
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)  # z_q_x_bar_flatten.shape([B*num_patches+1, 256])
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x)  # z_q_x_bar_.shape([B, num_patches+1,768])
        z_q_x_bar = z_q_x_bar_.contiguous()  # z_q_x_bar.shape([B, num_patches+1,768])

        return z_q_x, z_q_x_bar  # z_q_x.shape([B,num_patches+1,768]),z_q_x_bar.shape([B, num_patches+1,768])


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):  # input_dim=3 dim=256
        super().__init__()
        self.vit = vit_base_patch16_dec512d8b()
        self.encoder = self.vit.forward_encoder
        self.decoder = self.vit.forward_decoder

        self.codebook = VQEmbedding(K, dim)# K = 512 dim = 256
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        # 将encoder编码后的z_e_x 与当前的codebook最近的表征输出
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        # 往embedding层输入latent 得到 z_q_x
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):  # x.Size([16, 3, 32, 32])                          train : torch.Size([128, 3, 32, 32])
        '''
        输入 batch

        return
        x_tilde : [B, Channel, H, W]
            重建的图片
        z_e_x :  [B, codebook_dim, H_z, W_z]
            encode之后 但是还没被量化的 中间状态
        z_q_x : [B, codebookdim, H_z, W_z]
            量化后 但是还未经过codebook的 中间状态
        '''

        z_e_x = self.vit.forward_encoder(x) #[B,N**2+1,768]

        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x) # z_q_x_st.shape([B,num_patches+1,768]),z_q_x_bar.shape([B, num_patches+1,768])

        x_tilde = self.vit.forward_decoder(z_q_x_st)  # [B,num_pathes,patch_size**2*in_channel]
        x_recon = self.vit.unpatchify(x_tilde)  # [B,3,32,32]

        return x_recon, z_e_x, z_q_x
        # x_tilde.shape([B, 3, 32, 32]) z_e_x.shape([B,num_patches+1,768]) z_q_x.shape([B,num_patches+1,768])



if __name__ == '__main__':
    v = vit_base_patch16_dec512d8b()
    img = torch.randn(128,3,32,32)
    latent = v.forward_encoder(img)
    print(latent.shape)
    latent = latent[:,1:,:]
    print(latent.shape)
    #要去掉cls_token那一维再unpatchy
    _x = v.unpatchify(latent)
    print(_x.shape)


