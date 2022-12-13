
import argparse
import os
import numpy as np
import datetime
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from synthesis_network import *
from datasets import *
from id_loss import *
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='WiderFace70K', help="target dataset")
parser.add_argument("--source_img_path", type=str, default='', help="source face image path")
parser.add_argument("--source_list", type=str, default='data_list/source_list.txt', help="source face image list")
parser.add_argument("--target_img_path", type=str, default='', help="target face image path")
parser.add_argument("--target_list", type=str, default='data_list/target_list_wf_70k.txt', help="target face image list")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--ua", type=float, default=6, help="ua")
parser.add_argument("--la", type=float, default=0, help="la")
parser.add_argument("--um", type=float, default=0.65, help="um")
parser.add_argument("--lm", type=float, default=0.05, help="lm")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--lambda_identity", type=float, default=8, help="identity loss weight")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--backbone_dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=10, help="dimensionality of style coefficients")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of style codes")
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Directory name to save the checkpoints [checkpoint]')
parser.add_argument('--logs_dir', type=str, default='./logs', help='Root directory of samples')
parser.add_argument('--signature', default=str(datetime.datetime.now()))
parser.add_argument('--cuda', type=bool, default=True, help='True for GPU, False for CPU [False]')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

# #
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
print(opt)

# #
def checkpoint(checkpoint_dir, model_name, epoch):
    model_path = checkpoint_dir + '/' + model_name + '/synthsis-' + str(epoch) + '.pth'
    os.makedirs(os.path.join(checkpoint_dir, model_name), exist_ok=True)
    torch.save({
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizerG_state_dict': optimizer_G.state_dict(),
                'optimizerD_state_dict': optimizer_D.state_dict()
        }, model_path)

# #
sig = opt.signature
writer = SummaryWriter('%s/logs/%s/%s' % (opt.logs_dir, opt.model_name, sig))

## Initialize generator, encoder and discriminators
G = Generator(dim=opt.backbone_dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim, latent_dim=opt.latent_dim)
D = MultiDiscriminator()

if opt.cuda:
    G = G.cuda()
    D = D.cuda()

if opt.epoch != 0:
    # Load pretrained models
    all_model = torch.load(opt.checkpoint_dir + '/' + opt.model_name + '/synthsis-' + str(opt.epoch) + '.pth')
    G.load_state_dict(all_model['G_state_dict'])
    D.load_state_dict(all_model['D_state_dict'])
else:
    # Initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

# # Optimizers 
optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# # Loss
identity_criterion = IDLoss(ckpt_dict='id_weights/model_ir_se50.pth')
if opt.cuda:
    identity_criterion = identity_criterion.cuda()

# #  dataset
dataloader = DataLoader(
    SourceTargetFace(source_img_path=opt.source_img_path, 
                     source_list=opt.source_list, 
                     target_img_path=opt.target_img_path,
                     target_list=opt.target_list),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ------------------------------------------------------------------# 
# ----------
#  Training
# ----------
valid = 1
fake = 0
num_iter = 0
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        source_img = Variable(batch["source_img"]).cuda()
        target_img = Variable(batch["target_img"]).cuda()

        # -------------------------------
        #  Train Generator 
        # -------------------------------
        optimizer_G.zero_grad()

        # Produce output using sampled z
        if ((epoch+i)%2) == 0:
            sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (source_img.size(0), opt.style_dim)))).cuda()
            z_norm = torch.norm(sampled_z, dim=1, keepdim=True)
            target_cosine_dis = (z_norm-opt.la)*(opt.um-opt.lm)/(opt.ua-opt.la) + opt.lm
            w_latent, syn_img, loss_regu = G(source_img, sampled_z)
            loss_GAN = D.compute_loss(syn_img, valid)
            # loss_identity
            esti_cosine_dis = identity_criterion.forward(source_img, syn_img)
            loss_identity = torch.mean((esti_cosine_dis - target_cosine_dis)**2)

            # #
            loss_G = loss_GAN  + opt.lambda_identity*loss_identity + loss_regu   
        else:
            sampled_z = Variable(torch.zeros(source_img.size(0), opt.style_dim)).cuda()
            w_latent, syn_img, loss_regu = G(source_img, sampled_z)
            loss_pixel = torch.mean(torch.abs(source_img - syn_img))

            # #
            loss_G = loss_pixel + loss_regu

        loss_G.backward()
        optimizer_G.step()

        # ----------------------------------
        #  Train Discriminator 
        # ----------------------------------
        optimizer_D.zero_grad()
        loss_D = D.compute_loss(target_img, valid) + D.compute_loss(syn_img.detach(), fake)
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, D loss: %f, Identity loss: %f, Regu: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_GAN.item(),
                loss_D.item(),
                loss_identity.item(),
                loss_regu.item(),
                time_left,
            )
        )
        writer.add_scalar('loss/loss_G', loss_GAN, num_iter)
        writer.add_scalar('loss/loss_id', loss_identity, num_iter)
        writer.add_scalar('loss/loss_D', loss_D, num_iter) 
        writer.add_scalar('loss/loss_regu', loss_regu, num_iter)

        if batches_done % opt.sample_interval == 0:
            utils.sample_images(opt.logs_dir, opt.model_name, source_img, target_img, syn_img, epoch, i)

        num_iter = num_iter + 1

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        checkpoint(opt.checkpoint_dir, opt.model_name, epoch)
