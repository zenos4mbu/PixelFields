from tabnanny import verbose
from tokenize import group
from turtle import pos
from matplotlib.style import context
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import math
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat, reduce

from modules import get_features

import cv2

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
        print(f"({self.linear.weight.shape} , {torch.mean(self.linear.weight)})")

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


class SimpleModel(pl.LightningModule):
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS=FLAGS        
        self.criterion = nn.MSELoss()
        # self.B_gauss = torch.randn((self.FLAGS.mapping_size, 2)).cuda() * self.FLAGS.mapping_multiplier
        self.activations = {"RELU": nn.ReLU(), "SIN": Sine()}
        self.simplemlp = nn.Sequential(
            nn.Linear(FLAGS.feat_dim, FLAGS.hidden_dim),
            self.activations[self.FLAGS.activation],
            # nn.Linear(256, 256),
            # self.activations[self.FLAGS.activation],
            nn.Linear(FLAGS.hidden_dim, FLAGS.hidden_dim),
            self.activations[self.FLAGS.activation],
            nn.Linear(FLAGS.hidden_dim, FLAGS.n_channels),
        )
        # self.grid=[]
        if self.FLAGS.use_grid:
            if self.FLAGS.grid_type=='NGLOD':
                self.LODS = [4 * 2**L for L in range(FLAGS.base_lod,FLAGS.base_lod + FLAGS.num_LOD)]
                self.init_feature_structure()
            elif self.FLAGS.grid_type=='HASH':
                b = np.exp((np.log(self.FLAGS.max_grid_res) - np.log(self.FLAGS.min_grid_res)) / (self.FLAGS.num_LOD-1))
                self.LODS = [int(1 + np.floor(self.FLAGS.min_grid_res*(b**l))) for l in range(self.FLAGS.num_LOD)]
                self.init_hash_structure()
            else:
                print("NOT SUPPORTED")
        # for LOD in self.LODS:
        #     grid = normalized_grid(LOD, LOD, FLAGS.feat_dim).to(torch.device('cuda'))
        #     self.grid.append(grid)
        
    def forward(self, x):
        x = self.simplemlp(x)
        return x
    
    def init_feature_structure(self):
        self.codebook = nn.ParameterList([])
        for LOD in self.LODS:
            fts = torch.zeros(LOD**2, self.FLAGS.feat_dim) #+ self.feature_bias
            fts += torch.randn_like(fts) * self.FLAGS.feature_std
            self.codebook.append(nn.Parameter(fts))

    def init_hash_structure(self):
        self.codebook_size=2 ** self.FLAGS.band_width
        self.codebook = nn.ParameterList([])
        for LOD in self.LODS:
            num_pts=LOD**2
            fts = torch.zeros(min(self.codebook_size, num_pts), self.FLAGS.feat_dim) #+ self.feature_bias
            fts += torch.randn_like(fts) * self.FLAGS.feature_std
            self.codebook.append(nn.Parameter(fts))

    def input_mapping(self, x, B):
        if B is None:
            return x
        else:
            x_proj = (2. * np.pi * x) @ B.t()
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def get_psnr(self, pred, target):
        return 10 * torch.log10((1 ** 2) / torch.mean((pred - target) ** 2))

    def training_step(self, train_batch, batch_idx):
        pos = train_batch[0]
        image = train_batch[1]
        b,h,w,c = image.shape
        # pos = self.input_mapping(pos, self.B_gauss)

        pos = rearrange(pos, 'a h w c -> a (h w) c')

        if self.FLAGS.use_grid:
            features = get_features(pos, self.codebook, self.LODS, self.FLAGS.grid_type)
            x = self.forward(features)
        else:
            x = self.forward(pos)


        x = rearrange(x, 'a (h w) c -> a h w c', h = h, w = w)

        if(self.FLAGS.display):
            pred_true = np.hstack((x[0].detach().cpu().numpy(), image[0].detach().cpu().numpy()))
            pred_true = cv2.cvtColor(pred_true, cv2.COLOR_RGB2BGR)
            cv2.imshow("Pred_true", pred_true)
            cv2.waitKey(1)

        # image = TF.rotate(image.permute(0,3,1,2), 90).permute(0,2,3,1)
        

        loss = self.criterion(x, image)
        self.psnr = self.get_psnr(x, image).item()
        # self.ssim = ssim(x.view(1, self.FLAGS.n_channels, self.FLAGS.image_size, self.FLAGS.image_size), image.half().view(1, self.FLAGS.n_channels, self.FLAGS.image_size, self.FLAGS.image_size)).item()

        # self.log('Training/loss', loss)
        # self.log('Training/LR', self.optimizer.param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        # self.log('Training/PSNR', self.psnr, prog_bar=True)
        # self.log('Training/SSIM', self.ssim, prog_bar=True)
        return loss

    # def validation_step(self, val_batch, batch_idx):
    #     pos = val_batch[0]
    #     image = val_batch[1]
        
    #     # pos = self.input_mapping(pos, self.B_gauss)

    #     x = self.forward(pos)

    #     loss = self.criterion(x, image)

    #     self.psnr = self.get_psnr(x, image).item()
    #     # self.ssim = ssim(x.view(1, self.FLAGS.n_channels, self.FLAGS.image_size, self.FLAGS.image_size), image.view(1, self.FLAGS.n_channels, self.FLAGS.image_size, self.FLAGS.image_size)).item()

    #     self.log('Validation/loss', loss)
    #     self.log('Validation/PSNR', self.psnr, prog_bar=True)
    #     self.log('Validation/SSIM', self.ssim, prog_bar=True)
    #     return loss

    # def test_step(self, test_batch, batch_idx):
    #     pos = test_batch[0]
    #     image = test_batch[1]
        
    #     pos = self.input_mapping(pos, self.B_gauss)

    #     x = self.forward(pos)

    #     loss = self.criterion(x, image)

    #     self.psnr = self.get_psnr(x, image).item()
    #     # self.ssim = ssim(x.view(1, self.FLAGS.n_channels, self.FLAGS.image_size, self.FLAGS.image_size), image.view(1, self.FLAGS.n_channels, self.FLAGS.image_size, self.FLAGS.image_size)).item()

    #     self.log('Test/loss', loss)
    #     self.log('Test/PSNR', self.psnr, prog_bar=True)
    #     self.log('Test/SSIM', self.ssim, prog_bar=True)
    #     return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            # self.hnet.internal_params,
            self.parameters(),
            lr=self.FLAGS.learning_rate#,
            # weight_decay=self.FLAGS.weight_decay,
        )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=10, eta_min=self.FLAGS.learning_rate / 50
        # )

        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 
        #     mode = "min",
        #     factor=0.1,
        #     patience=self.FLAGS.patience
        # )

        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer, 
        #     base_lr=1e-5, 
        #     max_lr=self.FLAGS.learning_rate, 
        #     cycle_momentum=False,
        #     step_size_up=20)

        return {'optimizer': self.optimizer}#, 'lr_scheduler': lr_scheduler, 'monitor': "Training/loss"}