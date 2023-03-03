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
import wandb

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
    
class HashTable(nn.Module):
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS=FLAGS
        if self.FLAGS.visualize_collisions:
            self.collision_list = []
        self.codebook_size=2 ** self.FLAGS.band_width
        self.codebook = nn.ParameterList([])
        b = np.exp((np.log(self.FLAGS.max_grid_res) - np.log(self.FLAGS.min_grid_res)) / (self.FLAGS.num_LOD-1))
        self.LODS = [int(1 + np.floor(self.FLAGS.min_grid_res*(b**l))) for l in range(self.FLAGS.num_LOD)]
        for LOD in self.LODS:
            num_pts=LOD**2
            fts = torch.zeros(min(self.codebook_size, num_pts), self.FLAGS.feat_dim) #+ self.feature_bias
            fts += torch.randn_like(fts) * self.FLAGS.feature_std
            feat = nn.Parameter(fts)
            feat = feat.cuda()
            self.codebook.append(feat)
            if self.FLAGS.visualize_collisions:
                coll = torch.zeros_like(feat)
                self.collision_list.append(coll)
        

class Octree(nn.Module):
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS=FLAGS
        self.codebook = nn.ParameterList([])
        self.LODS = [2**L for L in range(FLAGS.base_lod,FLAGS.base_lod + FLAGS.num_LOD)]
        for LOD in self.LODS:
            fts = torch.zeros(LOD**2, self.FLAGS.feat_dim) #+ self.feature_bias
            fts += torch.randn_like(fts) * self.FLAGS.feature_std
            self.codebook.append(nn.Parameter(fts))

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
            # nn.Linear(FLAGS.hidden_dim, FLAGS.hidden_dim),
            # self.activations[self.FLAGS.activation],
            nn.Linear(FLAGS.hidden_dim, FLAGS.n_channels),
        )
        # self.grid=[]
        if self.FLAGS.use_grid:
            if self.FLAGS.grid_type=='NGLOD':
                self.init_feature_structure()
            elif self.FLAGS.grid_type=='HASH':
                self.init_hash_structure()
            else:
                print("NOT SUPPORTED")
        # for LOD in self.LODS:
        #     grid = normalized_grid(LOD, LOD, FLAGS.feat_dim).to(torch.device('cuda'))
        #     self.grid.append(grid)
        
    def forward(self, x):
        x = self.simplemlp(x)
        return x
    
    def init_hash_structure(self):
        self.grid = HashTable(self.FLAGS)
    
    def init_feature_structure(self):
        self.grid = Octree(self.FLAGS)

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

        masking_points = False
        b,h,w,c = image.shape


        if masking_points:
            mask = (image.sum(-1)/3) > 0.2
        
            mask = mask.detach().cpu().numpy().astype(np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
            mask = cv2.dilate(mask, kernel)

            #use plt to show the mask but first we neeed to convert it to a 3 channel image
            # mask = np.stack([mask*255, mask*255, mask*255], axis=-1)
            # mask = mask.astype(np.uint8)
            # plt.imshow(mask[0])
            # plt.show()

            pos = pos[mask, :]

            mask = torch.from_numpy(mask).unsqueeze(-1).cuda()

            image = image*mask

            pos = pos.unsqueeze(0)

            image_to_show = image
        else:
            image_to_show = image
            pos = rearrange(pos, 'a h w c -> a (h w) c')

        if self.FLAGS.use_grid:
            features = get_features(pos, self.grid, self.FLAGS.grid_type, self.FLAGS.visualize_collisions)
            x = self.forward(features)
        else:
            x = self.forward(pos)

        if masking_points:
            image = rearrange(image, 'a h w c -> a (h w) c')
            background = torch.zeros_like(image)
            mask = rearrange(mask, 'a h w c -> a (h w) c')
            mask = mask.squeeze(-1)
            background[mask, :] = x
            x = rearrange(background, 'a (h w) c -> a h w c', h = h, w = w)
        else:
            x = rearrange(x, 'a (h w) c -> a h w c', h = h, w = w)

        if(self.FLAGS.display):
            pred_true = np.hstack((x[0].detach().cpu().numpy(), image_to_show[0].detach().cpu().numpy()))
            pred_true = cv2.cvtColor(pred_true, cv2.COLOR_RGB2BGR)
            cv2.imshow("Pred_true", pred_true)
            cv2.waitKey(1)
        # if self.current_epoch == 50:
        #     self.log({"image ep{}".format(self.current_epoch) : [wandb.Image(x)]})
        # image = TF.rotate(image.permute(0,3,1,2), 90).permute(0,2,3,1)
        
        # self.grid.codebook[0][0].retain_grad()

        loss = self.criterion(x, image_to_show)
        self.psnr = self.get_psnr(x, image_to_show).item()
        # self.ssim = ssim(x.view(1, self.FLAGS.n_channels, self.FLAGS.image_size, self.FLAGS.image_size), image.half().view(1, self.FLAGS.n_channels, self.FLAGS.image_size, self.FLAGS.image_size)).item()

        self.log('Training/loss', loss)
        self.log('Training/LR', self.optimizer.param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        self.log('Training/PSNR', self.psnr, prog_bar=True)

        # self.log('Training/lerning_rate', self.optimizer.param_groups[1]['lr'], prog_bar=True, sync_dist=True)

        # self.log('Gradients', self.psnr, prog_bar=True)
        # self.log('Training/SSIM', self.ssim, prog_bar=True)
        return loss
    
    def on_after_backward(self):
        # print(self.grid.codebook[0].grad)
        # example to inspect gradient information in tensorboard


        ## PLOT OF CODEBOOK GRADIENTS


        if self.trainer.current_epoch > 0 and self.trainer.global_step % 1==0:  # don't make the tf file huge
            for i,code in enumerate(self.grid.codebook):
                if(self.FLAGS.display):
                    grad_list = self.grid.codebook[i].grad[:,0].tolist()
                    if self.FLAGS.visualize_collisions:
                        # normalize grad list from -1 to +1
                        grad_list = (grad_list - np.min(grad_list)) / (np.max(grad_list) - np.min(grad_list))

                        # nomralize grad list
                        grad_list = np.abs(grad_list)
                        # normalize collision table
                        self.grid.collision_list[i] = self.grid.collision_list[i]/self.grid.collision_list[i].max()
                        collision_table = self.grid.collision_list[i].tolist()

                        #plot table and collision table in a single plot
                        plt.plot(grad_list)
                        plt.plot(collision_table)
                        plt.title( "Codebook {} gradients".format(i))
                        plt.show()
                    else:

                    #absolute value of grad_list
                        grad_list = np.abs(grad_list)
                        plt.plot(grad_list)
                        plt.title( "Codebook {} gradients".format(i))
                        plt.show()
        #         # self.logger.experiment.add_histogram(f'codebook{i}_grad', code.grad, self.trainer.global_step)
                # wandb.log({'codebook{i}': wandb.plot.histogram(table, "gradients")})

        ## PLOT OF GRADIENTS IN IMAGE SPACE

        h = int(self.FLAGS.image_size)
        w = int(self.FLAGS.image_size)
        grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h), torch.linspace(0, 1, steps=w)])
        positions = torch.stack([grid_y, grid_x], dim=-1)
        positions = rearrange(positions, 'h w c -> (h w) c')
        positions = positions.to(torch.device('cuda'))
        positions = torch.unsqueeze(positions, 0)
        if self.FLAGS.use_grid:
            gradients = get_features(positions, self.grid, self.FLAGS.grid_type, only_grad=True)
            gradients = rearrange(gradients, 'a (h w) c -> a h w c', h = h, w = w)
            gradients = gradients.sum(dim=-1)

            #normalize gradients from -1 to 1
            gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())
            gradients = (gradients - 0.5) * 2
            gradients = (gradients + 1) / 2

            #get absolute value of gradients
            gradients = torch.abs(gradients)

            
            gradients = gradients[0].detach().cpu().numpy()
            # self.logger.experiment.add_image('gradients', gradients, self.trainer.global_step)
            # wandb.log({'gradients': wandb.Image(gradients)})
            if(self.FLAGS.display):
                cv2.imshow("gradients", gradients)
                cv2.waitKey(1)      

    def configure_optimizers(self):
        grid_params = []
        other_params = []

        if self.FLAGS.use_grid:
            grid_params = list(self.grid.parameters())
        
        other_params = list(self.simplemlp.parameters())
        # other_params.append(self.criterion.parameters())

        params = []
        params.append({'params': grid_params, 'lr': self.FLAGS.learning_rate*self.FLAGS.grid_lr_factor})
        params.append({'params': other_params, 'lr': self.FLAGS.learning_rate})

        
            
        self.optimizer = torch.optim.Adam(
            params
            # weight_decay=self.FLAGS.weight_decay,
        )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=10, eta_min=self.FLAGS.learning_rate / 50
        # )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode = "min",
            factor=0.1,
            patience=self.FLAGS.patience
        )

        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer, 
        #     base_lr=1e-5, 
        #     max_lr=self.FLAGS.learning_rate, 
        #     cycle_momentum=False,
        #     step_size_up=20)

        return {'optimizer': self.optimizer, 'lr_scheduler': lr_scheduler, 'monitor': "Training/loss"}