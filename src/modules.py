import torch
from torch import einsum, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import math
from scipy.interpolate import RegularGridInterpolator
import scipy
import torch.nn.functional as F
import numpy as np

PRIMES = [1,265443567,805459861]


# def normalized_grid(width, height, feat_dim, device='cuda'):
#     """Returns grid[x,y] -> coordinates for a normalized window.
    
#     Args:
#         width, height (int): grid resolution
#     """
#     features = nn.Parameter(torch.randn(width, height, feat_dim))
#     return features

def get_features(pts, grid, FLAGS, only_grad=False):
    _,feat_dim =grid.codebook[0].shape
    # batch,_,_  = pts.shape
    feats = []
    #Iterate in every level of detail resolution
    for i, res in enumerate(grid.LODS):
        features = bilinear_interpolation(res, grid.codebook[i], pts, FLAGS.grid_type, only_grad=only_grad)
        if FLAGS.multiscale_type == "sum":
            feats.append((torch.unsqueeze(features, dim =-1)))
        else:
            feats.append(features)
    ficiur = torch.cat(feats, -1)
    if FLAGS.multiscale_type == "cat":
        return ficiur
    else:
        return ficiur.sum(-1)

def bilinear_interpolation(res, grid, points, grid_type, visualize_collision=False,only_grad=False):
    """
    Performs bilinear interpolation of points with respect to a grid.

    Parameters:
        grid (numpy.ndarray): A 2D numpy array representing the grid.
        points (numpy.ndarray): A 2D numpy array of shape (n, 2) representing
            the points to interpolate.

    Returns:
        numpy.ndarray: A 1D numpy array of shape (n,) representing the interpolated
            values at the given points.
    """
    if only_grad:
        grid = grid.grad
    # Get the dimensions of the grid
    grid_size, feat = grid.shape
    delta = 1e-6
    # _,N, _ = points.shape
    # res = math.sqrt(grid_size) 
    # Get the x and y coordinates of the four nearest points for each input point
    x1 = torch.floor((points[:,:, 0]-delta)*(res-1)).int()
    y1 = torch.floor((points[:,:, 1]-delta)*(res-1)).int()
    x2 = x1 + 1
    y2 = y1 + 1

    # Compute the weights for each of the four points
    w1 = (x2 - points[:,:, 0]*(res-1)) * (y2 - points[:,:, 1]*(res-1))
    w2 = (points[:,:, 0]*(res-1) - x1) * (y2 - points[:,:, 1]*(res-1))
    w3 = (x2 - points[:,:, 0]*(res-1)) * (points[:,:, 1]*(res-1) - y1)
    w4 = (points[:,:, 0]*(res-1) - x1) * (points[:,:, 1]*(res-1) - y1)


    if grid_type=='NGLOD':
        # Interpolate the values for each point
        return torch.einsum('ab,abc->abc', w1 , grid[(x1+y1*res).long()]) + torch.einsum('ab,abc->abc', w2 , grid[(y1*res+x2).long()]) \
                + torch.einsum('ab,abc->abc', w3 , grid[(y2*res+ x1).long()]) + torch.einsum('ab,abc->abc', w4 , grid[(y2*res+ x2).long()])
    elif grid_type=='HASH':
        npts = res**2
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]).int() ^ (y1 * PRIMES[1]).int()) % (grid_size)
            id2 = ((x2 * PRIMES[0]).int() ^ (y1 * PRIMES[1]).int()) % (grid_size)
            id3 = ((x1 * PRIMES[0]).int() ^ (y2 * PRIMES[1]).int()) % (grid_size)
            id4 = ((x2 * PRIMES[0]).int() ^ (y2 * PRIMES[1]).int()) % (grid_size)
            if visualize_collision:
                #merge ids
                ids = torch.cat((id1,id2,id3,id4), dim=-1)
                for i in ids[-1]:
                    grid[i.long()] += 1
                return
            else:
                return torch.einsum('ab,abc->abc', w1 , grid[(id1).long()]) + torch.einsum('ab,abc->abc', w2 , grid[(id2).long()]) \
                    + torch.einsum('ab,abc->abc', w3 , grid[(id3).long()]) + torch.einsum('ab,abc->abc', w4 , grid[(id4).long()])       
        else:
            if visualize_collision:
                return
            return torch.einsum('ab,abc->abc', w1 , grid[(x1+y1*res).long()]) + torch.einsum('ab,abc->abc', w2 , grid[(y1*res+x2).long()]) \
                + torch.einsum('ab,abc->abc', w3 , grid[(y2*res+ x1).long()]) + torch.einsum('ab,abc->abc', w4 , grid[(y2*res+ x2).long()])
    else:
        print("NOT IMPLEMENTED")
        return

def update_collisions(pts, grid, FLAGS):
    feats = []
    #Iterate in every level of detail resolution
    for i, res in enumerate(grid.LODS):
        bilinear_interpolation(res, grid.collision_list[i], pts, FLAGS.grid_type, visualize_collision=True)
    return
    