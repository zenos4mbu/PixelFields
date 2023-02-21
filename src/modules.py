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

def get_features(pts, grid, resolutions, grid_type):
    _,feat_dim =grid[0].shape
    batch,_,_  = pts.shape
    feats = []

    #Iterate in every level of detail resolution
    for i, res in enumerate(resolutions):
        features = bilinear_interpolation(grid[i],pts, grid_type)
        feats.append((torch.unsqueeze(features, dim =-1)))
    ficiur = torch.cat(feats, -1)
    return ficiur.sum(-1)



def bilinear_interpolation(grid, points, grid_type):
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
    # Get the dimensions of the grid
    grid_size, feat = grid.shape
    dimension = math.sqrt(grid_size) 
    _,N, _ = points.shape

    # Get the x and y coordinates of the four nearest points for each input point
    x1 = torch.floor(points[:,:, 0]*dimension).int()
    y1 = torch.floor(points[:,:, 1]*dimension).int()
    x2 = (x1 + 1)
    y2 = (y1 + 1)

    # Ensure that the indices are within the grid
    x1 = torch.clip(x1, 0, dimension - 1)
    y1 = torch.clip(y1, 0, dimension - 1)
    x2 = torch.clip(x2, 0, dimension - 1)
    y2 = torch.clip(y2, 0, dimension - 1)

    # Compute the weights for each of the four points
    w1 = (x2 - points[:,:, 0]) * (y2 - points[:,:, 1])
    w2 = (points[:,:, 0] - x1) * (y2 - points[:,:, 1])
    w3 = (x2 - points[:,:, 0]) * (points[:,:, 1] - y1)
    w4 = (points[:,:, 0] - x1) * (points[:,:, 1] - y1)


    if grid_type=='NGLOD':
        # Interpolate the values for each point
        values = torch.einsum('ab,abc->abc', w1 , grid[(x1+y1*dimension).long()]) + torch.einsum('ab,abc->abc', w2 , grid[(y1*dimension+x2).long()]) \
                + torch.einsum('ab,abc->abc', w3 , grid[(y2*dimension+ x1).long()]) + torch.einsum('ab,abc->abc', w4 , grid[(y2*dimension+ x2).long()])
    elif grid_type=='HASH':
        npts = dimension**2
        if npts > grid_size:
            id1 = (x1 * PRIMES[0]).int() ^ (y1 * PRIMES[1]).int() % grid_size
            id2 = (x2 * PRIMES[0]).int() ^ (y1 * PRIMES[1]).int() % grid_size
            id3 = (x1 * PRIMES[0]).int() ^ (y2 * PRIMES[1]).int() % grid_size
            id4 = (x2 * PRIMES[0]).int() ^ (y2 * PRIMES[1]).int() % grid_size
            values = torch.einsum('ab,abc->abc', w1 , grid[(id1).long()]) + torch.einsum('ab,abc->abc', w2 , grid[(id2).long()]) \
                    + torch.einsum('ab,abc->abc', w3 , grid[(id3).long()]) + torch.einsum('ab,abc->abc', w4 , grid[(id4).long()])
        else:
            values = torch.einsum('ab,abc->abc', w1 , grid[(x1+y1*dimension).long()]) + torch.einsum('ab,abc->abc', w2 , grid[(y1*dimension+x2).long()]) \
                + torch.einsum('ab,abc->abc', w3 , grid[(y2*dimension+ x1).long()]) + torch.einsum('ab,abc->abc', w4 , grid[(y2*dimension+ x2).long()])
    else:
        print("NOT IMPLEMENTED")
    return values