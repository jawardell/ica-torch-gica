import torch
import numpy as np
import torch.nn as nn
from torch.linalg import norm


a = 0.8
b = 1 - a
EGv = 0.3745672075
ErChuPai = 2 / 3.141592653589793  



def nege(x):
    y = torch.log(torch.cosh(x))
    E1 = y.mean()
    E2 = 0.3745672075
    return (E1 - E2)**2

def joint_loss(sources, mag_norm, reference, m, a, b):
    #sources = sources from X = As
    #mag_norm = maginitude normalization as defined in Du., et al
    #reference = reference spatial maps
    #m = number of time steps
    #a = alpha for independence term
    #b = beta for spatial similarity term
    
    loss = -(a * ErChuPai * torch.arctan(mag_norm * nege(sources)) + b * (1 / m) * sources.t() @ reference)
    return loss


torch.set_default_tensor_type(torch.DoubleTensor)


class GIGICA(torch.nn.Module):
    #init_sources = y1, init_weights = 2c, mag_norm = c, m = m
    def __init__(self, init_weights,  mag_norm, m):
        super().__init__()
        self.W = nn.Linear(init_weights.shape[0], 1, bias=False)
        self.W.weight = nn.Parameter(init_weights.reshape([1,-1]))
        self.mag_norm = mag_norm 
        self.m = m

    def forward(self, X):
        sources = self.W(X)/ norm(self.W.weight)
        return sources