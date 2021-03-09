import torch
import torch.nn as nn
import numpy as np


class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.A_hat = A
        self.W     = nn.Parameter(torch.rand(in_channels,out_channels))
    
    def forward(self, X):
        out = torch.mm(torch.mm(self.A_hat, X), self.W)
        return out

class NLGCN(nn.Module):
    def __init__(self, channels):
        super(NLGCN, self).__init__()
        self.channels=channels
        self.c=nn.Parameter(torch.rand(channels,1))
        self.conv1D=nn.Conv1d(1,1,9,padding=4)
        
    def forward(self, X):
        self.c=nn.Parameter(torch.rand(self.channels,1))
        Av=X.matmul(self.c)
        indices_sorted=Av.argsort()
        indices_reorder=torch.sort(indices_sorted).values
        X=X[indices_sorted]
        shape=np.shape(X)
        X=torch.reshape(X, (shape[0],shape[2]))
        X=torch.transpose(X,0,1)
        X=torch.reshape(X,(shape[2],shape[1],shape[0]))
        X=self.conv1D(X)
        X=torch.reshape(X,(shape[2],shape[0]))
        X=torch.transpose(X,1,0)
        X=X[indices_reorder]
        X=torch.reshape(X,(shape[0],shape[2]))
        return X