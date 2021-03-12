import torch
import torch.nn as nn
import numpy as np
from layers import *


class AttConvNet1(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(AttConvNet1, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.conv3 = GCNConv(A,nhid2, nout)
        self.dec2 = nn.Linear(nhid2,nhid3)
        
    def forward(self,X):
        H  = self.conv1(X)
        H = torch.relu(H)
        H  = self.conv2(H)
        H = torch.relu(H)
        H=self.nlgcn(H)

        A=self.dec2(H)
        A=torch.matmul(H,H.T)
        A=torch.sigmoid(A)
        
        H = self.dense1(H)
        H = torch.relu(H)
        
        Att  = self.conv3(H)
        Att = torch.softmax(Att,1)
        return Att,A

class AttConvNet2(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(AttConvNet2, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.conv3 = GCNConv(A,nhid2, nhid3)
        self.conv4 = GCNConv(A,nhid3, nout)
        self.dec2 = nn.Linear(nhid2,nhid3)
        
    def forward(self,X):
        H  = self.conv1(X)
        H = torch.relu(H)
        H  = self.conv2(H)
        H = torch.relu(H)
        H=self.nlgcn(H)

        A=self.dec2(H)
        A=torch.matmul(H,H.T)
        A=torch.sigmoid(A)
        
        H = self.dense1(H)
        H = torch.relu(H)
        
        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class AttConvNet3(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(AttConvNet3, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.conv3 = GCNConv(A,nhid2, nhid3)
        self.conv3_1 = GCNConv(A,nhid3, nhid3)
        self.conv4 = GCNConv(A,nhid3, nout)
        self.dec2 = nn.Linear(nhid2,nhid3)
        
    def forward(self,X):
        H  = self.conv1(X)
        H = torch.relu(H)
        H  = self.conv2(H)
        H = torch.relu(H)
        H=self.nlgcn(H)

        A=self.dec2(H)
        A=torch.matmul(H,H.T)
        A=torch.sigmoid(A)
        
        H = self.dense1(H)
        H = torch.relu(H)
        
        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att  = self.conv3_1(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class AttConvNet4(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(AttConvNet4, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.conv3 = GCNConv(A,nhid2, nhid3)
        self.conv3_1 = GCNConv(A,nhid3, nhid3)
        self.conv3_2 = GCNConv(A,nhid3, nhid3)
        self.conv4 = GCNConv(A,nhid3, nout)
        self.dec2 = nn.Linear(nhid2,nhid3)
        
    def forward(self,X):
        H  = self.conv1(X)
        H = torch.relu(H)
        H  = self.conv2(H)
        H = torch.relu(H)
        H=self.nlgcn(H)

        A=self.dec2(H)
        A=torch.matmul(H,H.T)
        A=torch.sigmoid(A)
        
        H = self.dense1(H)
        H = torch.relu(H)
        
        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att  = self.conv3_1(H)
        Att = torch.relu(Att)
        Att  = self.conv3_2(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A