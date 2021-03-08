import torch
import torch.nn as nn
import numpy as np
from layers import *


class Net(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.conv3 = GCNConv(A,nhid2, nhid3)
        self.conv4 = GCNConv(A,nhid3, nout)
        # self.dec2 = GCNConv(A,nhid2,nhid3)
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
        H = self.dense2(H)
        H = torch.relu(H)
        


        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class Net1(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(Net1, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.sd_conv1=GCNConv(A,nhid2,nhid3)
        self.sd_conv2=GCNConv(A,nhid2,nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.conv3 = GCNConv(A,nhid2, nhid3)
        self.conv4 = GCNConv(A,nhid3, nout)
        # self.dec2 = GCNConv(A,nhid2,nhid3)
        self.dec2 = nn.Linear(nhid2,nhid3)
        
    def forward(self,X):
        H  = self.conv1(X)
        H = torch.relu(H)
        H  = self.conv2(H)
        H = torch.relu(H)
        H=self.nlgcn(H)


        A=self.sd_conv1(H)
        A=torch.relu(A)
        # A=self.sd_conv2(A)
        # A=torch.relu(A)
        #A=self.dec2(A)
        A=torch.matmul(A,A.T)
        A=torch.sigmoid(A)
        
        # H = self.dense1(H)
        # H = torch.relu(H)
        # H = self.dense2(H)
        # H = torch.relu(H)
        


        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

def get_model(model_name,dataset,Adj_norm):
    if model_name=='N1':
        if dataset=="Enron":
            return Net(Adj_norm,18,15,12,15,18)
        if dataset=="Amazon":
            return Net(Adj_norm,21,12,10,12,21)
        if dataset=="facebook":
            return Net(Adj_norm,10,7,5,7,10)
        if dataset=="Disney":
            return Net(Adj_norm,28,15,10,15,28)
    if model_name=='N2':
        if dataset=="Enron":
            return Net1(Adj_norm,18,15,12,15,18)
        if dataset=="Amazon":
            return Net1(Adj_norm,21,12,10,12,21)
        if dataset=="facebook":
            return Net1(Adj_norm,10,7,5,7,10)
        if dataset=="Disney":
            return Net1(Adj_norm,28,15,10,15,28)
        
        