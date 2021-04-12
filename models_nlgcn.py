import torch
import torch.nn as nn
import numpy as np
from layers import *
from train import *

def get_models_nlgcn(name):
    if name=="NLNet1":
        return NLNet1
    elif name=="NLNet2":
        return NLNet2
    elif name=="NLNet3":
        return NLNet3
    else:
        raise NameError(name+" not found!")

if __name__=="__main__":
    models=["NLNet"+str(i) for i in range(1,4)]
    for model in models:
        train(model,'Disney',logging=False,epochs=1)

class NLNet1(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(NLNet1, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
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
        H = self.dense2(H)
        H = torch.relu(H)
        
        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class NLNet2(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(NLNet2, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.nlgcn1=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.conv3 = GCNConv(A,nhid2, nhid3)
        self.conv4 = GCNConv(A,nhid3, nout)
        self.dec2 = nn.Linear(nhid2,nhid3)
        
    def forward(self,X):
        H  = self.conv1(X)
        H = torch.relu(H)

        H  = self.conv2(H)
        H = torch.relu(H)

        H=self.nlgcn(H)
        H=self.nlgcn1(H)

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

class NLNet3(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(NLNet3, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.nlgcn1=NLGCN(nhid2)
        self.nlgcn2=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.conv3 = GCNConv(A,nhid2, nhid3)
        self.conv4 = GCNConv(A,nhid3, nout)
        self.dec2 = nn.Linear(nhid2,nhid3)
        
    def forward(self,X):
        H  = self.conv1(X)
        H = torch.relu(H)

        H  = self.conv2(H)
        H = torch.relu(H)

        H=self.nlgcn(H)
        H=torch.relu(H)
        H=self.nlgcn(H)
        H=torch.relu(H)
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