import torch
import torch.nn as nn
import numpy as np
from layers import *
from train import *

def get_models_dense(name):
    if name=="DenseNet1":
        return DenseNet1
    elif name=="DenseNet2":
        return DenseNet2
    elif name=="DenseNet3":
        return DenseNet3
    elif name=="DenseNet4":
        return DenseNet4
    elif name=="DenseNet5":
        return DenseNet5
    elif name=="DenseNet6":
        return DenseNet6
    elif name=="DenseNet7":
        return DenseNet7
    elif name=="DenseNet8":
        return DenseNet8
    else:
        raise NameError(name+" not found!")

if __name__=="__main__":
    models=["DenseNet"+str(i) for i in range(1,9)]
    for model in models:
        train(model,'Disney',logging=False,epochs=1)

class DenseNet1(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(DenseNet1, self).__init__()
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

class DenseNet2(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(DenseNet2, self).__init__()
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

class DenseNet3(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(DenseNet3, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.dense3=nn.Linear(nhid2,nhid2)
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
        H = self.dense3(H)
        H = torch.relu(H)
        


        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class DenseNet4(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(DenseNet4, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.dense3=nn.Linear(nhid2,nhid2)
        self.dense4=nn.Linear(nhid2,nhid2)
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
        H = self.dense3(H)
        H = torch.relu(H)
        H = self.dense4(H)
        H = torch.relu(H)
        


        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class DenseNet5(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(DenseNet5, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.dense3=nn.Linear(nhid2,nhid2)
        self.dense4=nn.Linear(nhid2,nhid2)
        self.dense5=nn.Linear(nhid2,nhid2)
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
        H = self.dense3(H)
        H = torch.relu(H)
        H = self.dense4(H)
        H = torch.relu(H)
        H = self.dense5(H)
        H = torch.relu(H)
        


        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class DenseNet6(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(DenseNet6, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.dense3=nn.Linear(nhid2,nhid2)
        self.dense4=nn.Linear(nhid2,nhid2)
        self.dense5=nn.Linear(nhid2,nhid2)
        self.dense6=nn.Linear(nhid2,nhid2)
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
        H = self.dense3(H)
        H = torch.relu(H)
        H = self.dense4(H)
        H = torch.relu(H)
        H = self.dense5(H)
        H = torch.relu(H)
        H = self.dense6(H)
        H = torch.relu(H)
        


        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class DenseNet7(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(DenseNet7, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.dense3=nn.Linear(nhid2,nhid2)
        self.dense4=nn.Linear(nhid2,nhid2)
        self.dense5=nn.Linear(nhid2,nhid2)
        self.dense6=nn.Linear(nhid2,nhid2)
        self.dense7=nn.Linear(nhid2,nhid2)
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
        H = self.dense3(H)
        H = torch.relu(H)
        H = self.dense4(H)
        H = torch.relu(H)
        H = self.dense5(H)
        H = torch.relu(H)
        H = self.dense6(H)
        H = torch.relu(H)
        H = self.dense7(H)
        H = torch.relu(H)
        


        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A

class DenseNet8(torch.nn.Module):
    def __init__(self,A, nfeat, nhid1,nhid2,nhid3, nout):
        super(DenseNet8, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid1)
        self.conv2 = GCNConv(A,nhid1, nhid2)
        self.nlgcn=NLGCN(nhid2)
        self.dense1=nn.Linear(nhid2,nhid2)
        self.dense2=nn.Linear(nhid2,nhid2)
        self.dense3=nn.Linear(nhid2,nhid2)
        self.dense4=nn.Linear(nhid2,nhid2)
        self.dense5=nn.Linear(nhid2,nhid2)
        self.dense6=nn.Linear(nhid2,nhid2)
        self.dense7=nn.Linear(nhid2,nhid2)
        self.dense8=nn.Linear(nhid2,nhid2)
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
        H = self.dense3(H)
        H = torch.relu(H)
        H = self.dense4(H)
        H = torch.relu(H)
        H = self.dense5(H)
        H = torch.relu(H)
        H = self.dense6(H)
        H = torch.relu(H)
        H = self.dense7(H)
        H = torch.relu(H)
        H = self.dense8(H)
        H = torch.relu(H)
        


        Att  = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att,1)
        return Att,A
