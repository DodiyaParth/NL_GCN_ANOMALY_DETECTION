import torch
import torch.nn as nn
import numpy as np
from layers import *
from models_attconv import *
from models_conv import *
from models_dense import *
from models_nlgcn import *


class Net(torch.nn.Module):
    def __init__(self, A, nfeat, nhid1, nhid2, nhid3, nout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(A, nfeat, nhid1)
        self.conv2 = GCNConv(A, nhid1, nhid2)
        self.nlgcn = NLGCN(nhid2)
        self.dense1 = nn.Linear(nhid2, nhid2)
        self.dense2 = nn.Linear(nhid2, nhid2)
        self.conv3 = GCNConv(A, nhid2, nhid3)
        self.conv4 = GCNConv(A, nhid3, nout)
        self.dec2 = nn.Linear(nhid2, nhid3)

    def forward(self, X):
        H = self.conv1(X)
        H = torch.relu(H)
        H = self.conv2(H)
        H = torch.relu(H)
        H = self.nlgcn(H)

        A = self.dec2(H)
        A = torch.matmul(H, H.T)
        A = torch.sigmoid(A)

        H = self.dense1(H)
        H = torch.relu(H)
        H = self.dense2(H)
        H = torch.relu(H)

        Att = self.conv3(H)
        Att = torch.relu(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att, 1)
        return Att, A


class Net1(torch.nn.Module):
    def __init__(self, A, nfeat, nhid1, nhid2, nhid3, nout):
        super(Net1, self).__init__()
        self.conv1 = GCNConv(A, nfeat, nhid1)
        self.conv2 = GCNConv(A, nhid1, nhid2)
        self.nlgcn = NLGCN(nhid2)
        self.sd_conv1 = GCNConv(A, nhid2, nhid2)
        self.sd_conv2 = GCNConv(A, nhid2, nhid3)
        self.dense1 = nn.Linear(nhid2, nhid2)
        self.dense2 = nn.Linear(nhid2, nhid2)
        self.conv3 = GCNConv(A, nhid2, nhid3)
        self.conv4 = GCNConv(A, nhid3, nout)
        # self.dec2 = GCNConv(A,nhid3,nhid3)
        self.dec2 = nn.Linear(nhid3, nhid3)

    def forward(self, X):
        H = self.conv1(X)
        H = torch.relu(H)
        H = self.conv2(H)
        H = torch.relu(H)
        H = self.nlgcn(H)

        A = self.sd_conv1(H)
        A = torch.relu(A)
        A = self.sd_conv2(A)
        A = torch.sigmoid(A)
        A = self.dec2(A)
        A = torch.matmul(A, A.T)
        A = torch.sigmoid(A)

        H = self.dense1(H)
        H = torch.sigmoid(H)
        H = self.dense2(H)
        H = torch.sigmoid(H)

        Att = self.conv3(H)
        Att = torch.sigmoid(Att)
        Att = self.conv4(Att)
        Att = torch.softmax(Att, 1)
        return Att, A


class Net2(torch.nn.Module):
    def __init__(self, A, nfeat, nhid1, nhid2, nhid3, nout):
        super(Net2, self).__init__()
        self.encGcn1 = GCNConv(A, nfeat, nhid1)
        self.encGcn2 = GCNConv(A, nhid1, nhid2)
        self.attGcn1 = GCNConv(A, nhid2, nhid1)
        self.attGcn2 = GCNConv(A, nhid1, nfeat)
        self.strGcn1 = GCNConv(A, nhid2, nhid1)

    def forward(self, X):
        H = self.encGcn1(X)
        H = torch.relu(H)

        H = self.encGcn2(H)
        H = torch.relu(H)

        A = self.strGcn1(H)
        A = torch.relu(A)
        A = torch.mm(A, A.T)
        A = torch.sigmoid_(A)

        Att = self.attGcn1(H)
        Att = torch.relu(Att)
        Att = self.attGcn2(Att)
        Att = torch.relu(Att)
        return Att, A

class Net3(torch.nn.Module):
    def __init__(self, A, nfeat, nhid1, nhid2, nhid3, nout):
        super(Net3, self).__init__()
        self.encGcn1 = GCNConv(A, nfeat, nhid1)
        self.encGcn2 = GCNConv(A, nhid1, nhid2)
        self.encNlgcn1 = NLGCN(nhid1)
        self.encNlgcn2 = NLGCN(nhid2)
        self.attGcn1 = GCNConv(A, nhid2, nhid1)
        self.attNlGcn1 = NLGCN(nhid1)
        self.attGcn2 = GCNConv(A, nhid1, nfeat)
        self.strGcn1 = GCNConv(A, nhid2, nhid1)
 
    def forward(self, X):
        H = self.encGcn1(X)
        H = torch.relu(H)
 
        H = self.encNlgcn1(H)
        # H = torch.relu(H)
 
        H = self.encGcn2(H)
        H = torch.relu(H)
 
        H = self.encNlgcn2(H)
        # H = torch.relu(H)
 
        A = self.strGcn1(H)
        A = torch.relu(A)
        A = torch.mm(A, A.T)
        A = torch.sigmoid(A)
 
        Att = self.attGcn1(H)
        Att = torch.relu(Att)
        Att = self.attGcn2(Att)
        Att = torch.relu(Att)
        return Att, A

def get_model(model_name, dataset, Adj_norm):
    if model_name == "N1":
        if dataset == "Enron":
            return Net(Adj_norm, 18, 15, 12, 15, 18)
        if dataset == "Amazon":
            return Net(Adj_norm, 21, 12, 10, 12, 21)
        if dataset == "facebook":
            return Net(Adj_norm, 10, 7, 5, 7, 10)
        if dataset == "Disney":
            return Net(Adj_norm, 28, 15, 10, 15, 28)
        if dataset == "twitter":
            return Net(Adj_norm, 15, 11, 7, 11, 15)
    elif model_name == "N2":
        if dataset == "Enron":
            return Net1(Adj_norm, 18, 15, 12, 15, 18)
        if dataset == "Amazon":
            return Net1(Adj_norm, 21, 12, 10, 12, 21)
        if dataset == "facebook":
            return Net1(Adj_norm, 10, 7, 5, 7, 10)
        if dataset == "Disney":
            return Net1(Adj_norm, 28, 15, 10, 15, 28)
        if dataset == "twitter":
            return Net1(Adj_norm, 15, 8, 6, 8, 15)
    elif model_name == "N3":
        if dataset == "Enron":
            return Net2(Adj_norm, 18, 15, 12, 15, 18)
        if dataset == "Amazon":
            return Net2(Adj_norm, 21, 12, 10, 12, 21)
        if dataset == "facebook":
            return Net2(Adj_norm, 10, 7, 5, 7, 10)
        if dataset == "Disney":
            return Net2(Adj_norm, 28, 15, 10, 15, 28)
        if dataset == "twitter":
            return Net2(Adj_norm, 15, 11, 7, 11, 15)
    elif model_name == "N4":
        if dataset == "Enron":
            return Net2(Adj_norm, 18, 256, 64, 256, 18)
        if dataset == "Amazon":
            return Net2(Adj_norm, 21, 256, 64, 256, 21)
        if dataset == "facebook":
            return Net2(Adj_norm, 10, 256, 64, 256, 10)
        if dataset == "Disney":
            return Net2(Adj_norm, 28, 256, 64, 256, 28)
        if dataset == "twitter":
            return Net2(Adj_norm, 15, 256, 64, 256, 15)
    elif model_name == "N5":
        if dataset == "Enron":
            return Net2(Adj_norm, 18, 256, 64, 256, 18)
        if dataset == "Amazon":
            return Net2(Adj_norm, 21, 256, 64, 256, 21)
        if dataset == "facebook":
            return Net2(Adj_norm, 10, 256, 64, 256, 10)
        if dataset == "Disney":
            return Net2(Adj_norm, 28, 64, 32, 64, 28)
        if dataset == "twitter":
            return Net2(Adj_norm, 15, 256, 64, 256, 15)
    elif model_name == "N6":
        if dataset == "Enron":
            return Net3(Adj_norm, 18, 256, 64, 256, 18)
        if dataset == "Amazon":
            return Net3(Adj_norm, 21, 256, 64, 256, 21)
        if dataset == "facebook":
            return Net3(Adj_norm, 10, 256, 64, 256, 10)
        if dataset == "Disney":
            return Net3(Adj_norm, 28, 64, 32, 64, 28)
        if dataset == "twitter":
            return Net3(Adj_norm, 15, 256, 64, 256, 15)
    else:
        model = None
        if model_name.startswith("DenseNet"):
            model = get_models_dense(model_name)
        elif model_name.startswith("AttConvNet"):
            model = get_models_attconv(model_name)
        elif model_name.startswith("ConvNet"):
            model = get_models_conv(model_name)
        elif model_name.startswith("NLNet"):
            model = get_models_nlgcn(model_name)
        else:
            raise NameError(model + " does not exist!")
        if dataset == "Enron":
            return model(Adj_norm, 18, 15, 12, 15, 18)
        if dataset == "Amazon":
            return model(Adj_norm, 21, 12, 10, 12, 21)
        if dataset == "facebook":
            return model(Adj_norm, 10, 7, 5, 7, 10)
        if dataset == "Disney":
            return model(Adj_norm, 28, 15, 10, 15, 28)
        if dataset == "twitter":
            return Net(Adj_norm, 15, 64 * 4, 64, 64 * 4, 15)
