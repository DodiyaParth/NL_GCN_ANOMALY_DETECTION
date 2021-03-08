import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from preprocessing import get_data
from models import get_model
from tqdm import tqdm

def train(modelname,dataset,lr=0.01,logging=False,epochs=100,show_ROC=False):
    Adj_norm,Adj,X,labels = get_data(dataset)
    model = get_model(modelname,dataset,Adj_norm)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    loss_list=[]
    iterable=range(epochs)
    if logging==False:
        iterable=tqdm(range(epochs),desc=dataset)
    for i in iterable:
        optimizer.zero_grad()
        Att,A=model(X)
        feat_loss=torch.sqrt(torch.sum(torch.square(Att-X)))
        struct_loss=torch.sqrt(torch.sum(torch.square(A-Adj)))
        loss = torch.add(torch.mul(feat_loss,0.5),torch.mul(struct_loss,0.5))
        loss.backward()
        optimizer.step()
        l=(model(X))
        loss_list.append(loss.item())
        if logging:
            if i>3:
                if loss_list[-1]==loss_list[-2] and loss_list[-2]==loss_list[-3]:
                    print("\n")
                    print("Epoch : ",i," Loss: =", loss.item(), "Struct_loss = ", struct_loss.item(), "Feature_loss = ",feat_loss.item())
                    break
            if i%5==0:
                print("Epoch : ",i," Loss: =", loss.item(), "Struct_loss = ", struct_loss.item(), "Feature_loss = ",feat_loss.item())
        else:
            pass
        
    
    with torch.no_grad():
        feat_loss=torch.square(model(X)[0]-X)
        fl=[]
        for i in feat_loss:
            fl.append(np.sqrt(torch.sqrt(torch.sum(i))))
        struct_loss=torch.square(model(X)[1]-Adj)
        sl=[]
        for i in struct_loss:
            sl.append(np.sqrt(torch.sqrt(torch.sum(i))))
        diff=[]
        for i in range(len(feat_loss)):
            diff.append((fl[i]+sl[i])/2)
    fpr, tpr, thresholds = metrics.roc_curve(labels, diff, pos_label=None,drop_intermediate=False)
    auc_score=roc_auc_score(labels, diff)
    print(dataset+" AUC score : ",auc_score)
    plt.title(dataset+" AUC_score="+str(auc_score))
    plt.plot(fpr,tpr)
    if show_ROC:
        plt.show()
    return auc_score