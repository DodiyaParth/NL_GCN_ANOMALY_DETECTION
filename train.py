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
from os import path
from torchsummary import summary
import time
import json
import math

def train(modelname,dataset,lr=0.01,logging=False,epochs=100,show_ROC=False,saveResults=False,thresholding=False):
    Adj_norm,Adj,X,labels = get_data(dataset)
    model = get_model(modelname,dataset,Adj_norm)
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            fl.append(torch.sqrt(torch.sum(i)))
        struct_loss=torch.square(model(X)[1]-Adj)
        sl=[]
        for i in struct_loss:
            sl.append(torch.sqrt(torch.sum(i)))
        diff=[]
        for i in range(len(feat_loss)):
            diff.append((fl[i]+sl[i])/2)
    fpr, tpr, thresholds = metrics.roc_curve(labels, diff, pos_label=None,drop_intermediate=False)
    if thresholding:
        threshold=thresholds[0]
        min_d=10
        for i in range(len(fpr)):
            distance=math.sqrt((1-fpr[i])**2+(tpr[i])**2)
            if distance<min_d:
                min_d=distance
            threshold=thresholds[i]
        print("Optimum threshould: "+str(threshold))
    auc_score=roc_auc_score(labels, diff)
    print(dataset+" AUC score : ",auc_score)
    if saveResults:
        file_path = "Results/"+dataset+"/results.json"
        data = {}
        data[modelname] = {"auc_score":0.0,"model_summary":"fake summary"}
        if not (path.exists(file_path) and path.isfile(file_path)):
            f = open(file_path,'x')
            
        else:
            f = open(file_path)
            try:
                data1 = json.load(f)
                data =data1
            except:
                print("Issues in reading json file")
            f.close()
            f = open(file_path,'w')
            if modelname not in data.keys():
                data[modelname] = {"auc_score":0.0,"model_summary":"fake summary"}
        data[modelname]["auc_score"]=auc_score
        print(X.shape[0],X.shape[1])
        model_summary = summary(model,(X[0],X[1]),device='cpu')
        data[modelname]["model_summary"] = str(model_summary)
        f.write(json.dumps(data))
        f.close()

    fig = plt.figure()
    plt.title(dataset+" AUC_score="+str(auc_score))
    plt.plot(fpr,tpr)
    if show_ROC:
        plt.show()
    if saveResults:
        fig.savefig('./Results'+'/'+dataset+'/roc.png')
    return auc_score