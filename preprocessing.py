from data_reader import get_raw_data
import torch
import networkx as nx
import numpy as np

def get_data(dataset):
    data=get_raw_data(dataset)

    Adj_norm = np.zeros(data['adj_norm'][2])
    i=0
    for cord in data['adj_norm'][0]:
        Adj_norm[cord[0]][cord[1]]=data['adj_norm'][1][i]
        i+=1
    Adj_norm=torch.Tensor(Adj_norm)

    X=np.zeros(shape=data['features'][2],dtype='float32')
    i=0
    for cord in data['features'][0]:
        X[cord[0]][cord[1]]=data['features'][1][i]
        i+=1
    # if dataset=='Amazon':
    #     X = X/X.max(axis=0)
    if dataset=="Amazon" or dataset=="Disney":
        X=(X-X.mean(axis=0)+1)/(np.sqrt(X.var(axis=0)+0.001))
    else:
        X=(X-X.mean(axis=0))/(np.sqrt(X.var(axis=0)+0.001))
    X=torch.tensor(X)

    Adj=data['adj'].toarray()
    Adj.astype('float32')
    Adj=torch.Tensor(Adj)

    return Adj_norm, Adj, X, data['labels']