from train import *

for dataset in ['Disney','Amazon','facebook','Enron']:
    train('N1',dataset,logging=False,epochs=150)