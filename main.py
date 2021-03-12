from train import *

def test1():
    for dataset in ['Disney','Amazon','facebook','Enron']:
        train('N1',dataset,logging=False,epochs=100,saveResults=False)

def test2():
    for dataset in ['Disney','Amazon','facebook','Enron']:
        train('N2',dataset,logging=False,epochs=160, saveResults=False)

# test2()
# train('N2','Enron',logging=False,epochs=10)
train('N2','Disney',logging=False,epochs=160, saveResults=True)
