from train import *

def test1():
    for dataset in ['Disney','Amazon','facebook','Enron']:
        train('N1',dataset,logging=False,epochs=100)

def test2():
    for dataset in ['Disney','Amazon','facebook','Enron']:
        train('N2',dataset,logging=False,epochs=160)

test2()
# train('N2','Enron',logging=False,epochs=10)