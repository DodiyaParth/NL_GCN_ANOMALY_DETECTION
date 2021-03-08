from train import *

def test1():
    for dataset in ['Disney','Amazon','facebook','Enron']:
        train('N1',dataset,logging=False,epochs=100)

def test2():
    for dataset in ['Disney','Amazon','facebook']:
        train('N2',dataset,logging=False,epochs=80)

test2()
