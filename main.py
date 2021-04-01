from train import *

def test1():
    for dataset in ['Disney','Amazon','facebook','Enron']:
        train('N1',dataset,logging=False,epochs=100,saveResults=False)

def test2():
    for dataset in ['Disney','Amazon','facebook']:
        train("N2",dataset,epochs=150, saveResults=True,optimThreshould=True,filterFraction=0.25)
    train("N2","Enron",epochs=70, saveResults=True,optimThreshould=True,filterFraction=0.25)

def test_hyperparam_search():
    datasets=['Disney','Amazon','facebook']
    models1=["AttConvNet"+str(i) for i in range(1,5)]
    models2=["ConvNet"+str(i) for i in range(1,7)]
    models3=["DenseNet"+str(i) for i in range(1,9)]
    models4=["NLNet"+str(i) for i in range(1,4)]
    models=[]
    models.extend(models1)
    models.extend(models2)
    models.extend(models3)
    models.extend(models4)

    for dataset in datasets:
        for model in models:
            train(model,dataset,epochs=150, saveResults=True,optimThreshould=True,filterFraction=0.25)
    for model in models:
            train(model,'Enron',epochs=120, saveResults=True,optimThreshould=True,filterFraction=0.25)

if __name__=="__main__":
    test2()
    # train('N2','Enron',logging=False,epochs=10)
    # train('AttConvNet2','Disney',logging=False,epochs=160, saveResults=True)
    #test_hyperparam_search()