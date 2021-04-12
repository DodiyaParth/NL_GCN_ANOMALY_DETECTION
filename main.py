from train import *

def test1():
    for dataset in ['Disney','Amazon','facebook','Enron']:
        train('N1',dataset,logging=False,epochs=100,saveResults=False)

def test2():
    for dataset in ['Disney','Amazon','facebook']:
        train("N5",dataset, lr=.5*0.001, epochs=300, saveResults=True,optimThreshould=True,filterFraction=0.25)
    train("N5",'twitter', lr=.0133*0.001, epochs=300, saveResults=True,optimThreshould=True,filterFraction=0.25)


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

    # for dataset in datasets:
    #     for model in models:
    #         train(model,dataset,epochs=150, saveResults=True,optimThreshould=True,filterFraction=0.25)
    for model in models:
            train(model,'twitter',epochs=120, saveResults=False,optimThreshould=True,filterFraction=0.25)

if __name__=="__main__":
    train("N5","twitter", lr=.0133*0.001, epochs=200, saveResults=True,optimThreshould=True,filterFraction=0.25)  #mean=0
    train("N5","facebook", lr=.5*0.001, epochs=100, saveResults=True,optimThreshould=True,filterFraction=0.25)    #mean=0
    train("N5","Amazon", lr=.5*0.01, epochs=100, saveResults=True,optimThreshould=True,filterFraction=0.25)         #mean=1
    train("N5","Disney", lr=.5*0.01, epochs=100, saveResults=True,optimThreshould=True,filterFraction=0.25)         #mean=1
    # train("N5","Enron", lr=0.1, epochs=40, saveResults=True,optimThreshould=True,filterFraction=0.25) 
