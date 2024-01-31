import numpy as np
import scipy
import utils as ut
import matplotlib.pyplot as plt

def Q1C1():
    mat = scipy.io.loadmat('data/SwissRollData.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    softmax = ut.softmaxLoss(Xt,W,Ct)
    print(f'This is Soft-Max: {softmax}')
    x = []
    Yt0 = []
    Yt1 = []
    d = ut.genRandNormArr(W.shape[0],W.shape[1])
    for e in range(1, 200 ,1):
        t0 = ut.zeroTaylor(Xt,W,Ct, 0.1*e, d)
        t1 = ut.gradTest(Xt,W,Ct, 0.1*e, d)
        x.append(e)
        Yt0.append(t0)
        Yt1.append(t1)
    plt.plot(x, Yt0,label = 'Tylor 0 expansion' ,color = 'red')
    plt.plot(x, Yt1,label = 'Tylor 1 expansion' ,color = 'blue')
    plt.xlabel('Epsilon-axis')
    plt.ylabel('Tests-axis')
    plt.title('Tests score as a function of epsilon')
    plt.legend()
    plt.show()

def Q1C2():
    mat = scipy.io.loadmat('data/SwissRollData.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    softmax = ut.softmaxLoss(Xt,W,Ct)
    t = ut.gradTest(Xt,W,Ct, 0.001, ut.genRandNormArr(Xt.shape[0],Xt.shape[1]))

Q1C1()
