import numpy as np
import scipy
import utils as ut
import matplotlib.pyplot as plt

##Q1

def Q1C1(data):
    mat = scipy.io.loadmat('data/'+data+'.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    params = [Xt,W,Ct]
    softmax = ut.softmaxLoss(params)
    print(f'This is Soft-Max: {softmax}')
    x = []
    Yt0 = []
    Yt1 = []
    d = ut.genRandNormArr(W.shape[0],W.shape[1])
    for e in range(1, 200 ,1):
        t0 = ut.zeroTaylor(Xt,W,Ct, 0.1*e, d)
        t1 = ut.gradTest(params, ut.softmaxLoss, ut.calculateSMGradW, 0.1*e, d)
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
    A = ut.genRandArr(8,6)
    X = ut.genRandArr(6,7)
    b = ut.genRandArr(8,7)
    params = [A, X, b]
    x = []
    err = []
    for j in range(1, 15):
        err.append(ut.linearLeastSquares(params))
        x.append(j)
        params = ut.SGD(ut.calculateLLSGradX, params, 0.1)
    plt.plot(x, err ,color = 'blue')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Linear Least Squares Error')
    plt.title('Testing Stochastic Gradient Descent')
    plt.legend()
    plt.show()
    

def Q1C3(data):
    mat = scipy.io.loadmat('data/' + data + '.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    Cv = mat['Cv']
    Xv = mat['Yv']
    batchS = 400
    numEpoch = 10
    xGraph = []
    yGraphPrecent = []
    yGraphSoftMax = []
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    params = [Xt,W,Ct]
    for epoch in range(1, numEpoch):
        xGraph.append(epoch)
        for i in range(1, 1000):
            params = ut.SGD(ut.calculateSMGradW, params, 0.1)
        batchPrecent = 0
        selected_indices = np.random.choice(Xt.shape[1], batchS, replace=False)
        batchXt = Xt[:, selected_indices]
        batchCt = Ct[:, selected_indices]
        batchRes = np.matmul(np.transpose(W), batchXt)
        for j in range(1, batchS):
            if batchCt[np.argmax(batchRes[:, j]), j] == 1:
               batchPrecent = batchPrecent + 1
        yGraphPrecent.append(100*batchPrecent/batchS)
        yGraphSoftMax.append(ut.softmaxLoss([batchXt, W, batchCt]))
    batchPrecent = 0
    selected_indices = np.random.choice(Xv.shape[1], batchS, replace=False)
    batchXv = Xv[:, selected_indices]
    batchCv = Cv[:, selected_indices]
    batchResV = np.matmul(np.transpose(W), batchXv)
    for m in range(1, batchS):
        if batchCv[np.argmax(batchResV[:, m]), m] == 1:
            batchPrecent = batchPrecent + 1
    xGraph.append(numEpoch)
    yGraphPrecent.append(100*batchPrecent/batchS)
    yGraphSoftMax.append(ut.softmaxLoss([batchXv, W, batchCv]))
    plt.scatter(xGraph, yGraphPrecent, c='blue', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.scatter(xGraph[numEpoch-1], yGraphPrecent[numEpoch-1], c='red', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.text(xGraph[numEpoch-1], yGraphPrecent[numEpoch-1], "Test Batch", fontsize=8, ha='right', va='bottom')
    plt.xlabel('Amount of ephocs')
    plt.ylabel('Precent of accuracy')
    plt.title('Precent from different batches by different ephocs')
    plt.show()

            
Data = ['SwissRollData','PeaksData','GMMData']            
Q1C3(Data[2])


