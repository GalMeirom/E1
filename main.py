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
        params = ut.SGD(ut.calculateLLSGradX, params, 0.3)
    plt.plot(x, err ,color = 'blue')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Linear Least Squares Error')
    plt.title('Testing Stochastic Gradient Descent')
    plt.legend()
    plt.show()
    

def Q1C3(data):
    mat = scipy.io.loadmat('data/' + data + '.mat')
    
    # training labels
    Ct = mat['Ct']
    # training data
    Xt = mat['Yt']
    # val labels
    Cv = mat['Cv']
    # val data
    Xv = mat['Yv']
    numBatches = 50
    sizeSubSemp = 800
    numEpoch = 100
    
    # Graph Lists 
    xGraph = []
    yTGraphPrecent = []
    yTGraphSoftMax = []
    yVGraphPrecent = []
    yVGraphSoftMax = []
    
    # Initializing random weights
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    params = [Xt,W,Ct]
    
    # startin epochs
    for epoch in range(1, numEpoch):
        # Training data
        XtCopy = Xt.copy()
        comb = np.vstack([XtCopy, Ct])
        np.random.shuffle(np.transpose(comb))
        temp = np.vsplit(comb, [XtCopy.shape[0]])
        shuffXt = temp[0]
        shuffCt = temp[1]
        Xtbatches = np.hsplit(shuffXt, numBatches)
        Ctbatches = np.hsplit(shuffCt, numBatches)
        temp = 0
        for i in range(len(Xtbatches)):
            temp = params[1]
            params[0] = Xtbatches[i]
            params[2] = Ctbatches[i]
            params = ut.SGD(ut.calculateSMGradW, params, 0.001)
    
        # Preping training data for graph
        xGraph.append(epoch)
        SubSampPrecent = 0
        selected_indices = np.random.choice(Xt.shape[1], sizeSubSemp, replace=False)
        SubSampXt = Xt[:, selected_indices]
        SubSampCt = Ct[:, selected_indices]
        SubSampRes = scipy.special.softmax(np.matmul(np.transpose(params[1]), SubSampXt))
        for j in range(1, sizeSubSemp):
            if SubSampCt[np.argmax(SubSampRes[:, j]), j] == 1:
               SubSampPrecent = SubSampPrecent + 1
        yTGraphPrecent.append(100*SubSampPrecent/sizeSubSemp)
        yTGraphSoftMax.append(ut.softmaxLoss([SubSampXt, params[1], SubSampCt]))

        # preping validation data for graph
        SubSampPrecent = 0
        selected_indices = np.random.choice(Xv.shape[1], sizeSubSemp, replace=False)
        SubSampXv = Xv[:, selected_indices]
        SubSampCv = Cv[:, selected_indices]
        SubSampRes = np.matmul(np.transpose(params[1]), SubSampXv)
        for j in range(1, sizeSubSemp):
            if SubSampCv[np.argmax(SubSampRes[:, j]), j] == 1:
               SubSampPrecent = SubSampPrecent + 1
        yVGraphPrecent.append(100*SubSampPrecent/sizeSubSemp)
        yVGraphSoftMax.append(ut.softmaxLoss([SubSampXv, params[1], SubSampCv]))

    plt.scatter(xGraph, yTGraphSoftMax, c='blue', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.scatter(xGraph, yVGraphSoftMax, c='red', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.xlabel('Amount of ephocs')
    plt.ylabel('Precent of accuracy')
    plt.title('Precent from different batches by different ephocs')
    plt.show()

            
Data = ['SwissRollData','PeaksData','GMMData']            
Q1C3(Data[1])

