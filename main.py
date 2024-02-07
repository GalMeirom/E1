import numpy as np
import scipy
import utils as ut
import nn
import funcs as f
import matplotlib.pyplot as plt

##Q1

def Q1C1(data):
    mat = scipy.io.loadmat('data/'+data+'.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    params = [Xt,W,Ct]
    sm = f.SoftMaxLoss(params)
    softmax = sm.forward()
    print(f'This is Soft-Max loss: {softmax}')
    x = []
    Yt0 = []
    Yt1 = []
    d = ut.genRandNormArr(W.shape[0],W.shape[1])
    for e in range(1, 200 ,1):
        t0 = ut.zeroTaylor(sm, 1, 0.1*e, d)
        t1 = ut.gradTest(sm, 1, 0.1*e, d)
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
    llr = f.LinearLeastSquares(params)
    for j in range(1, 100):
        err.append(llr.forward())
        x.append(j)
        ut.SGD(llr, 1, 0.1)
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
    numBatches = 781
    sizeSubSemp = 800
    numEpoch = 400
    
    # Graph Lists 
    xGraph = []
    yTGraphPrecent = []
    yTGraphSoftMax = []
    yVGraphPrecent = []
    yVGraphSoftMax = []
    
    # Initializing random weights
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    params = [Xt,W,Ct]
    sm = f.SoftMaxLoss(params)
    
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
            sm.set(0, Xtbatches[i])
            sm.set(2,Ctbatches[i])
            ut.SGD(sm, 1, 0.001)
    
        # Preping training data for graph
        xGraph.append(epoch)
        SubSampPrecent = 0
        selected_indices = np.random.choice(Xt.shape[1], sizeSubSemp, replace=False)
        SubSampXt = Xt[:, selected_indices]
        SubSampCt = Ct[:, selected_indices]
        SubSampRes = ut.predict(SubSampXt, sm.params[1])
        for j in range(1, sizeSubSemp):
            if SubSampCt[np.argmax(SubSampRes[:, j]), j] == 1:
               SubSampPrecent = SubSampPrecent + 1
        yTGraphPrecent.append(100*SubSampPrecent/sizeSubSemp)
        sm.set(0, SubSampXt)
        sm.set(2, SubSampCt)
        yTGraphSoftMax.append(sm.forward())

        # preping validation data for graph
        SubSampPrecent = 0
        selected_indices = np.random.choice(Xv.shape[1], sizeSubSemp, replace=False)
        SubSampXv = Xv[:, selected_indices]
        SubSampCv = Cv[:, selected_indices]
        SubSampRes = ut.predict(SubSampXv, sm.params[1])
        for j in range(1, sizeSubSemp):
            if SubSampCv[np.argmax(SubSampRes[:, j]), j] == 1:
               SubSampPrecent = SubSampPrecent + 1
        yVGraphPrecent.append(100*SubSampPrecent/sizeSubSemp)
        sm.set(0, SubSampXv)
        sm.set(2, SubSampCv)
        yVGraphSoftMax.append(sm.forward())

    plt.subplot(1,2,1)
    plt.scatter(xGraph, yTGraphPrecent, label='training', c='blue', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.scatter(xGraph, yVGraphPrecent, label='val', c='red', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.xlabel('Amount of ephocs')
    plt.ylabel('Precent of accuracy')
    plt.legend()
    plt.title('Accurercy from different sub-samples after different ephocs')
    
    plt.subplot(1,2,2)
    plt.scatter(xGraph, yTGraphSoftMax, label='training', c='blue', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.scatter(xGraph, yVGraphSoftMax, label='val', c='red', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.xlabel('Amount of ephocs')
    plt.ylabel('Soft Max Lost')
    plt.legend()
    plt.title('Soft Max Loss from different sub-samples after different ephocs')
    
    plt.tight_layout()
    plt.show()



##Q2
    

def Q2C1(data):
    mat = scipy.io.loadmat('data/'+data+'.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    net = nn.nn(Xt, [5, 5], Ct, 0.01, Xt.shape[1])
    x = []
    Yt0 = []
    Yt1 = []
    for e in range(1, 50 ,1):
        t0 = ut.nnZeroTaylor(net, 0.001 * e)
        t1 = ut.nnGradTest(net, 0.001 * e)
        x.append(e)
        Yt0.append(t0)
        Yt1.append(t1)
    plt.plot(x, Yt0,label = 'Tylor 0 expansion' ,color = 'red')
    plt.plot(x, Yt1,label = 'Tylor 1 expansion' ,color = 'blue')
    plt.xlabel('Epsilon-axis')
    plt.ylabel('Tests-axis')
    plt.title('Tests score as a function of epsilon')
    plt.show()


def Q2C3(data, dims, LR):
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
    sizeSubSemp = int(Xt.shape[1]/numBatches)
    numEpoch = 1000
    
    # Graph Lists 
    xGraph = []
    yTGraphPrecent = []
    yTGraphSoftMax = []
    yVGraphPrecent = []
    yVGraphSoftMax = []
    
    # Initializing random weights
    net = nn.nn(Xt,dims,Ct,LR, sizeSubSemp)
    
    # startin epochs
    for epoch in range(0, numEpoch):
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
            net.layers[0].set(0, Xtbatches[i])
            net.layers[-1].set(2,Ctbatches[i])
            net.forward()
            net.backProp()
    
        # Preping training data for graph
        xGraph.append(epoch)
        SubSampPrecent = 0
        selected_indices = np.random.choice(Xt.shape[1], sizeSubSemp, replace=False)
        SubSampXt = Xt[:, selected_indices]
        SubSampCt = Ct[:, selected_indices]
        net.layers[0].set(0, SubSampXt)
        SubSampRes = net.forwardPred()
        for j in range(0, sizeSubSemp):
            if SubSampCt[np.argmax(SubSampRes[:, j]), j] == 1:
               SubSampPrecent = SubSampPrecent + 1
        yTGraphPrecent.append(100*SubSampPrecent/sizeSubSemp)
        net.layers[0].set(0, SubSampXt)
        net.layers[-1].set(2, SubSampCt)
        yTGraphSoftMax.append(net.forward())

        # preping validation data for graph
        SubSampPrecent = 0
        selected_indices = np.random.choice(Xv.shape[1], sizeSubSemp, replace=False)
        SubSampXv = Xv[:, selected_indices]
        SubSampCv = Cv[:, selected_indices]
        net.layers[0].set(0, SubSampXv)
        SubSampRes = net.forwardPred()
        for j in range(0, sizeSubSemp):
            if SubSampCv[np.argmax(SubSampRes[:, j]), j] == 1:
               SubSampPrecent = SubSampPrecent + 1
        yVGraphPrecent.append(100*SubSampPrecent/sizeSubSemp)
        net.layers[0].set(0, SubSampXv)
        net.layers[-1].set(2, SubSampCv)
        if (epoch % 50 == 0 ):
                print(f'Epoch: {epoch+1}   |   Loss: {net.forward()}   |   Accuracy:  {yVGraphPrecent[epoch]}')
        yVGraphSoftMax.append(net.forward())

    plt.subplot(1,2,1)
    plt.scatter(xGraph, yTGraphPrecent, label='training', c='blue', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.scatter(xGraph, yVGraphPrecent, label='val', c='red', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.xlabel('Amount of ephocs')
    plt.ylabel('Precent of accuracy')
    plt.legend()
    plt.title('Accurercy from different sub-samples after different ephocs')
    
    plt.subplot(1,2,2)
    plt.scatter(xGraph, yTGraphSoftMax, label='training', c='blue', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.scatter(xGraph, yVGraphSoftMax, label='val', c='red', marker='o', s=50, edgecolors='black', alpha=0.7)
    plt.xlabel('Amount of ephocs')
    plt.ylabel('Soft Max Lost')
    plt.legend()
    plt.title('Soft Max Loss from different sub-samples after different ephocs')
    
    plt.tight_layout()
    plt.show()




Data = ['SwissRollData','PeaksData','GMMData']            
# Q1C1(Data[1])
# Q1C2()
# Q1C3(Data[1])

# Q2C1(Data[2])

Q2C3(Data[1], [32, 32, 5], 0.1)


#Q1C3
# for each data set we used differnt hyper paramters that we found that work best.
# for data set [0] 
# for data set [1] PeaksData learning rate: 0.001, epochs: 100.
# for data set [2] GMMData learning rate: 0.001, epochs: 200.

#Q2C3
# for data set [2] learning rate 0.05, epochs 280, 1 hidden layer, 32/64 hidden units
# for data set [1] learning rate 0.1, epoches 1000,1 hidden layer, 64 hidden units