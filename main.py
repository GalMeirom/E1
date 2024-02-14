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
    
    d = ut.genRandNormArr(sm.params[1].shape[0], sm.params[1].shape[1])
    for e in range(1, 100 ,1):
        delta = 0.01*e
        t0 = ut.zeroTaylor(sm, 1, delta, d)
        t1 = ut.gradTest(sm, 1, delta, d)
        x.append(delta) 
        Yt0.append(t0)
        Yt1.append(t1)
    plt.plot(x, Yt0,label = 'Taylor 0 expansion' ,color = 'red')
    plt.plot(x, Yt1,label = 'Taylor 1 expansion' ,color = 'blue')
    plt.xlabel('Epsilon')
    plt.ylabel('Soft Max Loss')
    plt.title('Soft-Max Loss as a function of epsilon')
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
    numBatches = 500
    sizeSubSemp = 800
    numEpoch = 200
    
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
            Xtbatches[i] = ut.Padx(Xtbatches[i])
            sm.set(0, Xtbatches[i])
            sm.set(2,Ctbatches[i])
            ut.SGD(sm, 1, 0.001)
    
        # Preping training data for graph
        xGraph.append(epoch)
        SubSampPrecent = 0
        selected_indices = np.random.choice(Xt.shape[1], sizeSubSemp, replace=False)
        SubSampXt = Xt[:, selected_indices]
        SubSampCt = Ct[:, selected_indices]
        tempSubSampXt = np.vstack([SubSampXt, np.ones((1,SubSampXt.shape[1]))])
        SubSampRes = ut.predict(tempSubSampXt, sm.params[1])
        for j in range(1, sizeSubSemp):
            if SubSampCt[np.argmax(SubSampRes[:, j]), j] == 1:
               SubSampPrecent = SubSampPrecent + 1
        yTGraphPrecent.append(100*SubSampPrecent/sizeSubSemp)
        SubSampXt = ut.Padx(SubSampXt)
        sm.set(0, SubSampXt)
        sm.set(2, SubSampCt)
        yTGraphSoftMax.append(sm.forward())
        
        # preping validation data for graph
        SubSampPrecent = 0
        selected_indices = np.random.choice(Xv.shape[1], sizeSubSemp, replace=False)
        SubSampXv = Xv[:, selected_indices]
        SubSampCv = Cv[:, selected_indices]
        tempSubSampXv = np.vstack([SubSampXv, np.ones((1,SubSampXv.shape[1]))])
        SubSampRes = ut.predict(tempSubSampXv, sm.params[1])
        for j in range(1, sizeSubSemp):
            if SubSampCv[np.argmax(SubSampRes[:, j]), j] == 1:
               SubSampPrecent = SubSampPrecent + 1
        yVGraphPrecent.append(100*SubSampPrecent/sizeSubSemp)
        if (epoch % 50 == 0 ):
                print(f'Epoch: {epoch+1}   |   Loss: {sm.forward()}   |   Accuracy:  {yVGraphPrecent[epoch-1]}')
        SubSampXv = ut.Padx(SubSampXv)
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
    

def Q2C1sc1(data, width):
    mat = scipy.io.loadmat('data/'+data+'.mat')
    Xt = mat['Yt']
    Xt = Xt[:,0].reshape(-1, 1)
    layer = f.layerFunc([Xt, ut.genRandArr(width, Xt.shape[0]), ut.genRandArr(width, 1)], f.Tanh())
    x = []
    Yt0 = []
    Yt1s = []
    var = ['X', 'W', 'b']
    Yt1s.append([])
    Yt1s.append([])
    Yt1s.append([])
    d = []
    d.append(ut.genRandNormArr(layer.params[0].shape[0],layer.params[0].shape[1]))
    d.append(ut.genRandNormArr(layer.params[1].shape[0],layer.params[1].shape[1]))
    d.append(ut.genRandNormArr(layer.params[2].shape[0],layer.params[2].shape[1]))
    for e in range(1, 100 ,1):
        delta = 0.01*e
        t0, tX = ut.testsLayer(layer, 0, delta, d[0])
        x.append(delta)
        Yt0.append(t0)
        Yt1s[0].append(tX)
        for i in range(1, len(layer.params)):
            t0, t = ut.testsLayer(layer, i, delta, d[i])
            Yt1s[i].append(t)            
    
    for i in range(len(layer.params)):
        plt.subplot(1, len(layer.params)+1, i+1)
        plt.plot(x, Yt0,label = 'Taylor 0 expansion' ,color = 'red')
        plt.plot(x, Yt1s[i],label = f'Taylor 1 expansion of {var[i]}' ,color = 'blue')
        plt.xlabel('Epsilon')
        plt.ylabel('Norm of the Error')
        plt.title(f'Norm of the Error in variable {var[i]}')
        plt.legend()
    plt.show()

def Q2C1sc2(data, width):
    mat = scipy.io.loadmat('data/'+data+'.mat')
    Xt = mat['Yt']
    Xt = Xt[:,0].reshape(-1, 1)
    layer = f.layerFunc([Xt, ut.genRandArr(width, Xt.shape[0]), ut.genRandArr(width, 1)], f.Tanh())
    d = []
    for i in range(len(layer.params)):
        grad = layer.deriv(i)
        d.append([ut.genRandNormArr(layer.params[i].shape[0],layer.params[i].shape[1]), ut.genRandNormArr(grad.shape[0],1)])

    for i in range(len(layer.params)):
        currD = d[i]
        v = np.matrix.flatten(currD[0]).reshape(-1, 1)
        u = currD[1]
        left = np.matmul(np.transpose(u), np.matmul(layer.deriv(i), v))
        right = np.matmul(np.transpose(v), np.matrix.flatten(layer.derivT(i, u)).reshape(-1, 1))
        print(abs(left - right))

def Q2C2sc1(data, width):
    mat = scipy.io.loadmat('data/'+data+'.mat')
    Xt = mat['Yt']
    Xt = Xt[:,0].reshape(-1, 1)
    layer = f.ResidlayerFunc([Xt, ut.genRandArr(width, Xt.shape[0]), ut.genRandArr(width, 1), ut.genRandArr(Xt.shape[0], width)], f.Tanh())
    x = []
    Yt0 = []
    Yt1s = []
    var = ['X', 'W1', 'b', 'W2']
    Yt1s.append([])
    Yt1s.append([])
    Yt1s.append([])
    Yt1s.append([])
    d = []
    d.append(ut.genRandNormArr(layer.params[0].shape[0],layer.params[0].shape[1]))
    d.append(ut.genRandNormArr(layer.params[1].shape[0],layer.params[1].shape[1]))
    d.append(ut.genRandNormArr(layer.params[2].shape[0],layer.params[2].shape[1]))
    d.append(ut.genRandNormArr(layer.params[3].shape[0],layer.params[3].shape[1]))
    for e in range(1, 100 ,1):
        delta = 0.01*e
        t0, tX = ut.testsLayer(layer, 0, delta, d[0])
        x.append(delta)
        Yt0.append(t0)
        Yt1s[0].append(tX)
        for i in range(1, len(layer.params)):
            t0, t = ut.testsLayer(layer, i, delta, d[i])
            Yt1s[i].append(t)            
    
    for i in range(len(layer.params)):
        plt.subplot(1, len(layer.params)+1, i+1)
        plt.plot(x, Yt0,label = 'Taylor 0 expansion' ,color = 'red')
        plt.plot(x, Yt1s[i],label = f'Taylor 1 expansion of {var[i]}' ,color = 'blue')
        plt.xlabel('Epsilon')
        plt.ylabel('Norm of the Error')
        plt.title(f'Norm of the Error in variable {var[i]}')
        plt.legend()
    plt.show()

def Q2C2sc2(data, width):
    mat = scipy.io.loadmat('data/'+data+'.mat')
    Xt = mat['Yt']
    Xt = Xt[:,0].reshape(-1, 1)
    layer = f.ResidlayerFunc([Xt, ut.genRandArr(width, Xt.shape[0]), ut.genRandArr(width, 1), ut.genRandArr(Xt.shape[0], width)], f.Tanh())
    d = []
    for i in range(len(layer.params)):
        grad = layer.deriv(i)
        d.append([ut.genRandNormArr(layer.params[i].shape[0],layer.params[i].shape[1]), ut.genRandNormArr(grad.shape[0],1)])

    for i in range(len(layer.params)):
        currD = d[i]
        v = np.matrix.flatten(currD[0]).reshape(-1, 1)
        u = currD[1]
        left = np.matmul(np.transpose(u), np.matmul(layer.deriv(i), v))
        right = np.matmul(np.transpose(v), np.matrix.flatten(layer.derivT(i, u)).reshape(-1, 1))
        print(abs(left - right))

def Q2C3(data):
    mat = scipy.io.loadmat('data/'+data+'.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    Xt = Xt[:,0].reshape(-1, 1)
    Ct = Ct[:,0].reshape(-1, 1)
    net = nn.nn(Xt, [32, 5], Ct, 0.01, Xt.shape[1])
    x = []
    Yt0 = []
    Yt1 = []
    dAcc = []
    dWs = []
    dBs = []
    dW = ut.genRandArr(net.layers[-1].params[1].shape[0],net.layers[-1].params[1].shape[1])
    dWs.append(dW)
    dAcc.append(np.matrix.flatten(dW).reshape(-1, 1))
    for i in range(len(net.layers) - 2, -1, -1):
        dW = ut.genRandArr(net.layers[i].params[1].shape[0],net.layers[i].params[1].shape[1])
        dWs.append(dW)
        dB = ut.genRandArr(net.layers[i].params[2].shape[0],net.layers[i].params[2].shape[1])
        dBs.append(dB)
        dAcc.append(np.matrix.flatten(dW).reshape(-1, 1))
        dAcc.append(np.matrix.flatten(dB).reshape(-1, 1))
    dAcc = np.vstack(dAcc)
    norm = np.linalg.norm(dAcc)
    dAcc = dAcc/norm
    for e in range(1, 100 ,1):
        delta =  0.01 * e
        t0, t1 = ut.nnTest(net, delta, dWs, dBs, dAcc, norm)
        x.append(delta)
        Yt0.append(t0)
        Yt1.append(t1)
    plt.plot(x, Yt0,label = 'Taylor 0 expansion' ,color = 'red')
    plt.plot(x, Yt1,label = 'Taylor 1 expansion' ,color = 'blue')
    plt.xlabel('Epsilon')
    plt.ylabel('Soft Max Loss')
    plt.title('Soft-Max Loss as a function of epsilon')
    plt.legend()
    plt.show()

def Q2C4(data, dims, LR):
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

def Q2C5(data, dims, LR):
    mat = scipy.io.loadmat('data/' + data + '.mat')
    
    # training labels
    Ct = mat['Ct']
    # training data
    Xt = mat['Yt']
    # val labels
    Cv = mat['Cv']
    # val data
    Xv = mat['Yv']
    selected_indices = np.random.choice(Xt.shape[1], 200, replace=False)
    Xt = Xt[:, selected_indices]
    Ct = Ct[:, selected_indices]
    numBatches = 10
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

# Q2C1sc1(Data[1], 3)
# Q2C1sc2(Data[1], 3)
# Q2C2sc1(Data[2], 3)
# Q2C2sc2(Data[2], 3)
# Q2C3(Data[2])
# Q2C4(Data[1], [32, 32, 5], 0.1)
# Q2C5(Data[1], [32, 32, 5], 0.1)


#Q1C3
# for each data set we used differnt hyper paramters that we found that work best.
# for data set [0] 
# for data set [1] PeaksData learning rate: 0.001, epochs: 100.
# for data set [2] GMMData learning rate: 0.001, epochs: 200.

#Q2C3
# for data set [2] learning rate 0.05, epochs 280, 1 hidden layer, 32/64 hidden units
# for data set [1] learning rate 0.1, epoches 1000,1 hidden layer, 64 hidden units