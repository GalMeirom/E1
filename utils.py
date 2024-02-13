import numpy as np
import scipy


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(X, W):
    preds = []
    for i in range(X.shape[1]):
        preds.append(softmax(np.matmul(np.transpose(X)[i,:].reshape(1, -1), W)))
    return np.transpose(np.vstack(preds))


def softmaxLoss(X, W, C):
    F = 0
    for k in range (C.shape[0]):  # C.shape is supposed to be the number of classes
        F += np.matmul(np.transpose(C[k, :]) , np.log(calculateDiag(X,W,k)))
    
    F = (-1/(X.shape[1])) * F
    return F

## Satellite methods for Losses

def calculateDiag(X,W,k):

    # Calculate the diagonal multapliction from page 15 in the class Notes,
    #   according to X with W weights, running index k.
    # Assumptions:
        # X is : (t,m)
        # W is : (t,l)
        # C is : (l,m)
    sumOfExp = np.zeros(np.matmul(np.transpose(X), W[:, 0]).shape)
    for j in range(W.shape[1]):
        sumOfExp = sumOfExp + np.exp(np.matmul(np.transpose(X),W[:,j]))
    J = np.divide(np.exp(np.matmul(np.transpose(X),W[:,k])), sumOfExp)
    return J
    
## Satellite methods for Gradients Functions


def calculateDiagGrad(X,W,C):
    res = []
    for p in range(W.shape[1]):
        Jp = calculateDiag(X, W, p).reshape(-1, 1) - C[p,:].reshape(-1,1)
        res.append(Jp)
    return np.hstack(res)


## Convergence Tests

def zeroTaylor(f, k, eps, d):
    if k == 4:
        beofre = f.forward()
        tempW = f.params[1]
        tempB = f.params[2]
        f.set(1, f.params[1] + eps * d[0])
        f.set(2, f.params[2] + eps * d[1])
        after = f.forward()
        f.set(1, tempW)
        f.set(2, tempB)
        return np.linalg.norm(abs(after - beofre))
    else:
        beofre = f.forward()
        temp = f.params[k]
        f.set(k, f.params[k] + eps * d)
        after = f.forward()
        f.set(k, temp)
        return np.linalg.norm(abs(after - beofre))


def gradTest(f, k, eps, d):
    if k == 4:
        beofre = f.forward()
        grad = f.derivTheta(d)
        tempW = f.params[1]
        tempB = f.params[2]
        f.set(1, f.params[1] + eps * d[0])
        f.set(2, f.params[2] + eps * d[1])
        after = f.forward()
        f.set(1, tempW)
        f.set(2, tempB)
        return np.linalg.norm(abs(after - beofre - eps * np.matmul(np.transpose(np.matrix.flatten(d)),grad)))
    else:
        beofre = f.forward()
        temp = f.params[k]
        f.set(k, f.params[k] + eps * d)
        after = f.forward()
        f.set(k, temp)
        grad = np.matrix.flatten(f.deriv(k))
        return np.linalg.norm(abs(after - beofre - eps * np.matmul(np.transpose(np.matrix.flatten(d)),grad)))

def nnZeroTaylor(nn, eps):
    beofre = nn.forward()
    temp = nn.layers.copy()
    for layer in nn.layers:
        d = genRandNormArr(layer.params[1].shape[0],layer.params[1].shape[1])
        layer.set(1, layer.params[1] + eps * d)
    after = nn.forward()
    nn.setLayers(temp)
    return abs(after - beofre)

def nnGradTest(nn, eps):
    beofre = nn.forward()
    temp = nn.layers.copy()
    grad = nn.grad()
    dAcc = []
    dWs = []
    dBs = []
    dW = genRandNormArr(nn.layers[-1].params[1].shape[0],nn.layers[-1].params[1].shape[1])
    dWs.append(dW)
    dAcc.append(np.matrix.flatten(dW).reshape(-1, 1))
    for i in range(len(nn.layers) - 2, -1, -1):
        dW = genRandNormArr(nn.layers[i].params[1].shape[0],nn.layers[i].params[1].shape[1])
        dWs.append(dW)
        dB = genRandNormArr(nn.layers[i].params[2].shape[0],nn.layers[i].params[2].shape[1])
        dBs.append(dB)
        dAcc.append(np.matrix.flatten(dW).reshape(-1, 1))
        dAcc.append(np.matrix.flatten(dB).reshape(-1, 1))
    dAcc = np.vstack(dAcc)
    norm = np.linalg.norm(dAcc, 2)
    dAcc = dAcc/np.linalg.norm(dAcc, 2)
    nn.layers[-1].set(1, nn.layers[-1].params[1] + eps * dWs[0]/ norm)
    for i in range(len(nn.layers) - 2, -1, -1):
        nn.layers[i].set(1, nn.layers[i].params[1] + eps * dWs[len(nn.layers) - 1 - i]/ norm)
        nn.layers[i].set(2, nn.layers[i].params[2] + eps * dBs[len(nn.layers) - 2 - i]/ norm)
    after = nn.forward()
    nn.layers[-1].set(1, nn.layers[-1].params[1] - eps * dWs[0]/ norm)
    for i in range(len(nn.layers) - 2, -1, -1):
        nn.layers[i].set(1, nn.layers[i].params[1] - eps * dWs[len(nn.layers) - 1 - i]/ norm)
        nn.layers[i].set(2, nn.layers[i].params[2] - eps * dBs[len(nn.layers) - 2 - i]/ norm)
    a = nn.forward()
    # print(beofre - a)
    # print(f'without grad: {after - beofre}')
    # print(f'This is what grad helps: {eps * np.matmul(np.transpose(dAcc), grad)}')
    #nn.setLayers(temp)
    return abs(after - beofre - eps * np.matmul(np.transpose(dAcc), grad)[0][0])

## Utils

def genRandArr(rows, cols):
    return np.random.rand(rows,cols)

def genRandNormArr(rows, cols):
    d = genRandArr(rows, cols)
    return d/np.linalg.norm(d, 2)

def elmWiseMult(v, W):
    return np.vstack(list(map(lambda x:
                                    np.matmul(W, x),
                                        np.vsplit(v,v.shape[0]/W.shape[1]))))

## Optimizations


def SGD(f, k, LR):
    f.set(k, f.params[k] - LR * f.deriv(k))
    