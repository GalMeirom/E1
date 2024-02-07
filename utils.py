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
    beofre = f.forward()
    temp = f.params[k]
    f.set(k, f.params[k] + eps * d)
    after = f.forward()
    f.set(k, temp)
    return np.linalg.norm(abs(after - beofre))


def gradTest(f, k, eps, d):
    beofre = f.forward()
    temp = f.params[k]
    f.set(k, f.params[k] + eps * d)
    after = f.forward()
    f.set(k, temp)
    grad = f.deriv(k)
    return np.linalg.norm(abs(after - beofre - eps * np.matmul(np.transpose(d),grad)))

def nnZeroTaylor(nn, eps):
    beofre = nn.forward()
    temp = nn.layers.copy()
    for layer in nn.layers:
        d = genRandNormArr(layer.params[1].shape[0],layer.params[1].shape[1])
        layer.set(1, layer.params[1] + eps * d)
    after = nn.forward()
    nn.setLayers(temp)
    return np.linalg.norm(abs(after - beofre))

def nnGradTest(nn, eps):
    beofre = nn.forward()
    temp = nn.layers.copy()
    dAcc = []
    dWs = []
    dBs = []
    m = nn.layers[-1].params[1].shape[1]
    dW = genRandArr(nn.layers[-1].params[1].shape[0],nn.layers[-1].params[1].shape[1])
    dWs.append(dW)
    for i in range(len(nn.layers) - 2, -1, -1):
        if nn.layers[i].params[2].shape[1] + nn.layers[i].params[1].shape[1] > m:
            m = nn.layers[i].params[2].shape[1] + nn.layers[i].params[1].shape[1]
        dW = genRandArr(nn.layers[i].params[1].shape[0],nn.layers[i].params[1].shape[1])
        dWs.append(dW)
        dB = genRandArr(nn.layers[i].params[2].shape[0],nn.layers[i].params[2].shape[1])
        dBs.append(dB)
    dAcc.append(np.pad(dWs[0], ((0, 0), (0, m - dWs[0].shape[1])), mode='constant', constant_values=0))
    for j in range(len(dBs)):
        temp = np.hstack(dWs[j+1], dBs[j+1])
        dAcc.append(np.pad(temp, (0, 0), (0, m - temp.shape[1]), mode='constant', constant_values=0))
    dAcc = np.vstack(dAcc)
    norm = np.linalg.norm(dAcc, 2)
    dAcc = dAcc/np.linalg.norm(dAcc, 2)
    grad = nn.grad(m)
    nn.layers[-1].set(1, nn.layers[-1].params[1] + eps * dWs[0]/ norm)
    for i in range(len(nn.layers) - 2, -1, -1):
        nn.layers[i].set(1, nn.layers[i].params[1] + eps * dWs[len(nn.layers) - 1 - i]/ norm)
        nn.layers[i].set(2, nn.layers[i].params[2] + eps * dBs[len(nn.layers) - 2 - i]/ norm)
    after = nn.forward()
    nn.setLayers(temp)
    return np.linalg.norm(abs(after - beofre - eps * np.matmul(np.transpose(dAcc), grad)))

# def nnGradTest(nn, eps):
#     beofre = nn.forward()
#     temp = nn.layers.copy()
#     grad = nn.grad()
#     dAcc = []
#     dWs = []
#     dBs = []
#     dW = genRandArr(nn.layers[-1].params[1].shape[0],nn.layers[-1].params[1].shape[1])
#     dWs.append(dW)
#     dAcc.append(np.matrix.flatten(dW).reshape(-1, 1))
#     for i in range(len(nn.layers) - 2, -1, -1):
#         dW = genRandArr(nn.layers[i].params[1].shape[0],nn.layers[i].params[1].shape[1])
#         dWs.append(dW)
#         dB = genRandArr(nn.layers[i].params[2].shape[0],nn.layers[i].params[2].shape[1])
#         dBs.append(dB)
#         dAcc.append(np.matrix.flatten(dW).reshape(-1, 1))
#         dAcc.append(np.matrix.flatten(dB).reshape(-1, 1))
#     dAcc = np.vstack(dAcc)
#     norm = np.linalg.norm(dAcc, 2)
#     dAcc = dAcc/np.linalg.norm(dAcc, 2)
#     nn.layers[-1].set(1, nn.layers[-1].params[1] + eps * dWs[0]/ norm)
#     for i in range(len(nn.layers) - 2, -1, -1):
#         nn.layers[i].set(1, nn.layers[i].params[1] + eps * dWs[len(nn.layers) - 1 - i]/ norm)
#         nn.layers[i].set(2, nn.layers[i].params[2] + eps * dBs[len(nn.layers) - 2 - i]/ norm)
#     after = nn.forward()
#     nn.setLayers(temp)
#     return np.linalg.norm(abs(after - beofre - eps * np.matmul(np.transpose(dAcc), grad)))

## Utils

def genRandArr(rows, cols):
    return np.random.rand(rows,cols)

def genRandNormArr(rows, cols):
    d = genRandArr(rows, cols)
    return d/np.linalg.norm(d, 2)

## Optimizations


def SGD(f, k, LR):
    f.set(k, f.params[k] - LR * f.deriv(k))
    