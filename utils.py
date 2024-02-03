import numpy as np
import scipy

## Satellite methods for Losses

def calculateDiag(X,W,k):

    # Calculate the diagonal multapliction from page 15 in the class Notes,
    #   according to X with W weights, running index k.
    # Assumptions:
        # X is : (t,m)
        # W is : (t,l)
    sumOfExp = np.zeros(np.matmul(X.T, W[:, 0]).shape)
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
    return np.linalg.norm(abs(after - beofre - eps * np.matmul(np.transpose(d),f.deriv(k))))

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
    grad = nn.grad()
    dAcc = []
    for layer in nn.layers:
        d = genRandNormArr(layer.params[1].shape[0],layer.params[1].shape[1])
        dAcc.append(d)
        layer.set(1, layer.params[1] + eps * d)
    dAcc = np.vstack(dAcc)
    after = nn.forward()
    nn.setLayers(temp)
    return np.linalg.norm(abs(after - beofre - eps * np.matmul(np.transpose(dAcc), grad)))

## Utils

def genRandArr(rows, cols):
    return np.random.rand(rows,cols)

def genRandNormArr(rows, cols):
    d = genRandArr(rows, cols)
    return d/np.linalg.norm(d, 2)

## Optimizations


def SGD(f, k, LR):
    f.set(k, f.params[k] - LR * f.deriv(k))
    