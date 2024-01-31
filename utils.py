import numpy as np


def softmaxLoss(X,W,C):
    # Calculate the Soft-Max loss function according to X with W weights and indicator C
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
 
    F = 0
    for k in range (C.shape[0]):  # C.shape is supposed to be the number of classes
        F += np.matmul(np.transpose(C[k, :]) , np.log(calculateDiag(X,W,k)))
    
    F = (-1/(X.shape[1])) * F
    return F
    

def calculateDiag(X,W,k):

    # Calculate the diagonal multapliction from page 15 in the class Notes,
    #   according to X with W weights, running index k.
    # Assumptions:
        # X is : (t,m)
        # W is : (t,l)
    sumOfExp = 0
    for j in range(W.shape[1]):
        sumOfExp = sumOfExp + np.exp(np.matmul(np.transpose(X),W[:,j]))
    J = np.divide(np.exp(np.matmul(np.transpose(X),W[:,k])), sumOfExp)
    return J
    


def calculateSoftMaxGradW(X,W,C):
    # Calculate the Soft-Max gradient by W,
    #   according to X with W weights, indicator C.
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
    grad = 1/X.shape[1] *np.matmul(X,calculateDiagGrad(X,W,C))
    return grad

    
def calculateDiagGrad(X,W,C):
    res = []
    for p in range(W.shape[1]):
        Jp = calculateDiag(X, W, p).reshape(-1, 1)
        res.append(Jp)
    return np.hstack(res)

def zeroTaylor(X, W, C , eps, d):
    # Calculate gradient test
    #   according to X with W weights, indicator C, eps number, d random generated normalized matrix.
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
        # d is : (t,m)

    return np.linalg.norm(abs(softmaxLoss(X, W + eps * d, C) - softmaxLoss(X, W, C)))


def gradTest(X, W, C , eps, d):
    # Calculate gradient test
    #   according to X with W weights, indicator C, eps number, d random generated normalized matrix.
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
        # d is : (t,m)

    return np.linalg.norm(abs(softmaxLoss(X, W + eps * d, C) - softmaxLoss(X, W, C) - eps * np.matmul(np.transpose(d),calculateSoftMaxGradW(X, W, C))))

def genRandArr(rows, cols):
    return np.random.rand(rows,cols)

def genRandNormArr(rows, cols):
    d = genRandArr(rows, cols)
    return d/np.linalg.norm(d)


#def SGD(W,learning_rate=0.01 , beta ):
#    W = W - learning_rate * #grad W
    


