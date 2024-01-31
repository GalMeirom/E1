import numpy as np

#### Functions
def softmaxLoss(params):
    # Calculate the Soft-Max loss function according to X with W weights and indicator C
    # Assumptions:
        # params[0] = X is : (t,m) 
        # params[1] = W is : (t,l)
        # params[2] = C is : (l,m)
    X = params[0]
    W = params[1]
    C = params[2]
    F = 0
    for k in range (C.shape[0]):  # C.shape is supposed to be the number of classes
        F += np.matmul(np.transpose(C[k, :]) , np.log(calculateDiag(X,W,k)))
    
    F = (-1/(X.shape[1])) * F
    return F

def linearLeastSquares(params):
    # Calculate the Linear Lest Squares loss function according to X, A and B bias
    # Assumptions:
        # params[0] = A is : (n,m) 
        # params[1] = X is : (m,t)
        # params[2] = b is : (n,t)
    A = params[0]
    X = params[1]
    b = params[2]
    return np.linalg.norm(np.matmul(A, X) - b, 2)*0.5


## Satellite methods for Losses

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
    

## Gradient Functions

def calculateSMGradW(params):
    # Calculate the Soft-Max gradient by W,
    #   according to X with W weights, indicator C.
    # Assumptions:
        # params[0] = X is : (t,m) 
        # params[1] = W is : (t,l)
        # params[2] = C is : (l,m)
    X = params[0]
    W = params[1]
    C = params[2]
    grad = 1/X.shape[1] *np.matmul(X,calculateDiagGrad(X,W,C))
    return grad

def calculateLLSGradX(params):
    # Calculate the LLS gradient by X,
    #   according to X with A, and b bias.
    # Assumptions:
        # params[0] = A is : (n,m) 
        # params[1] = X is : (m,t)
        # params[2] = b is : (n,t)
    A = params[0]
    X = params[1]
    b = params[2]
    return np.matmul(np.matmul(np.transpose(A), A), X) - np.matmul(np.transpose(A), b)


## Satellite methods for Gradients Functions


def calculateDiagGrad(X,W,C):
    res = []
    for p in range(W.shape[1]):
        Jp = calculateDiag(X, W, p).reshape(-1, 1)
        res.append(Jp)
    return np.hstack(res)


## Convergence Tests

def zeroTaylor(X, W, C , eps, d):
    # Calculate gradient test
    #   according to X with W weights, indicator C, eps number, d random generated normalized matrix.
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
        # d is : (t,m)

    return np.linalg.norm(abs(softmaxLoss([X, W + eps * d, C]) - softmaxLoss([X, W, C])))


def gradTest(params, func, gradFunc, eps, d):
    # Calculate gradient test
    #   according to parameters of a function, function f, gradient function of f,
    #    some small epsilon and random generated normalized matrix d.
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
        # d is : (t,m)
    shift = params.copy()
    shift[1] = shift[1] + eps * d
    return np.linalg.norm(abs(func(shift) - func(params) - eps * np.matmul(np.transpose(d),gradFunc(params))))

## Utils

def genRandArr(rows, cols):
    return np.random.rand(rows,cols)

def genRandNormArr(rows, cols):
    d = genRandArr(rows, cols)
    return d/np.linalg.norm(d)


## Optimizations


def SGD(gradFunc, params, LR):
    step = params.copy()
    step[1] = step[1] - LR * gradFunc(params)
    return step
    
     

## Plots


