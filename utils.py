import numpy as np


def softmaxLoss(X,W,C):
    # Calculate the Soft-Max loss function according to X with W weights and indicator C
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
 
    F = 0
    for k in range (C.shape[0]):  # C.shape is supposed to be the number of classes
        F += np.matmul(np.transpose(C[k, :]) , np.log(calculateDiag(X,W,C,k)))
    
    F = (-1/(X.shape[1])) * F
    return F
    

def calculateDiag(X,W,C,k):

    # Calculate the diagonal multapliction from page 15 in the class Notes,
    #   according to X with W weights, indicator C and running index k.
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
    sumOfExp = 0
    for j in range(C.shape[0]):
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

    
def calculateDiagGrad(X,W,C, k):
    sumOfExp = 0
    print(f'X Shape: {X.shape}')
    print(f'W Shape: {W.shape}')
    for j in range(C.shape[0]):
        print(f'Wj Shape: {W[:,j].reshape(-1, 1).shape}')
        print(f'Wj : {W[:,j]}')
        temp = np.exp(np.matmul(np.transpose(X),W[:,j].reshape(-1, 1)))
        print(f'temp shape: {temp.shape}')
        sumOfExp = sumOfExp + temp
        print(f'sumofexp Shape: {sumOfExp.shape}')
        
    print(f'C Shape: {C.shape}')
    print(f"mashu shape: {np.divide(np.exp(np.matmul(np.transpose(X),W)), sumOfExp).shape}")
    J = np.divide(np.exp(np.matmul(np.transpose(X),W[k,:])), sumOfExp) - C
    return J

def gradTest(X, W, C , eps, d):
    # Calculate gradient test
    #   according to X with W weights, indicator C, eps number, d random generated normalized matrix.
    # Assumptions:
        # X is : (t,m) 
        # C is : (l,m)
        # W is : (t,l)
        # d is : (t,m)
    return abs(softmaxLoss(X + eps * d, W, C) - softmaxLoss(X, W, C) - eps * np.transpose(d)*calculateSoftMaxGradW(X, W, C))

def genRandArr(rows, cols):
    return np.random.rand(rows,cols)

def genRandNormArr(rows, cols):
    d = genRandArr(rows, cols)
    return d/np.linalg.norm(d)


#def SGD(W,learning_rate=0.01 , beta ):
#    W = W - learning_rate * #grad W
    


