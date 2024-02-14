import numpy as np
import utils as ut
import scipy
class func:
    
    def __init__(self, params):
        self.params = params

class SoftMaxLoss(func):
    # params = [X, W1, C]
    def __init__(self, params):
        b = ut.genRandArr(params[1].shape[1], 1)
        params[1] = np.vstack([params[1], np.transpose(b)])
        params[0] = np.vstack([params[0], np.ones((1,params[0].shape[1]))])
        super().__init__(params)
    
    def set(self, k, M):
        # if k == 0:
        #     self.params[0] = np.vstack([M, np.ones((1, M.shape[1]))])
        # else:
        self.params[k] = M

    def deriv(self, k):
        if k == 0:
            return self.derivX()
        if k == 1:
            return self.derivW()
    
    def derivT(self, k):
        if k == 0:
            return self.derivXT()
        if k == 1:
            return self.derivWT()
    
    def forward(self):
        # scalar
        F = 0
        for k in range (self.params[2].shape[0]):  # C.shape is supposed to be the number of classes
            F += np.matmul(np.transpose(self.params[2][k, :]) , np.log(ut.calculateDiag(self.params[0],self.params[1],k)))
    
        F = (-1/(self.params[0].shape[1])) * F
        return F
    
    #def derivW(self):
    #    # [X.rows x C.rows]
    #    grad = 1/self.params[0].shape[1] *np.matmul(self.params[0],ut.calculateDiagGrad(self.params[0],self.params[1],self.params[2]))
    #    return grad
    #   

    # paper logics
    # l is number of classes
    # m is number of samples in batch
    # n is number of features
    # X in n on m
    # W in n on l
    # C in m on l


    def derivW(self):
        # [X.rows x C.rows]
        sum = np.exp(np.matmul(np.transpose(self.params[0]), self.params[1][:, 0].reshape(-1, 1)))
        for j in range(1, self.params[1].shape[1]):
            sum = sum + np.exp(np.matmul(np.transpose(self.params[0]), self.params[1][:, j].reshape(-1, 1)))
        div = np.exp(np.matmul(np.transpose(self.params[0]), self.params[1]))/sum
        grad = 1/self.params[0].shape[1] * np.matmul(self.params[0],div - np.transpose(self.params[2]))
        return np.matrix.flatten(grad).reshape(-1,1)
    
    def derivWT(self):
        # [X.rows x C.rows]
        sum = np.exp(np.matmul(np.transpose(self.params[0]), self.params[1][:, 0].reshape(-1, 1)))
        for j in range(1, self.params[1].shape[1]):
            sum = sum + np.exp(np.matmul(np.transpose(self.params[0]), self.params[1][:, j].reshape(-1, 1)))
        div = np.exp(np.matmul(np.transpose(self.params[0]), self.params[1]))/sum
        grad = 1/self.params[0].shape[1] * np.matmul(self.params[0],div - np.transpose(self.params[2]))
        return grad

    #def derivX(self):
    #    # [W.rows x C.cols]
    #    grad = 1/self.params[0].shape[1] *np.matmul(self.params[1],np.transpose(ut.calculateDiagGrad(self.params[0],self.params[1],self.params[2])))
    #    return grad

    def derivX(self):
        sum = np.exp(np.matmul(np.transpose(self.params[0]), self.params[1][:, 0].reshape(-1, 1)))
        for j in range(1, self.params[1].shape[1]):
            sum = sum + np.exp(np.matmul(np.transpose(self.params[0]), self.params[1][:, j].reshape(-1, 1)))
        div = np.divide(np.exp(np.matmul(np.transpose(self.params[0]), self.params[1])), sum)
        grad = 1/self.params[0].shape[1] *np.matmul(self.params[1],np.transpose(div - np.transpose(self.params[2])))
        return np.matrix.flatten(grad).reshape(-1,1)
    
    def derivXT(self):
        sum = np.exp(np.matmul(np.transpose(self.params[0]), self.params[1][:, 0].reshape(-1, 1)))
        for j in range(1, self.params[1].shape[1]):
            sum = sum + np.exp(np.matmul(np.transpose(self.params[0]), self.params[1][:, j].reshape(-1, 1)))
        div = np.divide(np.exp(np.matmul(np.transpose(self.params[0]), self.params[1])), sum)
        grad = 1/self.params[0].shape[1] *np.matmul(self.params[1],np.transpose(div - np.transpose(self.params[2])))
        return grad


class LinearLeastSquares(func):
    # params = [A, X, b]
    def __init__(self, params):
        super().__init__(params)

    def set(self, k, M):
        self.params[k] = M
    
    def deriv(self, k):
        if k == 1:
            return self.derivX()

    def forward(self):
        return np.linalg.norm(np.matmul(self.params[0], self.params[1]) - self.params[2], 2)*0.5
    
    def derivX(self):
        return np.matmul(np.matmul(np.transpose(self.params[0]), self.params[0]), self.params[1]) - np.matmul(np.transpose(self.params[0]), self.params[2])


class Tanh():
    def __init__(self):
        None
    
    def forward(self, X):
        return np.tanh(X)
    
    def deriv(self, X):
        return np.ones(X.shape) - np.square(np.tanh(X))


class Relu():
    def __init__(self):
        None
    
    def forward(self, X):
        return np.maximum(0, X)

    def reluDerivativeSingleElement(self, xi):
        if xi > 0:
            return 1
        elif xi <= 0:
            return 0
    
    def deriv(self, X):
        return np.array([self.reluDerivativeSingleElement(xi) for xi in X])
    

class layerFunc(func):
    # params = [X, W1, b]
    def __init__(self, params, act):
        super().__init__(params)
        self.act = act

    def set(self, k, M):
        self.params[k] = M

    def forward(self):
        mul = np.matmul(self.params[1], self.params[0])
        return self.act.forward(mul + self.params[2])

    def derivT(self, k, v):
        if k == 0:
            return self.derivXTv(v)
        if k == 1:
            return self.derivWTv(v)
        if k == 2:
            return self.derivBTv(v)

    def derivXTv(self, v):
        # [W.cols x X.cols]
        der = self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2])
        output = np.matmul(np.transpose(self.params[1]), np.multiply(der, v))
        return output
    
    def derivWTv(self, v):
        # [W.rows x X.rows]
        der = self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2])
        output = np.matmul(np.multiply(der, v), np.transpose(self.params[0]))
        return output
    
    def derivBTv(self, v):
        # [W.rows x X.cols]
        output = np.multiply(self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2]), v) 
        return output
    
    
    def derivX(self):
        der = self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2])
        dig = np.diag(np.matrix.flatten(der))
        output = np.matmul(dig, self.params[1])
        return output
    
    def derivW(self):
        der = self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2])
        dig = np.diag(np.matrix.flatten(der))
        kr = np.kron(np.transpose(self.params[0]), np.identity(self.params[2].shape[0]))
        output = np.matmul(dig, kr)
        return output
    
    def derivB(self):
        der = self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2])
        dig = np.diag(np.matrix.flatten(der))
        return dig
    
    def deriv(self, k):
        if k == 0:
            return self.derivX()
        if k == 1:
            return self.derivW()
        if k == 2:
            return self.derivB()
    


class ResidlayerFunc(func):
    
    def __init__(self, params, act):
        # params = [X, W1, b, W2]
        super().__init__(params)
        self.act = act

    def set(self, k, M):
        self.params[k] = M

    def forward(self):
        return self.params[0] + np.matmul(self.params[3] ,self.act.forward(np.matmul(self.params[1], self.params[0]) + self.params[2]))

    def derivT(self, k, v):
        if k == 0:
            return self.derivXTv(v)
        if k == 1:
            return self.derivW1Tv(v)
        if k == 2:
            return self.derivBTv(v)
        if k == 3:
            return self.derivW2Tv(v)

    def derivXTv(self, v):
        # [X.cols x X.cols]
        der = self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2])
        three = np.matmul(np.transpose(self.params[3]), v)
        output = np.matmul(np.identity(self.params[1].shape[1]), v) + np.matmul(np.transpose(self.params[1]), np.multiply(der, three))
        return output
    
    def derivW1Tv(self, v):
        # [W1.rows x X.rows]
        output = np.matmul(np.multiply(self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2]), np.matmul(np.transpose(self.params[3]), v)), np.transpose(self.params[0]))
        return output
    
    def derivBTv(self, v):
        # [W1.rows x X.cols]
        output = np.multiply(self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2]), np.matmul(np.transpose(self.params[3]), v))
        return output
    
    def derivW2Tv(self, v):
        # [v.rows x W1.rows]
        output = np.matmul(v, np.transpose(self.act.forward(np.matmul(self.params[1], self.params[0]) + self.params[2])))    
        return output

    def deriv(self, k):
        if k == 0:
            return self.derivX()
        if k == 1:
            return self.derivW1()
        if k == 2:
            return self.derivB()
        if k == 3:
            return self.derivW2()
    
    def derivX(self):
        der = self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2])
        dig = np.diag(np.matrix.flatten(der))
        first = np.matmul(dig, self.params[1])
        second = np.matmul(self.params[3], first)
        output = np.identity(self.params[0].shape[0]) + second
        return output
    
    def derivW1(self):
        der = self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2])
        dig = np.diag(np.matrix.flatten(der))
        kr = np.kron(np.transpose(self.params[0]), np.identity(self.params[2].shape[0]))
        output = np.matmul(dig, kr)
        output = np.matmul(self.params[3], output)
        return output
    
    def derivB(self):
        der = self.act.deriv(np.matmul(self.params[1], self.params[0])+ self.params[2])
        dig = np.diag(np.matrix.flatten(der))
        return np.matmul(self.params[3] ,dig)
    
    def derivW2(self):
        ford = self.act.forward(np.matmul(self.params[1], self.params[0])+ self.params[2])
        output = np.kron(np.transpose(ford), np.identity(self.params[0].shape[0]))
        return output