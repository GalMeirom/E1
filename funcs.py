import numpy as np
import utils as ut
class func:
    
    def __init__(self, params):
        self.params = params


class SoftMaxLoss(func):

    def __init__(self, params):
        super().__init__(params)
    
    def set(self, k, M):
        self.params[k] = M

    def deriv(self, k):
        if k == 0:
            return self.derivX()
        if k == 1:
            return self.derivW()

    def forward(self):
        F = 0
        for k in range (self.params[2].shape[0]):  # C.shape is supposed to be the number of classes
            F += np.matmul(np.transpose(self.params[2][k, :]) , np.log(ut.calculateDiag(self.params[0],self.params[1],k)))
    
        F = (-1/(self.params[0].shape[1])) * F
        return F
    
    def derivW(self):
        grad = 1/self.params[0].shape[1] *np.matmul(self.params[0],ut.calculateDiagGrad(self.params[0],self.params[1],self.params[2]))
        return grad
    
    def derivX(self):
        grad = 1/self.params[0].shape[1] *np.matmul(self.params[1],np.transpose(ut.calculateDiagGrad(self.params[0],self.params[1],self.params[2])))
        return grad



class LinearLeastSquares(func):

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
    
    def __init__(self, params, act):
        super().__init__(params)
        self.act = act
    
    def set(self, k, M):
        self.params[k] = M

    def forward(self):
        return self.act.forward(np.matmul(self.params[1], self.params[0]) + self.params[2])

    def deriv(self, k, v):
        if k == 0:
            return self.derivXTv(v)
        if k == 1:
            return self.derivWTv(v)
        if k == 2:
            return self.derivBTv(v)

    def derivXTv(self, v):
        return np.matmul(np.transpose(self.params[1]), np.multiply(self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2]), v))
    
    def derivWTv(self, v):
        return np.matmul(np.multiply(self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2]), v), np.transpose(self.params[0]))
    
    def derivBTv(self, v):
        return np.multiply(self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2]), v)    


    # def deriv(self, k, X):
    #     if k == 0:
    #         return self.derivX(X)
    #     if k == 1:
    #         return self.derivW(X)
    #     if k == 2:
    #         return self.derivB(X)
        
    # def derivX(self, X, v):
    #     return np.matmul(np.diag(self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2])), self.params[1])
    
    # def derivW(self, X):
    #     l = self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2]).shape[0]
    #     w = self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2]).shape[1]
    #     print(self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2]).shape)
    #     return np.matmul(np.fill_diagonal(np.identity(l*w),self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2])) , np.kron(np.transpose(self.params[0]), np.identity(self.params[0].shape[0])))
    
    # def derivB(self, X):
    #     return np.diag(self.act.deriv(np.matmul(self.params[1], self.params[0]) + self.params[2]))