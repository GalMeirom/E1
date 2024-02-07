import numpy as np
import utils as ut
import funcs

class nn:
    def __init__(self, X, dims, C, LR, batchS):
        layers = []
        self.depth = len(dims) + 1
        self.LR = LR
        self.batchS = batchS
        dims.insert(0,X.shape[0])
        for i in range(1, self.depth):
            layers.append(funcs.layerFunc([X, ut.genRandNormArr(dims[i], dims[i-1]), ut.genRandNormArr(dims[i], batchS)], funcs.Tanh()))
        layers.append(funcs.SoftMaxLoss([X, ut.genRandNormArr(dims[-1], C.shape[0]), C]))
        self.layers = layers
        
            
    def forward(self):
        output = self.layers[0].params[0]
        numI = output.shape[1]
        for i in range(len(self.layers)):
            currLayer = self.layers[i]
            currLayer.set(0, output)
            output = currLayer.forward()
        return output
    
    def forwardPred(self):
        output = self.layers[0].params[0]
        numI = output.shape[1]
        for i in range(len(self.layers) - 1):
            currLayer = self.layers[i]
            currLayer.set(0, output)
            output = currLayer.forward()
        return ut.predict(output, self.layers[-1].params[1])

    
    def setLayers(self, layers):
        self.depth = len(layers)
        self.layers = layers

    def derivByThetaK(self, k):
        ks = []
        if k + 1 == len(self.layers):
            ks.append(self.layers[-1].deriv(1))
            return ks
        else: 
            output = self.layers[-1].deriv(0)
            for i in range(len(self.layers) - 2, k, -1):
                output = self.layers[i].deriv(0, output)
            ks.append(self.layers[k].deriv(1, output))
            ks.append(self.layers[k].deriv(2, output))
            return ks
    
    def grad(self, m):
        grad = []
        grads = []
        gradVec = []
        for i in range(len(self.layers) - 1, -1, -1):
            grads.append(self.derivByThetaK(i))
        grad.append(np.pad(grads[0][0], ((0, 0), (0, m - grads[0][0].shape[0])), mode='constant', constant_values=0))
        temp = np.vstack(grads[0][0])
        for i in range(len(grads) - 1):
            temp = np.vstack(gradTheta[0], gradTheta[1])
        output = np.vstack(gradVec) 
        return output
    
    # def grad(self):
    #     grads = []
    #     gradVec = []
    #     for i in range(len(self.layers) - 1, -1, -1):
    #         grads.append(self.derivByThetaK(i))
    #     for gradTheta in grads:
    #         for grad in gradTheta:
    #             gradVec.append(np.matrix.flatten(grad).reshape(-1, 1))
    #     output = np.vstack(gradVec) 
    #     return output
    
    def backProp(self):
        grads = []
        for i in range(len(self.layers) - 1, -1, -1):
            grads.append(self.derivByThetaK(i))
        self.layers[-1].set(1, self.layers[-1].params[1] - self.LR * grads[0][0])
        for j in range(len(self.layers) - 2, -1, -1):
            self.layers[j].set(1, self.layers[j].params[1] - self.LR * grads[len(grads) - j - 1][0])
            self.layers[j].set(2, self.layers[j].params[2] - self.LR * grads[len(grads) - j - 1][1])
        

class Residnn:
    def __init__(self, X, dims, C, LR):
        layers = []
        self.depth = len(dims) + 1
        self.LR = LR
        dims.insert(0,X.shape[0])
        for i in range(1, self.depth):
            layers.append(funcs.layerFunc([X, ut.genRandNormArr(dims[i], dims[i-1]), ut.genRandNormArr(dims[i], X.shape[1])], funcs.Tanh()))
        layers.append(funcs.SoftMaxLoss([X, ut.genRandNormArr(dims[-1], C.shape[0]), C]))
        self.layers = layers
        
            
    def forward(self):
        output = self.layers[0].params[0]
        numI = output.shape[1]
        for i in range(len(self.layers)):
            currLayer = self.layers[i]
            currLayer.set(0, output)
            output = currLayer.forward()
        return output
    
    def setLayers(self, layers):
        self.depth = len(layers)
        self.layers = layers

    def derivByThetaK(self, k):
        ks = []
        if k + 1 == len(self.layers):
            ks.append(self.layers[-1].deriv(1))
            return ks
        else: 
            output = self.layers[-1].deriv(0)
            for i in range(len(self.layers) - 2, k, -1):
                output = self.layers[i].deriv(0, output)
            ks.append(self.layers[k].deriv(1, output))
            ks.append(self.layers[k].deriv(2, output))
            return ks
    
    def grad(self):
        grads = []
        gradVec = []
        for i in range(len(self.layers) - 1, -1, -1):
            grads.append(self.derivByThetaK(i))
        for gradTheta in grads:
            for grad in gradTheta:
                gradVec.append(np.matrix.flatten(grad))
        output = np.vstack(grads) 
        return output
    
    def backProp(self):
        grads = []
        for i in range(len(self.layers) - 1, -1, -1):
            grads.append(self.derivByThetaK(i))
        for j in range(len(self.layers) - 1, -1, -1):
            self.layers[j].set(1, self.layers[j] - self.LR * grads[len(grads) - j - 1])
    