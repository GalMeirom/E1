import numpy as np
import utils as ut
import funcs

class nn:
    def __init__(self, X, depth, C):
        
        layers = []
        self.depth = depth
        for i in range(depth-1):
            layers.append(funcs.layerFunc([X, ut.genRandNormArr(X.shape[0], X.shape[0]), ut.genRandNormArr(X.shape[0], X.shape[1])], funcs.Tanh()))
        layers.append(funcs.SoftMaxLoss([X, ut.genRandNormArr(X.shape[0], X.shape[0]), C]))
        self.layers = layers
    
    def forward(self):
        output = self.layers[0].params[0]
        for i in range(len(self.layers)):
            currLayer = self.layers[i]
            currLayer.set(0, output)
            output = currLayer.forward()
        return output
    
    def setLayers(self, layers):
        self.depth = len(layers)
        self.layers = layers

    def derivByThetaK(self, k):
        if k + 1 == len(self.layers):
            return self.layers[-1].deriv(1)
        else:
            
            output = self.layers[-1].deriv(0)
            for i in range(len(self.layers) - 2, k, -1):
                output = self.layers[i].deriv(0, output)
            output = self.layers[k].deriv(1, output)
            return output
    
    def grad(self):
        grads = []
        for i in range(len(self.layers) - 1, -1, -1):
            grads.append(self.derivByThetaK(i))
        output = np.vstack(grads) 
        return output

    