import numpy as np
import scipy.io
import utils as ut

def Q1C1():
    mat = scipy.io.loadmat('data/SwissRollData.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    softmax = ut.softmaxLoss(Xt,W,Ct)
    t = ut.gradTest(Xt,W,Ct, 0.001, ut.genRandNormArr(Xt.shape[0],Xt.shape[1]))

def Q1C2():
    mat = scipy.io.loadmat('data/SwissRollData.mat')
    Ct = mat['Ct']
    Xt = mat['Yt']
    W = ut.genRandArr(Xt.shape[0],Ct.shape[0])
    softmax = ut.softmaxLoss(Xt,W,Ct)
    t = ut.gradTest(Xt,W,Ct, 0.001, ut.genRandNormArr(Xt.shape[0],Xt.shape[1]))
