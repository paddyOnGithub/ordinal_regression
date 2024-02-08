import torch
import math
import CONFIG

def gaussVal(x,std=1,mean=0) -> float:
    factor = 1/(std*math.sqrt(2*math.pi))
    inExp = -(0.5*((x-mean)/std)**2)
    result = factor*math.exp(inExp)
    return result

def getGaussList(zeroIdx:int,listLen=5) -> list:
    idxList = range(-zeroIdx,listLen-zeroIdx)
    gaussList = [gaussVal(x) for x in idxList]
    return gaussList

gaussEncTens = torch.tensor(
    [getGaussList(0),
     getGaussList(1),
     getGaussList(2),
     getGaussList(3),
     getGaussList(4),]
).type(torch.float)

oneHotEncTens = torch.tensor(
    [[1,0,0,0,0],
     [0,1,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]]
).type(torch.float)

contEncTens = torch.tensor(
    [[0.00],
     [0.25],
     [0.50],
     [0.75],
     [1.00]]
).type(torch.float)

progBarEncTens = torch.tensor(
    [[0,0,0,0],
     [1,0,0,0],
     [1,1,0,0],
     [1,1,1,0],
     [1,1,1,1]]
).type(torch.float)

softProgEncTens = torch.tensor(
    [[0.5,0,0,0,0],
     [1,0.5,0,0,0],
     [1,1,0.5,0,0],
     [1,1,1,0.5,0],
     [1,1,1,1,0.5]]
).type(torch.float)

binNumEncTens = torch.tensor(
    [[0,0,1],
     [0,1,0],
     [0,1,1],
     [1,0,0],
     [1,0,1]]
).type(torch.float)

TARGET_FUNC_TO_TENS = {
    CONFIG.ARG_GAUSS:        gaussEncTens,
    CONFIG.ARG_ONEHOT:       oneHotEncTens,
    CONFIG.ARG_CONTINUOUS:   contEncTens,
    CONFIG.ARG_PROGBAR:      progBarEncTens,
    CONFIG.ARG_SOFTPROG:     softProgEncTens,
    CONFIG.ARG_BINNUM:       binNumEncTens,
}

TARGET_FUNC_TO_LEN = {
    CONFIG.ARG_GAUSS:        5,
    CONFIG.ARG_ONEHOT:       5,
    CONFIG.ARG_CONTINUOUS:   1,
    CONFIG.ARG_PROGBAR:      4,
    CONFIG.ARG_SOFTPROG:     5,
    CONFIG.ARG_BINNUM:       3,
}

def createTargetLabel(numLabelList,targetFunc):
    numLabelList = torch.tensor(numLabelList)
    encTens = TARGET_FUNC_TO_TENS[targetFunc]
    ret = torch.index_select(encTens,0,numLabelList)
    return ret