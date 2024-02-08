import targetMap
import torch
import CONFIG

NUMBER_OF_DIFFERENT_CLASSES = 5



def argmaxMap(inpTens,_): # targetFunc is not needed here
    '''Wrapper function for argmax'''
    return torch.argmax(inpTens,dim=1)

def sumMap(inpTens,_): # targetFunc is not needed here
    '''Sums up the output vector'''
    retTens = torch.round(torch.sum(inpTens,dim=1))
    retTens = torch.maximum(retTens,torch.full(retTens.size(),0,dtype=retTens.dtype,device=retTens.device))
    retTens = torch.minimum(retTens,torch.full(retTens.size(),CONFIG.NUMBER_OF_LEVELS_PER_CLASS -1,dtype=retTens.dtype,device=retTens.device))          
    return retTens

def l1Dist(inpTens,targetFunc):
    '''Computes the class of the target vector closest to the original vector, as measured by the L1 distance.'''
    targetTens = targetMap.TARGET_FUNC_TO_TENS[targetFunc].to(dtype=inpTens.dtype,device=inpTens.device)
    distTens = torch.zeros(inpTens.size()[0],CONFIG.NUMBER_OF_LEVELS_PER_CLASS,dtype=inpTens.dtype,device=inpTens.device)
    for i, inpVect in enumerate(inpTens):
        distTens[i] = torch.sum(torch.abs(targetTens - inpVect[None,:]),dim=1) #calculate the l1 distance to each vector in the target tensor
    targetClass = torch.argmin(distTens,dim=1)
    return targetClass

def dotProd(inpTens,targetFunc):
    '''Computes the class of the target vector closest to the original vector, as measured by the normalized dot product'''
    targetTens = targetMap.TARGET_FUNC_TO_TENS[targetFunc].to(dtype=inpTens.dtype,device=inpTens.device)
    normInpVect = torch.nn.functional.normalize(inpTens,dim=1)
    normTargetTens = torch.nn.functional.normalize(targetTens,dim=1)
    similarity = torch.matmul(normInpVect,normTargetTens.T)
    targetClass = torch.argmax(similarity,dim=1)
    return targetClass

#map string from argument to the right function above
classFuncArgToFunc = {
    CONFIG.ARG_CLS_ARGMAX: argmaxMap,
    CONFIG.ARG_CLS_SUM: sumMap,
    CONFIG.ARG_CLS_L1DIST: l1Dist,
    CONFIG.ARG_CLS_DOTPROD: dotProd,   
}



def mapFeatureToClass(inpFeatureTens,classFunc,targetFunc):
    '''Apply a given classFunc to the outputs'''
    mapFunc = classFuncArgToFunc[classFunc]
    return mapFunc(inpFeatureTens,targetFunc).type(torch.int)

    
