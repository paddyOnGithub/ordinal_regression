import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from targetMap import TARGET_FUNC_TO_LEN
import torchvision
import torch.nn as nn
import CONFIG
import torch
import classMap
import torchmetrics



class LightningModel(pl.LightningModule):
    '''Main module for training and evaluation management'''
    def __init__(self,modelName,learningRate,numEpochs,lossFunc,optimizerClass,targetFunc,classFuncList):
        super().__init__()
        self.modelName = modelName
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.lossFunc = lossFunc
        self.optimizerClass = optimizerClass
        self.targetFunc = targetFunc
        self.targetLen = TARGET_FUNC_TO_LEN[self.targetFunc]
        self.classFuncList = classFuncList

        outFeatrueNum = CONFIG.FEATURE_NUM*self.targetLen
        if modelName == CONFIG.ARG_RESNET50_MODEL:
            self.model = createResNet50Model(outFeatrueNum)
        elif modelName == CONFIG.ARG_DEIT_MODEL:
            self.model = createDeiTmodel(outFeatrueNum)
        else:
            assert False
        
        self.validationStepPred = list()
        self.validationStepTarget = list()

        self.testStepPred = list()
        self.testStepTarget = list()
    
    def training_step(self, batch, batch_idx):

        img,target = batch

        pred = self.model(img)
        loss = self.lossFunc(pred,target)
        self.log("train_loss",loss)
        return loss
    
    def on_train_epoch_end(self):
        currLearningRate = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", currLearningRate)
    
    def configure_optimizers(self):
        
        optimizer = self.optimizerClass(self.parameters(),lr=self.learningRate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.numEpochs)

        return [optimizer,], [scheduler,]
    
    def validation_step(self,batch,batch_idx):

        img,target = batch
        pred = self.model(img)

        self.validationStepPred.append(pred)
        self.validationStepTarget.append(target)
        return pred
    
    def test_step(self,batch,batch_idx):
        img,target = batch
        pred = self.model(img)

        self.testStepPred.append(pred)
        self.testStepTarget.append(target)
        return pred
    
    def evalModel(self,mode):
        '''Evaluate the current model'''
        assert mode == "valid" or mode == "test"
        if mode == "valid":
            allPred = shapeBatchList(self.validationStepPred,self.targetLen)
            allTarget = shapeBatchList(self.validationStepTarget,self.targetLen)
            self.validationStepPred.clear()
            self.validationStepTarget.clear()
            namePrefix = "valid/"
        elif mode == "test":
            allPred = shapeBatchList(self.testStepPred,self.targetLen)
            allTarget = shapeBatchList(self.testStepTarget,self.targetLen)
            self.testStepPred.clear()
            self.testStepTarget.clear()
            namePrefix = "test/"
        else:
            assert False

        loss = self.lossFunc(allPred,allTarget)
        self.log(namePrefix + "loss",loss)

        for classFunc in self.classFuncList:
            unweightedAvgList = list()
            linearAvgList = list()
            quadraticAvgList = list()


            for featureIdx in range(CONFIG.FEATURE_NUM):
                featurePred = classMap.mapFeatureToClass(allPred[featureIdx],classFunc,self.targetFunc)
                featureTarget = classMap.mapFeatureToClass(allTarget[featureIdx],classFunc,self.targetFunc)
                featureName = CONFIG.FEATURE_IDX_TO_NAME[featureIdx]

                unweightedKappa = torchmetrics.functional.cohen_kappa(featurePred,featureTarget,task="multiclass",num_classes=CONFIG.FEATURE_NUM,weights="none")
                linearKappa = torchmetrics.functional.cohen_kappa(featurePred,featureTarget,task="multiclass",num_classes=CONFIG.FEATURE_NUM,weights="linear")
                quadraticKappa = torchmetrics.functional.cohen_kappa(featurePred,featureTarget,task="multiclass",num_classes=CONFIG.FEATURE_NUM,weights="quadratic")
                

                self.log(namePrefix + featureName + "/" + classFunc + "/unweighted_Kappa",unweightedKappa)
                self.log(namePrefix + featureName + "/" + classFunc +"/linear_Kappa",linearKappa)
                self.log(namePrefix + featureName + "/" + classFunc +"/quadratic_Kappa",quadraticKappa)

                unweightedAvgList.append(unweightedKappa)
                linearAvgList.append(linearKappa)
                quadraticAvgList.append(quadraticKappa)

            unweightedAvgKappa = torch.mean(torch.stack(unweightedAvgList))
            linearAvgKappa = torch.mean(torch.stack(linearAvgList))
            quadraticAvgKappa = torch.mean(torch.stack(quadraticAvgList))

            self.log(namePrefix + "avg/" + classFunc + "/unweighted_Kappa",unweightedAvgKappa)
            self.log(namePrefix + "avg/" + classFunc + "/linear_Kappa",linearAvgKappa)
            self.log(namePrefix + "avg/" + classFunc + "/quadratic_Kappa",quadraticAvgKappa)

        

    
    def on_validation_epoch_end(self):
        self.evalModel("valid")
    
    def on_test_epoch_end(self):
        self.evalModel("test")



def shapeBatchList(tensList,targetLen):
    oneTens = torch.cat(tensList)
    featureAwareTens = torch.reshape(oneTens,(oneTens.size()[0],CONFIG.FEATURE_NUM,targetLen))
    retTens = torch.transpose(featureAwareTens,0,1)
    return retTens

def createResNet50Model(numOutFeatures): 

    #create pretrained model
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

    #change input layer to one channel, keep weights
    avgWeight = torch.mean(model.conv1.weight,1,True)
    newConv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    newConv1.weight = nn.Parameter(avgWeight)
    model.conv1 = newConv1

    #create new output layer
    newOutLayer = nn.Linear(in_features=2048,out_features=numOutFeatures)
    nn.init.xavier_uniform_(newOutLayer.weight)
    model.fc = newOutLayer

    return model

def createDeiTmodel(numOutFeatures,noNorm=False):

    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

    avgWeight = torch.mean(model.patch_embed.proj.weight,1,True)
    newConvProj = nn.Conv2d(1,768,kernel_size=(16,16),stride=(16,16),padding=(3,3))
    newConvProj.weight = nn.Parameter(avgWeight)
    model.patch_embed.proj = newConvProj

    newOutLayer = nn.Linear(in_features=768,out_features=numOutFeatures,bias=True)
    nn.init.xavier_uniform_(newOutLayer.weight)
    model.head = newOutLayer
    if noNorm:
        model.norm = nn.Identity()
    print(model)
    return model



        

