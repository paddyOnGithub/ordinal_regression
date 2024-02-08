from torch.utils.data import DataLoader
import pytorch_lightning as pl
import CONFIG
from CXRDataset import CXRDataset

class DataModule(pl.LightningDataModule):

    def __init__(
            self,
            numWorker,
            batchSize,
            targetFunc,
            fiveFold,
            runOnHpc,
        ):
        super().__init__()
        self.numWorker = numWorker
        self.batchSize = batchSize
        self.targetFunc = targetFunc
        self.fiveFold = fiveFold

        if runOnHpc:
            self.metaFilePath = CONFIG.HPC_META_FILE_PATH
            self.imgFolderPath = CONFIG.HPC_IMG_FOLDER_PATH
        else:
            self.metaFilePath = CONFIG.LOCAL_META_FILE_PATH
            self.imgFolderPath = CONFIG.LOCAL_IMG_FOLDER_PATH
    
    def setup(self,stage):
        self.trainDataset = CXRDataset(self.metaFilePath,self.imgFolderPath,"train",self.targetFunc,self.fiveFold)

        self.validDataset = CXRDataset(self.metaFilePath,self.imgFolderPath,"valid",self.targetFunc,self.fiveFold)

        self.testDataset = CXRDataset(self.metaFilePath,self.imgFolderPath,"test",self.targetFunc,self.fiveFold)
    
    def train_dataloader(self):
        trainLoader = DataLoader(self.trainDataset,batch_size=self.batchSize,num_workers=self.numWorker,shuffle=True)
        return trainLoader
    
    def val_dataloader(self):
        validLoader = DataLoader(self.validDataset,batch_size=self.batchSize,num_workers=self.numWorker,shuffle=False)
        return validLoader
    
    def test_dataloader(self):
        testLoader = DataLoader(self.testDataset,batch_size=self.batchSize,num_workers=self.numWorker,shuffle=False)
        return testLoader
        

