import torch
import pytorch_lightning as pl
from DataModule import DataModule
from LightningModel import LightningModel
from pytorch_lightning.loggers import WandbLogger

def trainer(
    batchSizeArg,
    learningRateArg,
    numEpochsArg,
    numWorkerArg,
    netModelArg,
    debugModeArg,
    pathOfCheckpointArg,
    runOnHpcArg,
    infoStringArg,
    targetFuncArg,
    classFuncArg,
    fiveFoldArg,
    testArg,
    lossFuncArg = torch.nn.functional.mse_loss,
    ):
    
    pl.seed_everything(fiveFoldArg,workers=True)

    if debugModeArg:
        wandbLogger = False
        print("+++++DEBUNG MODE IS USED / NO LOGGING+++++")
    else:
        runName = str(infoStringArg) + ";FNC=" + str(targetFuncArg) + ";FLD=" + str(fiveFoldArg) + ";LR=" + str(learningRateArg) + ";EP=" + str(numEpochsArg) + ";BZ=" + str(batchSizeArg)
        wandbLogger = WandbLogger(name=runName,project="regressionEvaluation")

    dataModule = DataModule(numWorkerArg,batchSizeArg,targetFuncArg,fiveFoldArg,runOnHpcArg)

    optimClass = torch.optim.AdamW

    netModel = LightningModel(
        netModelArg,
        learningRateArg,
        numEpochsArg,
        lossFuncArg,
        optimClass,
        targetFuncArg,
        classFuncArg, # list of class fuctions
        )
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision=16,
        max_epochs=numEpochsArg,
        val_check_interval=1.0,
        logger=wandbLogger,
    )

    trainer.fit(
        model=netModel,
        datamodule=dataModule,
        #ckpt_path=pathOfCheckpointArg, # add this to also use checkpoints
        )
    if testArg:
        trainer.test(model=netModel,datamodule=dataModule)
    
    print("FINISHED")