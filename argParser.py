if __name__ == '__main__': #for multiprocessing
    import getopt, sys
    from trainer import trainer
    import CONFIG
    import multiprocessing
    import torch

    argumentList = sys.argv[1:]
    shortOptions = "a:"
    longOptions = [
    "batchSize=",
    "learningRate=",
    "numEpochs=",
    "numWorker=",
    "netModel=",
    "debugMode=",
    "pathOfCheckpoint=",
    "runOnHpc=",
    "infoString=",
    "targetFunc=",
    "classFunc=",
    "fiveFold=",
    "test=",
    "loss=",
    ]

    try:
        argumentCollection, _ = getopt.getopt(argumentList,shortOptions,longOptions)

        #default values for some options
        pathOfCheckpointArg = None
        #debugModeArg=False
        runOnHpcArg = False
        infoStringArg = "noInfo"
        #useAugmentArg = True #TODO delete, not used anymore
        numWorkerArg = multiprocessing.cpu_count()
        testArg = False

        lossFuncArg = torch.nn.functional.mse_loss

        lossFuncDict = {
            "mse":  torch.nn.functional.mse_loss,
            "cse":  torch.nn.functional.cross_entropy,
            "bce": torch.nn.functional.binary_cross_entropy,
        }

        for currArg, currVal in argumentCollection:
            print("SET ARGUMENT",currArg,currVal) #TODO comment out???

            if currArg in ("--batchSize"):
                batchSizeArg = int(currVal)
            
            if currArg in ("--learningRate"):
                learningRateArg = float(currVal)

            if currArg in ("--numEpochs"):
                numEpochsArg = int(currVal)
            
            if currArg in ("--numWorker"):
                numWorkerArg = int(currVal)
            
            if currArg in ("--netModel"):
                netModelArg = str(currVal) 
            
            if currArg in ("--debugMode"):
                debugModeArg = currVal == "True"
            
            if currArg in ("--pathOfCheckpoint"):
                pathOfCheckpointArg = str(currVal) 
            
            if currArg in ("--runOnHpc"):
                runOnHpcArg = currVal == "True"
            
            if currArg in ("--infoString"):
                infoStringArg = str(currVal) 
            
            if currArg in ("--targetFunc"):
                targetFuncArg = str(currVal)
                assert targetFuncArg in CONFIG.POSSIBLE_TARGET_FUNC, str(el) + "is not a valid Argument for " + currArg + ". Possible arguments:" + str(CONFIG.POSSIBLE_TARGET_FUNC)
            
            if currArg in ("--classFunc"): 
                classFuncArg = currVal.strip().split(sep=";")
                for el in classFuncArg:
                    assert el in CONFIG.POSSIBLE_CLASS_FUNC, str(el) + "is not a valis Argument for " + currArg + ". Possible arguments:" + str(CONFIG.POSSIBLE_CLASS_FUNC) 
            
            if currArg in ("--fiveFold"):
                fiveFoldArg = int(currVal)
            
            if currArg in ("--test"):
                testArg = currVal == "True"
            
            if currArg in ("--loss"):
                lossFuncArg = lossFuncDict[currVal]


        
        #execute trainer
        trainer(
            batchSizeArg = batchSizeArg,
            learningRateArg = learningRateArg,
            numEpochsArg = numEpochsArg,
            numWorkerArg = numWorkerArg,
            netModelArg = netModelArg,
            debugModeArg = debugModeArg,
            pathOfCheckpointArg = pathOfCheckpointArg,
            runOnHpcArg = runOnHpcArg,
            infoStringArg = infoStringArg,
            targetFuncArg = targetFuncArg,
            classFuncArg = classFuncArg,
            fiveFoldArg = fiveFoldArg,
            testArg = testArg,
            lossFuncArg=lossFuncArg,
        )



    except getopt.error as err:
        print(str(err))

    