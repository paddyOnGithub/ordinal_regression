from torch.utils.data import Dataset
import csv
from PIL import Image
import torchvision
import torch
import CONFIG
from targetMap import createTargetLabel
import random

class CXRDataset(Dataset):
    '''The Python dataset file for accessing and preprocessing the chest radiographs'''
    def __init__(
        self,
        metaFile,
        imageFolder,
        split,
        targetFunc,
        fiveFold,
    ):
        self.metaFile = metaFile
        self.imageFolder = imageFolder
        self.split = split
        self.targetFunc = targetFunc
        self.fiveFold = fiveFold

        with open(metaFile,newline="") as f: #read the meta file 
            reader = csv.reader(f)
            metaFileList = list(reader)
        
        self.imgList = metaFileList[1:]

        if split == "test":
            self.imgList = [x for x in self.imgList if x[1] == "test"]

        if split == "train":
            self.imgList = [x for x in self.imgList if x[1] != "test" and x[19] != str(fiveFold)]
        
        if split == "valid":
            self.imgList = [x for x in self.imgList if x[1] != "test" and x[19] == str(fiveFold)]


        print(split,"; len:",len(self.imgList))

        self.augmentTransform = torch.nn.Sequential( # inspired by https://arxiv.org/abs/1812.01187
            torchvision.transforms.RandomRotation((-15,15),interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.RandomResizedCrop(size=(CONFIG.IMG_SIZE,CONFIG.IMG_SIZE),scale=(0.08,1),ratio=(3/4,4/3),antialias=True),
            torchvision.transforms.ColorJitter(brightness=0.4)
        )

        self.stdTransform = torchvision.transforms.Resize(size=(CONFIG.IMG_SIZE,CONFIG.IMG_SIZE),antialias=True)

        self.imgToTensor = torchvision.transforms.ToTensor()

        self.normalizeImg = torchvision.transforms.Normalize(mean=CONFIG.IMG_NORM_MEAN,std=CONFIG.IMG_NORM_STD,inplace=True)

    def __len__(self):
        return len(self.imgList)
    
    def __getitem__(self, idx):
        sample = self.imgList[idx]

        #prepare Image
        img = self.imgFromSample(sample)
        doAugment = True if self.split == "train" else False
        img = self.prepareImg(img,doAugment)
        
        #prepare Label
        rawLabel = sample[9:16]
        numLabel = [CONFIG.LABEL_CORRECTION_DICT[x] for x in rawLabel] #change mapping to ordinal interval [0,...,4]
        targetLabel = createTargetLabel(numLabel,self.targetFunc)
        targetLabel = torch.flatten(targetLabel)

        return img, targetLabel

    def imgFromSample(self,sample):
        '''Given a sample, open the image'''
        imgPath = self.imageFolder + str(sample[2]) + "/" + str(sample[0]) + ".jpg"
        img = Image.open(imgPath)
        return img

    def prepareImg(self,img,augment):
        '''Preprocess image and apply augmentaions'''
        img = self.imgToTensor(img)
        assert img.size()[1] == 512
        assert img.size()[2] == 512
        if augment:
            img = self.augmentTransform(img)
            img = img + torch.normal(mean=0,std=(0.1/255),size=img.size()) # (0.1/255) because our image is scaled from 0 to 1, in the paper (https://arxiv.org/abs/1812.01187) its from 0 to 255
        else:
            img = self.stdTransform(img)
        img = self.normalizeImg(img)
        return img






        

    