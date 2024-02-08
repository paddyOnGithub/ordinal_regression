#This file is used to set some constants

IMG_SIZE = 224

IMG_NORM_MEAN = (0.449,) #mean of [0.485, 0.456, 0.406]
IMG_NORM_STD = (0.226,) ##mean of [0.229, 0.224, 0.225]

#correct the mapping from numbers in the dataset to interval [0,...,4]
LABEL_CORRECTION_DICT = {
    "1": 0,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 1,
}

FEATURE_IDX_TO_NAME = ["congestion","pleural_effusion_right","pleural_effusion_left","pneumonic_infiltrates_right","pneumonic_infiltrates_left","atelectasis_right","atelectasis_left"]

FEATURE_NUM = len(FEATURE_IDX_TO_NAME) #number of features (here number of diseases)

NUMBER_OF_LEVELS_PER_CLASS = 5



ARG_GAUSS = "gauss"
ARG_ONEHOT = "oneHot"
ARG_CONTINUOUS = "continuous"
ARG_PROGBAR = "progBar"
ARG_SOFTPROG = "softProg"
ARG_BINNUM = "binNum"

POSSIBLE_TARGET_FUNC = [
    ARG_GAUSS,
    ARG_ONEHOT,
    ARG_CONTINUOUS,
    ARG_PROGBAR,
    ARG_SOFTPROG,
    ARG_BINNUM,
]

ARG_CLS_ARGMAX = "argmax"
ARG_CLS_SUM = "sum"
ARG_CLS_L1DIST = "l1dist"
ARG_CLS_DOTPROD = "dotProd"

POSSIBLE_CLASS_FUNC = [
    ARG_CLS_ARGMAX,
    ARG_CLS_SUM,
    ARG_CLS_L1DIST,
    ARG_CLS_DOTPROD,
]

ARG_RESNET50_MODEL = "resnet50"
ARG_DEIT_MODEL = "deit"

POSSIBLE_NET_MODELS = [
    ARG_RESNET50_MODEL,
    ARG_DEIT_MODEL,
]

LOCAL_META_FILE_PATH="/path/to/dataset/original_UKA_master_list_five_fold.csv"
LOCAL_IMG_FOLDER_PATH="/path/to/dataset/UKA_preprocessed/"

HPC_META_FILE_PATH = "/path/on/hpc/original_UKA_master_list_five_fold.csv"
HPC_IMG_FOLDER_PATH = "/path/on/hpc/UKA_preprocessed/"









