
# An Ordinal Regression Framework for a Deep Learning Based Severity Assessment for Chest Radiographs

In this repository we share our code from the identically named paper. You can find it using the following link: [TODO ADD LINK].
Unfortunately, we are not allowed to share the clinical dataset we used.


## Requirements
| module |
| :-------- |
| `torch`      |
| `torchvision`      |
| `torchmetrics`      |
| `pytorch_lightning`      |
| `PIL`      |
| `wandb`      |

---
## Run the code (example)

```
    python3 argParser.py \
    --batchSize 32 \
    --learningRate 0.005 \
    --numEpochs 30 \
    --netModel resnet50 \
    --fiveFold 0 \
    --targetFunc gauss \
    --classFunc "l1dist;dotProd" \
    /
```

| Argument | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `--batchSize` | `int` | **Required**. Batch size |
| `--learningRate` | `float` | **Required**. Learning rate |
| `--numEpochs` | `int` | **Required**. Number of epochs |
| `--netModel` | `str` | **Required**. Used model (e.g. `resnet50`, `deit`) |
| `--fiveFold` | `int` | **Required**. Fold of five fold cross validation |
| `--targetFunc` | `str` | **Required**. See below |
| `--classFunc` | `str` | **Required**. See below |

### Target Function
The target function can be specified with the `--targetFunc` argument.

| `--targetFunc` | Description                  |
| :-------- | :-------------------------------- |
| `gauss`      | Gaussian encoding |
| `oneHot`      | One-Hot encoding |
| `continuous`      | Continuous encoding |
| `progBar`      | Progress-Bar encoding |
| `softProg`      | Soft-Progress-Bar encoding |
| `binNum`      | Binary-Number encoding |

The exact mappings are defined in [`targetMap.py`](targetMap.py).

### Classification Function
The target function can be specified with the `--classFunc` argument. Since the training does not depend on the classification function, several can be evaluated on the same training.
To do this, separate the different arguments for the classification function with a `;`.

| `--classFunc` | Description                  |
| :-------- | :-------------------------------- |
| `argmax`      | Argmax function |
| `sum`      | Sum of vector |
| `l1dist`      | L1 distance |
| `dotProd`      | Normalized dot product |

The implementation of this can be found in [`classMap.py`](classMap.py)

## How to cite our work

If you find the paper or code useful for your academic work, please consider citing our work.:
```
[TODO ADD BIBTEX]
```





