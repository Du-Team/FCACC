# Fuzzy Cluster-Aware Contrastive Clustering for Time Series (FCACC)
FCACC (Fuzzy Cluster-Aware Contrastive Clustering) is an innovative unsupervised time series clustering framework that combines the advantages of contrastive learning and fuzzy clustering. It is designed to improve the representation learning and clustering performance of time series data. This method introduces a three-view data augmentation strategy to comprehensively capture the diverse features of time series, and employs a cluster-aware hard negative sample generation mechanism along with a clustering-aware positive and negative sample pair selection to effectively enhance the model's discriminative power. By utilizing the fuzzy c-means (FCM) clustering algorithm, FCACC can handle complex time series structures, dynamically generate cluster structures, and guide the optimization process of contrastive learning.

## Introduction

A Python implementation of the clustering algorithm presented in:

     <b><i> Congyu Wang, Mingjing Du*, Xiang Jiang, Yongquan Dong*. Fuzzy cluster-aware contrastive clustering for time series. Pattern Recognition, 2026, 173: 112899.</i></b>

The paper is available online at: <a href="https://dumingjing.github.io/files/paper-21_Fuzzy%20cluster-aware%20contrastive%20clustering%20for%20time%20series/2026_PR_Fuzzy%20cluster-aware%20contrastive%20clustering%20for%20time%20series.pdf">pdf</a>. 


If you use this implementation in your work, please add a reference/citation to the paper. You can use the following bibtex entry:

```
@article{wang26,
  author       = {Congyu Wang and
                  Mingjing Du and
                  Xiang Jiang and
                  Yongquan Dong},
  title        = {Fuzzy cluster-aware contrastive clustering for time series},
  journal      = {Pattern Recognition},
  volume       = {173},
  pages        = {112899},
  year         = {2026},
}
```

## Requirements
The recommended requirements for FCACC are specified as follows:
* Python 3.8.10
* numpy==1.21.4
* pandas==2.0.3
* torch==1.10.0+cu113
* torchvision==0.11.1+cu113
* scikit-learn==1.3.2
* scipy==1.10.1
* matplotlib==3.5.0
* tqdm==4.62.3
* pillow==8.4.0
* scikit-image==0.19.3
* jupyterlab==3.2.4
* ipykernel==6.5.1
* seaborn==0.11.2



Note that you should have CUDA installed for running the code.

### Dataset Preparation

Before running FCACC, it is essential to prepare the dataset. The time series dataset used in this project is sourced from the [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

If you encounter any issues while downloading the dataset, you can also use the datasets available in the `dataset` directory. The datasets are stored in `dataset(1).7z` and `dataset(2).7z` files. A brief overview of the datasets is provided in the `DataSummary.xlsx` file located in the same directory. For detailed information about each dataset, please refer to the `README.md` file within each dataset's respective folder inside the compressed archive.


## Usage

To train FCACC on UCR dataset, run the following command:

```run
python batch_run.py
```

## Hyperparameters

The following hyperparameters are used to configure the FCACC model training and fine-tuning process:

- **`batch_size`** (default: `8`): The number of samples processed together in one forward/backward pass.   Smaller batch sizes can help the model generalize better, but may increase training time.

- **`repr_dims`** (default: `64`): The number of dimensions for the representation (embedding) space.   This determines the size of the feature vectors used by the model.

- **`lr`** (default: `0.001`): The learning rate for the pre-training phase.   This controls how much the model's weights are adjusted with respect to the loss gradient.

- **`pretraining_epoch`** (default: `100`): The number of epochs to train the model during the pre-training phase.   Pre-training helps the model learn useful representations before fine-tuning.

- **`MaxIter`** (default: `30`): The number of iterations (epochs) for the fine-tuning phase.   This phase focuses on refining the pre-trained model to achieve better accuracy on the specific dataset.

- **`w_c`** (default: `0.2`): A hyperparameter controlling the weight of a specific regularization term in the loss function.   It helps balance the model's complexity and generalization ability.

- **`m`** (default: `1.5`): A hyperparameter that influences the magnitude of the loss function, affecting the learning process during fine-tuning.   It may impact how quickly the model converges.

- **`seed`** (default: `1127`): The random seed used for initializing the model and data shuffling.

- **`irregular`** (default: `0`): Controls the dropout rate applied to the training data.   A non-zero value simulates missing data (irregularities) by randomly dropping parts of the data.   This can help make the model more robust to missing or noisy inputs.

- **`hard_w`** (default: `0.2`): The weight parameter for "hard negatives," which helps control the difficulty of negative samples used during training.   Adjusting this can improve the model's ability to discriminate between similar and dissimilar samples.

- **`explanation`** (default: `""`): A custom string that can be added to the results file name to describe or categorize the specific experiment being run.

These hyperparameters are configurable through command-line arguments when running the script, allowing you to tailor the training process to your specific dataset and objectives.
