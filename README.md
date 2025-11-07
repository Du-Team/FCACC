# Fuzzy Cluster-Aware Contrastive Clustering for Time Series (FCACC)



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
