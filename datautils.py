import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import random

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)


def load_data(dataset_name, data_size, batch_size):
    x = np.load('./datasets/' + f'{dataset_name}/' + f'cluster_x_{data_size}.npy')
    y = np.load('./datasets/' + f'{dataset_name}/' + f'cluster_y_{data_size}.npy')
    dataset = Mydataset(x, y)
    print('n_cluster:', np.unique(y))

    # torch.manual_seed(123)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    return data_loader

def load_csv_data(data_path, data_name, split='TRAIN'):
    data_file = os.path.join(data_path, f"{split}.csv")
    label_file = os.path.join(data_path, f"{split}_label.csv")

    data = pd.read_csv(data_file, header=None).values #(样本数, 时间步长）
    labels = pd.read_csv(label_file, header=None).values.flatten()
    # 生成索引数组
    indices = np.arange(data.shape[0])  # 创建一个与数据行数相同的索引数组

    # 重塑数据形状为 (样本数, 时间步长, 特征数)
    data = data.reshape(data.shape[0], -1, 1)
    if data_name not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return data, labels, indices

    # #添加数据归一化 效果暂时不好 注释了
    mean = np.nanmean(data)
    std = np.nanstd(data)
    data = (data - mean) / std

    return data, labels, indices

def create_data_loader(data, labels, index, batch_size):
    #将时间序列数据中的有效数据部分居中
    temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        data = centerize_vary_length_series(data)

    data = data[~np.isnan(data).all(axis=2).all(axis=1)]

    tensor_data = torch.tensor(data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    tensor_index = torch.tensor(index, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_labels,tensor_index)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    return data_loader

#记得启用！！
def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

def set_seed(seed=123):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def save_results(results, results_path):
    results_df = pd.DataFrame(results, columns=['Dataset', 'ACC', 'NMI', 'ARI', 'RI', 'FMI'])
    results_df.to_csv(results_path, index=False)

def load_existing_results(results_path):
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    else:
        return pd.DataFrame(columns=['Dataset', 'ACC', 'NMI', 'ARI','RI', 'FMI'])


def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]