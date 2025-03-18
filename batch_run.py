import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from fcacc import FCACCModel
import time
import datetime
import argparse
import datautils


def train_and_evaluate(data_name, data_path, log_dir, irregular, config):
    print(f"Processing dataset: {data_name}")


    train_data, train_labels, train_index = datautils.load_csv_data(data_path,data_name, split='TRAIN')
    test_data, test_labels, test_index = datautils.load_csv_data(data_path, data_name, split='TEST')

    if irregular > 0:
        train_data = datautils.data_dropout(train_data.astype(float), irregular)
        test_data = datautils.data_dropout(test_data.astype(float), irregular)
        print(f'step_mask{irregular} done')

    max_train_index = np.max(train_index)

    adjusted_test_index = test_index + max_train_index + 1

    combined_data = np.concatenate((train_data, test_data), axis=0)
    combined_labels = np.concatenate((train_labels, test_labels), axis=0)
    combined_index = np.concatenate((train_index, adjusted_test_index), axis=0)


    all_data_loader =  datautils.create_data_loader(combined_data, combined_labels, combined_index, config['batch_size'])

    config['dataset_name'] = data_name
    config['dataset_size'] = combined_data.shape[0] #优化
    config['timesteps_len'] = combined_data.shape[1]
    config['input_dims'] = combined_data.shape[2]
    print(f"config: {config}")


    unique_labels = np.unique(combined_labels)
    num_classes = len(unique_labels)

    config['n_cluster'] = num_classes

    model = FCACCModel(all_data_loader, **config)
    t = time.time()

    if config['pretraining_epoch'] != 0:
        model.Pretraining()
    if config['MaxIter'] != 0:
        model.Finetuning()

    t = time.time() - t
    print(f"Training time: {datetime.timedelta(seconds=t)}\n")

    model.encoder = torch.load(f"{data_name}_Finetuning_phase")
    print('finish load')

    model.dataset_size = combined_data.shape[0]
    acc, nmi, ari, ri, fmi = model.eval_with_test_data(data_name, log_dir, all_data_loader, model)
    print(f"test_data: acc={acc}  nmi={nmi} ari={ari} ri={ri} fmi={fmi}")

    return acc, nmi, ari, ri, fmi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='./datasets/UCRArchive_2018_csv/', type=str,
                        help='The dataset directory')
    parser.add_argument('--results_dir', default='./results/', type=str, help='The results directory')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size')
    parser.add_argument('--repr_dims', type=int, default=64, help='The representation dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate of pre-training phase')
    parser.add_argument('--pretraining_epoch', type=int, default=100, help='The epoch of pre-training phase')
    parser.add_argument('--MaxIter', type=int, default=30, help='The epoch of fine-tuning phase')
    parser.add_argument('--w_c', type=float, default=0.2, help='The epoch of fine-tuning phase')
    parser.add_argument('--m', type=float, default=1.5, help='The epoch of fine-tuning phase')
    parser.add_argument('--seed', type=int, default=1127, help='The epoch of fine-tuning phase')
    parser.add_argument('--irregular', type=float, default=0, help='')
    parser.add_argument('--hard_w', type=float, default=0.2, help='hard negative hardness')
    parser.add_argument('--explanation', type=str, default="", help='explanation')
    args = parser.parse_args()


    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    config = {
        'batch_size': args.batch_size,
        'output_dims': args.repr_dims,
        'lr': args.lr,
        'pretraining_epoch': args.pretraining_epoch,
        'MaxIter': args.MaxIter,
        'w_c':args.w_c,
        'm': args.m,
        'hard_w':args.hard_w,
    }

    data_list = os.listdir(args.dataset_dir)
    results = []
    datautils.set_seed(args.seed)

    now = datetime.datetime.now()
    # 提取年份、月份和日期
    current_year = now.year
    current_month = now.month
    current_day = now.day

    results_dir = os.path.join(args.results_dir, f"results_{config['batch_size']}_{config['lr']}_{config['pretraining_epoch']}\
_{config['MaxIter']}_{config['w_c']}_{config['m']}_{args.seed}_{args.irregular}_{args.repr_dims}_{args.hard_w}_{current_year}.{current_month}.{current_day} {args.explanation}.csv")

    log_dir = os.path.join(args.results_dir, f"results_{config['batch_size']}_{config['lr']}_{config['pretraining_epoch']}\
_{config['MaxIter']}_{config['w_c']}_{config['m']}_{args.seed}_{args.irregular}_{args.repr_dims}_{args.hard_w}_{current_year}.{current_month}.{current_day}")
    config['log_dir'] = log_dir

    existing_results_df = datautils.load_existing_results(results_dir)
    completed_datasets = set(existing_results_df['Dataset'])

    results = existing_results_df.values.tolist()


    for data_name in data_list:
        if data_name in completed_datasets:
            print(f"Skipping already completed dataset: {data_name}")
            continue

        data_path = os.path.join(args.dataset_dir, data_name)
        if os.path.isdir(data_path):
            acc, nmi, ari, ri, fmi = train_and_evaluate(data_name, data_path, log_dir, irregular = args.irregular, config = config)
            results.append((data_name, acc, nmi, ari, ri, fmi))

            datautils.save_results(results, results_dir)  # 每次成功运行后立即保存结果




if __name__ == "__main__":
    t = time.time()
    main()
    t = time.time() - t
    print(f"\nrunning time: {datetime.timedelta(seconds=t)}\n")
    # os.system("/usr/bin/shutdown")
