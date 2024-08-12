import torch
import numpy as np

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = dataset

        train_file = train_data_dir + str(idx) + '.npz'

        train_file = dataset
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = dataset

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

def read_client_data(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

def load_train_data(dataset, id, batch_size=None):
    if batch_size == None:
        batch_size = batch_size
    train_data = read_client_data(dataset, id, is_train=True)
    return torch.utils.data.DataLoader(train_data, batch_size, drop_last=True, shuffle=True)