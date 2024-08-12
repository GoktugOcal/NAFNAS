from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
import warnings
from collections import OrderedDict
import json
import io
import os
import sys
import pickle
import base64
from traceback import print_exc

import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from time import sleep, time

import logging

from non_iid_generator.customDataset import CustomDataset




def load_data(train_dataset_path, test_dataset_path):
    """Load CIFAR-10 (training and test set)."""

    batch_size = 128

    train_data = pickle.load(open(train_dataset_path, "rb"))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)

    test_data = pickle.load(open(test_dataset_path, "rb"))
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True)

    return train_loader, test_loader

if __name__ == "__main__":

    client_id = 0

    train_dataset_path = f"./data/Cifar10/train/{client_id}.pkl"
    test_dataset_path = f"./data/Cifar10/test/{client_id}.pkl"

    batch_size = 128

    train_data_client = pickle.load(open(train_dataset_path, "rb"))
    test_data_client = pickle.load(open(test_dataset_path, "rb"))
    

    train_data_server = pickle.load(open("./data/Cifar10/server/train.pkl", "rb"))
    test_data_server = pickle.load(open("./data/Cifar10/server/test.pkl", "rb"))
    
    print(train_data_client.data.shape)
    train_data_client.data = np.concatenate((train_data_client.data, train_data_server.data))
    print(train_data_client.data.shape)

    print(np.unique(train_data_server.labels, return_counts=True))



    
    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=batch_size,
    #     shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    #     test_data,
    #     batch_size=batch_size,
    #     shuffle=True)

    # print(train_data_client.data)