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
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from time import sleep, time

import logging

from non_iid_generator.customDataset import CustomDataset

batch_size = 128

def load_time_test(train_loader):
    durations = []
    s = time()
    for i, (input, target) in enumerate(tqdm(train_loader)):
        duration = round(time()-s,4)
        durations.append(duration)
        logging.info(f"\tBatch no: {i} | loaded in {duration} seconds.")
        if i == 1: break
        s = time()
    return durations

if __name__ == "__main__":

    logging.basicConfig(
        filename="./logs/load_test_2.log",
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    # Cifar default w/ transform
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4), 
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
        transform=transform
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size)
    logging.info("Cifar default w/ transform")
    load_time_test(train_loader)

    # Cifar default w/o transform
    transform = transforms.Compose([
            transforms.ToTensor()
            ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size)
    logging.info("Cifar default w/o transform")
    load_time_test(train_loader)

    # Cifar pickle
    train_dataset_path = "./data/32_Cifar10_NIID_20c_a01/train/2.pkl"
    train_data = pickle.load(open(train_dataset_path, "rb"))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)
    logging.info("Cifar pickle")
    load_time_test(train_loader)

    # MNIST default w/ transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((227,227)),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    logging.info("MNIST default w/ transform")
    load_time_test(train_loader)

    # MNIST default w/o transform
    transform = transforms.Compose([
            transforms.ToTensor()
            ])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    logging.info("MNIST default w/o transform")
    load_time_test(train_loader)