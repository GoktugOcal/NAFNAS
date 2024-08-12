from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
import warnings
from collections import OrderedDict
import json
import io
import os
import glob
import sys
import pickle
import base64
from traceback import print_exc

import numpy as np
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

DEVICE = os.environ["TORCH_DEVICE"]



def load_data(train_dataset_path, test_dataset_path):
    """Load CIFAR-10 (training and test set)."""

    batch_size = 128
    momentum = 0.9
    weight_decay = 1e-4
    finetune_lr = 0.001

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

def fine_tune(model, no_epoch, train_loader, print_frequency=100):
    '''
        short-term fine-tune a simplified model
        
        Input:
            `model`: model to be fine-tuned.
            `iterations`: (int) num of short-term fine-tune iterations.
            `print_frequency`: (int) how often to print fine-tune info.
        
        Output:
            `model`: fine-tuned model.
    '''

    # Data loaders for fine tuning and evaluation.
    batch_size = 128
    momentum = 0.9
    weight_decay = 1e-4
    finetune_lr = 0.001

    # train_loader, val_loader = load_data(train_dataset_path, test_dataset_path)

    criterion = torch.nn.BCEWithLogitsLoss()
    
    _NUM_CLASSES = 10
    optimizer = torch.optim.SGD(
        model.parameters(),
        finetune_lr, 
        momentum=momentum,
        weight_decay=weight_decay)

    model = model.to(DEVICE)
    model.train()
    # dataloader_iter = iter(train_loader)
    logging.info("Fine tuning started.")
    for i in tqdm(range(no_epoch)):
        if i % print_frequency == 0:
            logging.info('Fine-tuning Epoch {}'.format(i))
            sys.stdout.flush()
        for i, (input, target) in enumerate(train_loader):
            
            logging.info(f"\tBatch no: {i}")
            # (input, target) = next(dataloader_iter)
            
            # Ensure the target shape is sth like torch.Size([batch_size])
            if len(target.shape) > 1: target = target.reshape(len(target))

            target.unsqueeze_(1)
            target_onehot = torch.FloatTensor(target.shape[0], _NUM_CLASSES)
            target_onehot.zero_()
            target_onehot.scatter_(1, target, 1)
            target.squeeze_(1)
            input, target = input.to(DEVICE), target.to(DEVICE)
            target_onehot = target_onehot.to(DEVICE)

            pred = model(input)
            loss = criterion(pred, target_onehot)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()

            del loss, pred

    return model

def test(model, testloader):
    _NUM_CLASSES = 10
    """Validate the model on the test set."""
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    latency_measurements = []
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            
            target_onehot = torch.FloatTensor(labels.shape[0], _NUM_CLASSES)
            target_onehot.zero_()
            target_onehot.scatter_(1, labels, 1)
            labels = labels.to(DEVICE)
            
            s = time()
            outputs = model(images.to(DEVICE))
            labels_one_hot = target_onehot.to(DEVICE)
            latency_measurements.append(time() - s)

            loss += criterion(outputs, labels_one_hot).item()
            correct += (torch.max(outputs.data, 1)[1] == labels.squeeze_(1)).sum().item()

    accuracy = correct / len(testloader.dataset)
    print("Test Accuracy :", round(accuracy, 3), "| Total no samples :", len(testloader.dataset))
    return loss, accuracy, latency_measurements


if __name__ == "__main__":

    print("DEVICE:",DEVICE)
    test_name = "test-6"

    model_paths = glob.glob(f"./models/alexnet/test/{test_name}/worker/*.pth.tar")
    model_paths.sort()

    train_dataset_path = f"./data/Cifar10/server/train.pkl"
    test_dataset_path = f"./data/Cifar10/server/test.pkl"
    trainloader, testloader = load_data(train_dataset_path, test_dataset_path)

    main_cols = "iteration,block,path,server_acc,"
    train_cols = ",".join([f"train_acc_{i}" for i in range(10)])
    test_cols = ",".join([f"test_acc_{i}" for i in range(10)])

    with open(f"./models/alexnet/test/{test_name}/fine-tune.txt", "w") as f:
        f.write(main_cols+train_cols+test_cols+"\n")

    client_train_loaders = []
    client_test_loaders = []
    for client_id in range(10):
        client_train_dataset_path = f"./data/Cifar10/train/{client_id}.pkl"
        client_test_dataset_path = f"./data/Cifar10/test/{client_id}.pkl"
        client_trainloader, client_testloader = load_data(client_train_dataset_path, client_test_dataset_path)
        client_train_loaders.append(client_trainloader)
        client_test_loaders.append(client_testloader)

    for model_path in model_paths:
        print(model_path)
        splitted = model_path.split("_")
        iter_no = splitted[1]
        block_no = splitted[3]
        temp_model = torch.load(model_path).to(DEVICE)
        fine_tuned_model = fine_tune(
            model=temp_model,
            no_epoch=10,
            train_loader=trainloader,
            print_frequency=1
            )

        torch.save(fine_tuned_model, os.path.join("./models/alexnet/test",test_name,"fine-tuned",model_path.split("/")[-1]))
        print("Model fine tuned and saved.")

        _, server_test_acc, _ = test(temp_model, testloader)
        print(f"Model test acc: {server_test_acc}")

        client_train_acc = []
        client_test_acc = []
        for i in range(10):
            _, train_acc, _ = test(fine_tuned_model, client_train_loaders[i])
            _, test_acc, _ = test(fine_tuned_model, client_test_loaders[i])
            client_train_acc.append(train_acc)
            client_test_acc.append(test_acc)

        t_main_cols = f"{iter_no},{block_no},{model_path},{server_test_acc},"
        t_train_cols = ",".join([f"{client_train_acc[i]}" for i in range(10)])
        t_test_cols = ",".join([f"{client_test_acc[i]}" for i in range(10)])

        with open(f"./models/alexnet/test/{test_name}/fine-tune.txt", "a") as f:
            f.write(t_main_cols+t_train_cols+t_test_cols+"\n")

