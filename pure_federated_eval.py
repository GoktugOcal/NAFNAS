from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
import os
import io
import base64

import json
import pickle
from time import time
import torch
import torchvision
from torchvision import transforms

from shutil import copyfile
import subprocess
import sys
import warnings
import common
from traceback import print_exc

from server import flower_server_execute, NetStrategy

from copy import deepcopy
from collections import OrderedDict

import logging
from tqdm import tqdm

DEVICE = os.getenv("TORCH_DEVICE")
DEVICE = "cuda"
_MASTER_FOLDER_FILENAME = 'master'



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

def test(net, testloader):
    _NUM_CLASSES = 10
    """Validate the model on the test set."""
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    latency_measurements = []
    with torch.no_grad():
        for images, labels in tqdm(testloader):

            labels = labels.reshape(-1,1)
            # if len(labels.shape) > 1: labels = labels.reshape(len(labels))
            
            target_onehot = torch.FloatTensor(labels.shape[0], _NUM_CLASSES)
            target_onehot.zero_()
            target_onehot.scatter_(1, labels, 1)
            labels = labels.to(DEVICE)
            
            s = time()
            outputs = net(images.to(DEVICE))
            labels_one_hot = target_onehot.to(DEVICE)
            latency_measurements.append(time() - s)

            loss += criterion(outputs, labels_one_hot).item()
            correct += (torch.max(outputs.data, 1)[1] == labels.squeeze_(1)).sum().item()

    accuracy = correct / len(testloader.dataset)
    print("Test Accuracy :", round(accuracy, 3), "| Total no samples :", len(testloader.dataset))
    return loss, accuracy, latency_measurements

def federated_eval(model, args):

    # Define data transformations
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Download and load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Create data loaders for training and testing
    batch_size = 128
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    _, accuracy, _ = test(net=model,testloader=trainloader)
    logging.info(f"Overall | Train Acc: {round(accuracy, 2)}")
    _, accuracy, _ = test(net=model,testloader=testloader)
    logging.info(f"Overall | Test Acc: {round(accuracy, 2)}")

    train_dataset_path = f"./data/Cifar10/server/train.pkl"
    test_dataset_path = f"./data/Cifar10/server/test.pkl"
    trainloader, testloader = load_data(train_dataset_path, test_dataset_path)
    _, accuracy, _ = test(net=model,testloader=testloader)
    logging.info(f"Server | Acc: {round(accuracy, 2)}")

    for client_id in range(args.nc):
        train_dataset_path = f"./data/Cifar10/train/{client_id}.pkl"
        test_dataset_path = f"./data/Cifar10/test/{client_id}.pkl"
        trainloader, testloader = load_data(train_dataset_path, test_dataset_path)
        _, accuracy, _ = test(net=model,testloader=testloader)
        logging.info(f"Client {client_id} | Acc: {round(accuracy, 2)}")

def folder_things(args):
    master_folder = os.path.join(args.working_folder, _MASTER_FOLDER_FILENAME)
    # Create the folder structure.
    if not os.path.exists(args.working_folder):
        os.makedirs(args.working_folder)
        print('Create directory', args.working_folder)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('working_folder', type=str,
                            help='Root folder where models, related files and history information are saved.')
    arg_parser.add_argument('-nc', '--nc', type=int,
                            help="Number of clients for federated learning.")
    args = arg_parser.parse_args()
    
    folder_things(args)

    logfilename = os.path.join(args.working_folder,"logs.txt")
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    logging.basicConfig(
        filename=logfilename,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    try:
        no_clients = args.nc

        logging.info("Model")
        # model = torch.load("models/alexnet/model_cuda.pth.tar", map_location=torch.device(DEVICE))
        # model = torch.load("models/alexnet/model_cpu_2.pth.tar", map_location=torch.device(DEVICE))
        model = torch.load("models/alexnet/iter_6_best_model.pth.tar", map_location=torch.device(DEVICE))
        logging.info("Model loaded")

        federated_eval(model, args)

        logging.info("DONE")

    except Exception as e:
        print_exc()
        logging.error(e)