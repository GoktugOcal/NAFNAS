import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from non_iid_generator.utils.dataset_utils import check, separate_data, split_data, save_file, split_server_data, server_data_save
from non_iid_generator.customDataset import CustomDataset
from argparse import ArgumentParser
import pickle


random.seed(1)
np.random.seed(1)
# num_clients = 20
# num_classes = 10


# Allocate data to users
def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, alpha):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    server_path = dir_path + "server/"

    if check(config_path, train_path, test_path, server_path, num_clients, num_classes, niid, balance, partition):
        return
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((227,227)),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    dataset_image = []
    dataset_image.extend(trainset.data)
    dataset_image.extend(testset.data)
    dataset_image = np.array(dataset_image)

    train_targets = trainset.targets
    test_targets = testset.targets

    dataset_label = []
    dataset_label.extend(train_targets)
    dataset_label.extend(test_targets)
    dataset_label = np.array(dataset_label)

    ################## Splitting ##################
    ###############################################

    ######## FOR SERVER'S INITIAL TRAINING ########
    split_idx = int(len(dataset_label)/(num_clients+1))
    initial_train_image = dataset_image[:split_idx]
    initial_train_label = dataset_label[:split_idx]

    train_dataset, test_dataset = split_server_data(
        initial_train_image=initial_train_image,
        initial_train_label=initial_train_label,
        split_idx=split_idx,
        transform=transform)

    server_data_save(
        server_path=server_path,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )

    ######## REST OF THE DATA ########
    dataset_image = dataset_image[split_idx:]
    dataset_label = dataset_label[split_idx:]

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    alpha, niid, balance, partition)
    train_data, test_data = split_data(X, y)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, transform, alpha, niid, balance, partition)


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-dir", "--dir_path", type=str,
                            help="Directory to save the generated data.")
    arg_parser.add_argument("-nc", "--nc", type=int,
                            help="Number of clients.")
    arg_parser.add_argument("-c", "--c", type=int,
                            help="Numer of classes")
    arg_parser.add_argument("-niid", "--niid", type=str,
                            help="Non-IID format.")
    arg_parser.add_argument("-b", "--b", type=str,
                            help="Balanced scenario")
    arg_parser.add_argument("-p", "--p",
                            help="Pathological or Practical(Dirichlet) scenerio . (pat/dir)")
    arg_parser.add_argument("-a", "--alpha",
                            help="alpha value in Dirichlet distribution.")

    args = arg_parser.parse_args()

    dir_path = "./data/MNIST_NIID_200c/"
    dir_path = args.dir_path

    print(args)

    niid = True if args.niid == "noniid" else False
    balance = True if args.b == "balance" else False
    partition = args.p if args.p != "-" else None

    num_clients = args.nc
    num_classes = args.c
    alpha = args.alpha

    generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition)