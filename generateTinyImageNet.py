import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from argparse import ArgumentParser
import pickle
from operator import itemgetter

from non_iid_generator.utils.dataset_utils import check, split_server_data, server_data_save
from non_iid_generator.utils.dataset_utils_imagenet import separate_data, split_data, save_file, file_to_tensor_formatter

random.seed(1)
np.random.seed(1)
dir_path = "./data/timagenet/"


# http://cs231n.stanford.edu/tiny-imagenet-200.zip
# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
            self.data = np.array(imagefolder_obj.samples)[self.dataidxs,0]
            self.targets = np.array(imagefolder_obj.samples)[self.dataidxs,1]

        else:
            self.samples = np.array(imagefolder_obj.samples)
            self.data = np.array(imagefolder_obj.samples)[:,0]
            self.targets = np.array(imagefolder_obj.samples)[:,1].astype(int)
        
    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


# Allocate data to users
def generate_dataset(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    server_path = dir_path + "server/"

    if check(config_path, train_path, test_path, server_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get data
    transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4), 
            # transforms.Resize(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    trainset = ImageFolder_custom(root=dir_path+'tiny-imagenet-200/train/', transform=transform)
    # testset = ImageFolder_custom(root=dir_path+'tiny-imagenet-200/test/', transform=transform)
    # valset = ImageFolder_custom(root=dir_path+'tiny-imagenet-200/val/', transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=len(testset), shuffle=False)
    # valloader = torch.utils.data.DataLoader(
    #     valset, batch_size=len(valset), shuffle=False)


    # for _, train_data in enumerate(trainloader, 0):
    #     trainset.data, trainset.targets = train_data
    # for _, test_data in enumerate(testloader, 0):
    #     testset.data, testset.targets = test_data
    # for _, val_data in enumerate(valloader, 0):
    #     valset.data, valset.targets = val_data

    dataset_image = []
    dataset_image.extend(trainset.data)
    # dataset_image.extend(testset.data)
    # dataset_image.extend(valset.data)
    # dataset_image = np.array(dataset_image)

    dataset_label = []
    dataset_label.extend(trainset.targets)
    # dataset_label.extend(testset.targets)
    # dataset_label.extend(valset.targets)
    dataset_label = np.array(dataset_label).astype(int)

    dataset_tuples = [(dataset_image[i],int(dataset_label[i])) for i in range(len(dataset_label))]

    ################## Splitting ##################
    ###############################################

    ######## FOR SERVER'S INITIAL TRAINING ########
    # split_idx = int(len(dataset_label)/3)
    sample_size_each_class = int(len(dataset_label) / num_classes / 3)

    from collections import defaultdict
    class_data = defaultdict(list)
    for idx, item in enumerate(dataset_tuples):
        data, class_label = item
        class_data[class_label].append(idx)

    sampled_data = []
    client_idx = []

    # Perform uniform sampling by class
    for class_label, data in class_data.items():
        if len(data) >= sample_size_each_class:
            sampled_data.extend(random.sample(data, sample_size_each_class))
        else:
            sampled_data.extend(data)  # If a class has fewer samples, include all of them

    server_data = itemgetter(*sampled_data)(dataset_tuples)

    initial_train_image = np.array([item[0] for item in server_data])
    initial_train_label = np.array([item[1] for item in server_data])

    train_dataset, test_dataset = split_server_data(
        initial_train_image=initial_train_image,
        initial_train_label=initial_train_label,
        split_idx=None,
        transform=transform)
    
    train_dataset = file_to_tensor_formatter(train_dataset)
    test_dataset = file_to_tensor_formatter(test_dataset)

    server_data_save(
        server_path=server_path,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )

    del train_dataset, test_dataset

    ######## REST OF THE DATA ########
    client_idx = [i for i in range(len(dataset_tuples)) if i not in sampled_data]
    client_data = itemgetter(*client_idx)(dataset_tuples)

    dataset_image = np.array([item[0] for item in client_data])
    dataset_label = np.array([item[1] for item in client_data])

    print("start separate_data")
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)

    print("start split_data")
    train_data, test_data = split_data(X, y)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, transform, niid, balance, partition)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
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

    args = arg_parser.parse_args()

    print(args)

    niid = True if args.niid == "noniid" else False
    balance = True if args.b == "balance" else False
    partition = args.p if args.p != "-" else None

    num_clients = args.nc
    num_classes = args.c

    generate_dataset(dir_path, num_clients, num_classes, niid, balance, partition)