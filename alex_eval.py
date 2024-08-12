from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
import os
import time
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import pickle
import numpy as np

import nets as models
import functions as fns
from non_iid_generator.customDataset import CustomDataset
from utils import imagenet_loader

_NUM_CLASSES = 200
# DEVICE = os.environ["TORCH_DEVICE"]
DEVICE = "cuda"


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def get_avg(self):
        return self.avg
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    
def compute_accuracy(output, target):
    output = output.argmax(dim=1)
    acc = 0.0
    acc = torch.sum(target == output).item()
    acc = acc/output.size(0)*100
    return acc

def eval(test_loader, model):
    batch_time = AverageMeter()
    acc = AverageMeter()

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(test_loader):
        images = images.to(DEVICE)
        target = target.to(DEVICE)

        if len(target.shape) > 1: target = target.reshape(len(target))
        
        output = model(images)
        batch_acc = compute_accuracy(output, target)
        acc.update(batch_acc, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Update statistics
        estimated_time_remained = batch_time.get_avg()*(len(test_loader)-i-1)
        fns.update_progress(i, len(test_loader), 
            ESA='{:8.2f}'.format(estimated_time_remained)+'s',
            acc='{:4.2f}'.format(float(batch_acc))
            )
    print()
    print('Test accuracy: {:4.2f}% (time = {:8.2f}s)'.format(
            float(acc.get_avg()), batch_time.get_avg()*len(test_loader)))
    print('===================================================================')
    return float(acc.get_avg())

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4), 
            # transforms.Resize(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    batch_size = 32
    lr = 0.001

    data = "./data"

    train_dataset_path = os.path.join(data, "timagenet/server/train.npz")
    train_data = np.load(train_dataset_path, allow_pickle=True)["data"].tolist()
    train_dataset = CustomDataset(train_data["x"], train_data["y"], transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True)
    
    test_dataset_path = os.path.join(data, "timagenet/server/test.npz")
    test_data = np.load(test_dataset_path, allow_pickle=True)["data"].tolist()
    test_dataset = CustomDataset(test_data["x"], test_data["y"], transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size,
        shuffle=True)

    # Network
    cudnn.benchmark = True
    num_classes = _NUM_CLASSES
    model = torchvision.models.alexnet(pretrained=True)
    
    model = model.to(DEVICE)

    acc = eval(train_loader, model)
    print(acc)