import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
import torchvision.models as models

import io
import sys
import pickle
from tqdm import tqdm

from copy import deepcopy

model = torch.load("./models/alexnet/model_pt21.pth.tar")

### FINE TUNING
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_NUM_CLASSES = 10
batch_size = 128
num_workers = 4
momentum = 0.9
weight_decay = 1e-4
finetune_lr = 0.001
iterations = 500
print_frequency = 10

dataset_path = "data/"
train_dataset = datasets.CIFAR10(root="data/", train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=True, shuffle=True)#, sampler=train_sampler)

val_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True) #, sampler=valid_sampler)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), finetune_lr, 
                                momentum=momentum, weight_decay=weight_decay)
        
dataloader_iter = iter(train_loader)

correct, loss = 0, 0.0
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        
        labels.unsqueeze_(1)
        target_onehot = torch.FloatTensor(labels.shape[0], _NUM_CLASSES)
        target_onehot.zero_()
        target_onehot.scatter_(1, labels, 1)
        # labels.squeeze_(1)
        labels = labels.to(DEVICE)

        outputs = model(images.to(DEVICE))
        labels_one_hot = target_onehot.to(DEVICE)

        loss += criterion(outputs, labels_one_hot).item()
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        print(f"ACC: {correct / len(outputs.data)}")

accuracy = correct / len(val_loader.dataset)