#Modified from https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/LeNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms

from torchshape import tensorshape

import warnings
warnings.filterwarnings("ignore")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        image_size = (4,3,32,32) #(batch_size, in_channels, image_height, image_width)

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        out_size = tensorshape(self.conv1, image_size)
        print(out_size)

        self.maxpool1 = nn.MaxPool2d(2)
        out_size = tensorshape(self.maxpool1, out_size)
        print(out_size)


        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        out_size = tensorshape(self.conv2, out_size)
        print(out_size)

        
        self.maxpool2 = nn.MaxPool2d(2)
        out_size = tensorshape(self.maxpool2, out_size)
        print(out_size)


        self.flatten = nn.Flatten()
        out_size = tensorshape(self.flatten, out_size)
        print(out_size)


        self.fc1 = nn.Linear(out_size[-1], 120)
        out_size = tensorshape(self.fc1, out_size)
        print(out_size)


        self.fc2 = nn.Linear(120, 84)
        out_size = tensorshape(self.fc2, out_size)
        print(out_size)


        self.fc3 = nn.Linear(84, 10)
        out_size = tensorshape(self.fc3, out_size)
        print(out_size)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class AlexNet_reduced(nn.Module):

    def __init__(self):
        super(AlexNet_reduced, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=32, shuffle=False)

# model = LeNet()

# model.eval()
# with torch.no_grad():
#     data, target = next(iter(test_loader))
#     output = model(data)

model = AlexNet_reduced()
with torch.no_grad():
    data, target = next(iter(test_loader))
    output = model(data)
    inp = data
    inp_size = inp.shape
    for f in model.features:
        inp = f(inp)
        inp_size = tensorshape(f, inp_size)
        if type(f).__name__ == "Conv2d": print()
        print(type(f).__name__, ":", inp_size)
    
    inp = model.avgpool(inp)
    inp_size = tensorshape(model.avgpool, inp_size)
    print(type(model.avgpool).__name__, ":", inp_size)


    inp = torch.flatten(inp, 1)
    # inp_size = tensorshape(torch.flatten(inp, 1), inp_size)
    # print(type(torch.flatten).__name__, ":", inp_size)

    print("------------")

    # inp = model.classifier(inp)
    # inp_size = tensorshape(model.classifier, inp_size)
    # print("Classifier", ":", inp_size)


    for f in model.classifier:
        inp = f(inp)
        try:
            inp_size = tensorshape(f, inp_size)
            print(type(f).__name__, ":", inp_size)
        except:
            pass

