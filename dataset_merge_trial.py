from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
import os
import time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import pickle
import numpy as np
import json

from copy import deepcopy
import shutil

from flwr.common.parameter import *
from flwr.server.strategy.aggregate import *
from collections import OrderedDict

import nets as models
import functions as fns
from non_iid_generator.customDataset import CustomDataset

data = "./projects/test_1_fed_sim_NIID_alpha05_ft20/data"
batch_size = 128

server_train_dataset_path = os.path.join(data,"server","train.pkl")
server_train_data = pickle.load(open(server_train_dataset_path, "rb"))

train_dataset_path = os.path.join(data, "train", f"0.pkl")
train_data = pickle.load(open(train_dataset_path, "rb"))
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True)

train_data.data = np.concatenate((train_data.data, server_train_data.data))
train_data.labels = np.concatenate((train_data.labels, server_train_data.labels))