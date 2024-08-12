import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time

import torch
import os

import nets as models
import functions as fns
from non_iid_generator.customDataset import CustomDataset

from tqdm import tqdm
import json 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
def compute_accuracy(output, target, debug=False):
    output = output.argmax(dim=1)
    acc = 0.0
    acc = torch.sum(target == output).item()
    acc = acc/output.size(0)*100
    return acc

def eval(test_loader, model, debug=False):
    batch_time = AverageMeter()
    acc = AverageMeter()

    # switch to eval mode
    model.eval()
    model = model.to(DEVICE)

    end = time.time()
    for i, (images, target) in enumerate(test_loader):

        if len(target.shape) > 1: target = target.reshape(len(target))

        images = images.to(DEVICE)
        target = target.to(DEVICE)
        
        output = model(images)
        batch_acc = compute_accuracy(output, target, debug=debug)
        acc.update(batch_acc, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Update statistics
        estimated_time_remained = batch_time.get_avg()*(len(test_loader)-i-1)
        # fns.update_progress(i, len(test_loader), 
        #     ESA='{:8.2f}'.format(estimated_time_remained)+'s',
        #     acc='{:4.2f}'.format(float(batch_acc))
        #     )
    # print('Test accuracy: {:4.2f}% (time = {:8.2f}s)'.format(
    #         float(acc.get_avg()), batch_time.get_avg()*len(test_loader)))
    # print('===================================================================')
    return float(acc.get_avg())
   
def federated_eval(model, no_clients, dataset_path):

    train_acc_list = []
    test_acc_list = []
    train_len = []
    test_len = []

    for client_id in range(no_clients):

        train_dataset_path = os.path.join(dataset_path, "train", f"{client_id}.pkl")
        train_data = pickle.load(open(train_dataset_path, "rb"))
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=128,
            shuffle=True)

        test_dataset_path = os.path.join(dataset_path, "test", f"{client_id}.pkl")
        test_data = pickle.load(open(test_dataset_path, "rb"))
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=128,
            shuffle=True)
        
        len_train = len(train_data)
        len_test = len(test_data)

        train_acc = eval(train_loader, model)
        test_acc = eval(test_loader, model)

        train_acc_list.append(train_acc * len_train)
        test_acc_list.append(test_acc * len_test)
        train_len.append(len_train)
        test_len.append(len_test)
    
    aggregated_train_accuracy = sum(train_acc_list) / sum(train_len)
    aggregated_test_accuracy = sum(test_acc_list) / sum(test_len)
   
    return aggregated_train_accuracy, aggregated_test_accuracy


def tester(key, test_path, data_path, no_clients):

    resources = []
    train_accs = []
    test_accs = []

    for iter_no in range(1,16):
        try:
            model = torch.load(os.path.join(test_path,f"master/iter_{iter_no}_best_model.pth.tar"), map_location=torch.device(DEVICE))
            network_def = fns.get_network_def_from_model(model, (3,32,32))
            resource = fns.compute_resource(network_def, "FLOPS")
            resources.append(resource)
            train_acc, test_acc = federated_eval(model, no_clients, data_path)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        except Exception as e:
            print(e)
            resources.append(None)
            train_accs.append(None)
            test_accs.append(None)

    return {
        "resource" : resources,
        "train_acc" : train_accs,
        "test_acc" : test_accs
    }

tests = [
    {
        "test_name": "fedavg_56c_basic_prior",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_a03_56c",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
    {
        "test_name": "fedavg_56c_basic_0",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_56c",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
    {
        "test_name": "fedavg_56c_basic_1",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_56c_1",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },{
        "test_name": "fedavg_56c_basic_2",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_56c_2",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
    {
        "test_name": "fedavg_56c_basic_3",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_56c_3",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
    #############
    #############
    #############
    {
        "test_name": "fedavg_56c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_56c",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedavg_56c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_56c",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },

        {
        "test_name": "fedavg_56c_optimized_coeff08",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_2_a03_56c_coeff08",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_56c_a03"
    },
    
    {
        "test_name": "fedavg_112c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_112c_1",
        "no_clients": 112,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_112c_a03"
    },
        

    {
        "test_name": "fedavg_112c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_112c_1",
        "no_clients": 112,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_112c_a03"
    },
        

    {
        "test_name": "fedavg_112c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_112c_1",
        "no_clients": 112,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_112c_a03"
    },
        

    {
        "test_name": "fedavg_210c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_210c_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedavg_210c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_210c_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedavg_210c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_210c_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedavg_210c_optimized_nr50",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_210c_nr50",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedprox_56c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_basic_a03_56c_fedprox_1",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedprox_56c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_network_optimized_a03_56c_fedprox_1",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedprox_56c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_56c_fedprox_1",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedprox_210c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_basic_a03_210c_fedprox_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedprox_210c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_network_optimized_a03_210c_fedprox_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedprox_210c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_210c_fedprox_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },


    #####
    #####

    {
        "test_name": "fed_master_optimized_a03_210c_1_ft100",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_210c_1_ft100",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },

    {
        "test_name": "fed_master_optimized_a03_210c_1_ft200",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_210c_1_ft200",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },

    {
        "test_name": "fed_master_optimized_a03_210c_nr100",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_210c_nr100",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },

    {
        "test_name": "fed_master_optimized_a03_210c_nr50",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_210c_nr50",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },

]



# tests = {
#     "fedavg_56c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_56c_3",
#     "fedavg_56c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_56c",
#     "fedavg_56c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_56c",

#     "fedavg_112c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_112c_1",
#     "fedavg_112c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_112c_1",
#     "fedavg_112c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_112c_1",

#     "fedavg_210c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_210c_1",
#     "fedavg_210c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_210c_1",
#     "fedavg_210c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_210c_1",

#     "fedavg_210c_optimized_nr50" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_210c_nr50",
    
#     "fedprox_56c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_basic_a03_56c_fedprox_1",
#     "fedprox_56c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_network_optimized_a03_56c_fedprox_1",
#     "fedprox_56c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_56c_fedprox_1",

#     "fedprox_210c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_basic_a03_210c_fedprox_1",
#     "fedprox_210c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_network_optimized_a03_210c_fedprox_1",
#     "fedprox_210c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_210c_fedprox_1",
# }


test_results = {

}
for test in tests:

    print(test["test_name"])

    test_results[test["test_name"]] = tester(
        test["test_name"],
        test["test_path"],
        test["data_path"],
        test["no_clients"],
        )
    
    print("ok.")

with open("/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/comp_3.json","w") as f:
    json.dump(test_results,f)

print("DONE...")