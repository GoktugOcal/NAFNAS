from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
import os
import io
import base64

import pandas as pd
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

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('working_folder', type=str,
                            help='Root folder where models, related files and history information are saved.')
    arg_parser.add_argument('-ms', '--model_paths', nargs='+',
                            help="Model paths")
    arg_parser.add_argument('-bws', '--bandwidth_table', type=str, help="Bandwidth tiers table")
    
    args = arg_parser.parse_args()
    bw_table = pd.read_csv(args.bandwidth_table)

    model_metadata = {}
    for model_path in args.model_paths:
        model_size = os.path.getsize(model_path)
        model_metadata[model_path] = {
            "clients" : None,
            "size" : model_size / 1000000
        }
    
    model_metadata = dict(sorted(
        model_metadata.items(),
        key=lambda x:x[1]["size"],
        reverse=False))
    
    grp_no = 0
    for k, v in model_metadata.items():
        clients = bw_table[bw_table["group"] == grp_no]["client_id"].values.tolist()
        model_metadata[k]["clients"] = clients
        grp_no +=1


    os.mkdir(args.working_folder)
    # print(json.dumps(model_metadata, indent=2))
    with open(os.path.join(args.working_folder,"model_groups.json"), "w") as json_file:
        json.dump(model_metadata, json_file)

    # for model_path in model_metadata.keys():
    #     print(os.path.basename(model_path).split(".")[0])
        