import numpy as np
import pandas as pd
import json
import io
import os
import sys
import pickle
import math
import random
from time import time

import matplotlib.pyplot as plt


NO_CLIENTS = 210
NO_GROUPS = 7
NO_CLASSES = 10
ALPHA = "03"

replace_dict = {
    0: 7,
    1: 6,
    2: 5,
    3: 4,
    4: 3,
    5: 2,
    6: 1,
}

path = f"./data/alpha/fedavg/Cifar10_NIID_{NO_CLIENTS}c_a{ALPHA}/config.json"
conf = json.loads(open(path, "r").read())
data = [dict(zip(np.array(cli)[:,0], np.array(cli)[:,1])) for cli in conf["Size of samples for labels in clients"]]

main_label_vectors = np.zeros((NO_CLIENTS,NO_CLASSES))
for client_id in range(NO_CLIENTS):
    for class_id in range(NO_CLASSES):
        if class_id in data[client_id].keys():
            main_label_vectors[client_id][class_id] = data[client_id][class_id]

def spread_bw_random(main_label_vectors, no_clients, tiers):
    NO_CLIENTS = no_clients
    
    bw_types = []
    bw = []
    for k, v in tiers.items():
        for i in range(int(NO_CLIENTS / len(tiers))):
            bw_types.append(k)
            bw.append(random.choice(tiers[k]))

    print(int(NO_CLIENTS / len(tiers)))
    print(len(bw))
    print(len(bw_types))

    ids = np.arange(NO_CLIENTS)
    random.shuffle(ids)

    bws = []
    for idx in ids:
        bws.append((bw[idx],bw_types[idx]))

    clients_list = []
    for idx in range(len(main_label_vectors)):
        clients_list.append(
            {
                "client_id" : idx,
                "bw" : bws[idx][0],
                "bw_type" : bws[idx][1],
            }
        )

    clients_df = pd.DataFrame(clients_list)

    maindf = clients_df.sort_values("bw")
        
    maindf.reset_index(drop=True)

    return maindf

tiers = {
    "TIER_11" : np.arange(2_500_000, 3_500_000, 500_000),
    "TIER_12" : np.arange(4_000_000, 5_500_000, 500_000),
    "TIER_13" : np.arange(6_000_000, 8_000_000, 500_000),

    "TIER_21" : np.arange(8_500_000, 12_500_000, 1_000_000),
    "TIER_22" : np.arange(13_000_000, 16_000_000, 1_000_000),
    "TIER_23" : np.arange(17_000_000, 25_000_000, 1_000_000),
    "TIER_24" : np.arange(26_000_000, 30_000_000, 1_000_000),
    "TIER_25" : np.arange(31_000_000, 34_000_000, 1_000_000),

    "TIER_31" : np.arange(35_000_000, 40_000_000, 1_000_000),
    "TIER_32" : np.arange(41_000_000, 55_000_000, 1_000_000),
    "TIER_33" : np.arange(46_000_000, 65_000_000, 1_000_000),
    "TIER_34" : np.arange(66_000_000, 75_000_000, 1_000_000),
    "TIER_35" : np.arange(75_000_000, 100_000_000, 1_000_000),

    "TIER_41" : np.arange(100_000_000, 400_000_000, 20_000_000)
}

main_df = spread_bw_random(main_label_vectors, 210, tiers)

assignment = {
    "block_0": {},
    "block_1": {},
    "block_2": {},
    "block_3": {},
    "block_4": {},
    "block_5": {},
    "block_6": {},
}

for bid in assignment.keys():
    selected_clients = random.sample(list(main_df.client_id.values), 30)
    assignment[bid] = main_df.iloc[selected_clients]["bw_type"].to_dict()
    assignment[bid] = {str(idx):v for idx, (k,v) in enumerate(assignment[bid].items())} 

with open(f"./data/alpha/fedavg/Cifar10_NIID_210c_a03/client_groups.json","w") as f:
    json.dump(assignment,f)