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

from multiprocessing import Pool

import matplotlib.pyplot as plt

replace_dict = {
    0: 7,
    1: 6,
    2: 5,
    3: 4,
    4: 3,
    5: 2,
    6: 1,
}

def vis_bw_counts(maindf, no_clients, no_groups):

    member_per_group = int(no_clients / no_groups)

    x = np.array(np.meshgrid(np.arange(1,no_groups+1),np.arange(1,member_per_group+1))).T.reshape(-1,2)[:,0]
    y = np.array(np.meshgrid(np.arange(1,no_groups+1),np.arange(1,member_per_group+1))).T.reshape(-1,2)[:,1]
    s = np.ones((no_groups,member_per_group))*(600/8*member_per_group)
    
    tiers = [[] for i in range(no_groups)]
    for group_name, group_df in maindf.groupby("group"):
        tiers[group_name-1] = group_df["bw_type"].tolist()


    circle_size = member_per_group/600 * 8
    s_low = []
    for group_no in range(no_groups):
        s_low.append([])
        for client_no in range(member_per_group):
            tier = tiers[group_no][client_no]
            if tier == "low": size = circle_size
            else: size = 0
            s_low[group_no].append(size)

    s_medium = []
    for group_no in range(no_groups):
        s_medium.append([])
        for client_no in range(member_per_group):
            tier = tiers[group_no][client_no]
            if tier == "medium": size = circle_size
            else: size = 0
            s_medium[group_no].append(size)

    s_high = []
    for group_no in range(no_groups):
        s_high.append([])
        for client_no in range(member_per_group):
            tier = tiers[group_no][client_no]
            if tier == "high": size = circle_size
            else: size = 0
            s_high[group_no].append(size)


    plt.figure(figsize=(10,5))
    plt.scatter(x, y, s_low,    c="brown",      label="Low Bandwidth Client")
    plt.scatter(x, y, s_medium, c="orange",     label="Medium Bandwidth Client")
    plt.scatter(x, y, s_high,   c="limegreen",  label="High Bandwidth Client")
    plt.legend(loc="upper center", ncol=3)
    plt.ylim(0.5,9.5)
    plt.yticks(np.arange(1,member_per_group+1))
    plt.xlabel("Group No (Pruned Model No)")
    plt.ylabel("Clients")
    plt.show()

def vis_group_dist(df):

    class_dist = []
    for group_name, group_df in df.groupby("group"):
        class_dist.append(np.sum(group_df.distribution.tolist(),axis=0))
    
    class_dist = np.array(class_dist)

    x = np.array(np.meshgrid(np.arange(1,8),np.arange(NO_CLASSES))).T.reshape(-1,2)[:,0]
    y = np.array(np.meshgrid(np.arange(1,8),np.arange(NO_CLASSES))).T.reshape(-1,2)[:,1]
    s = class_dist.reshape(1,-1)[0]
    s = ((s - np.amin(s))/(np.amax(s) - np.amin(s))*600).astype("int")
    plt.scatter(x,y,s)
    plt.yticks(np.arange(10))
    plt.xlabel("Group ID")
    plt.ylabel("Class")
    plt.show()

def EMD(Z_i, Z_global):
    magnitude = lambda vector: math.sqrt(sum(pow(element, 2) for element in vector))
    return magnitude(Z_i/magnitude(Z_i) - Z_global/magnitude(Z_global))

# def spread_bw_random(main_label_vectors, no_clients, no_groups):
#     NO_GROUPS = no_groups
#     NO_CLIENTS = no_clients
#     weak    = 2_500_000
#     normal = 35_000_000
#     strong  = 80_000_000


#     low_count = int(10/56*NO_CLIENTS)
#     high_count = int(10/56*NO_CLIENTS)
#     medium_count = NO_CLIENTS - low_count - high_count

#     bw_types = (["low"]*low_count) + (["medium"]*medium_count) + (["high"]*high_count)


#     lows = [weak] * low_count
#     mediums = [normal] * medium_count
#     highs = [strong] * high_count
#     bw = lows + mediums + highs

#     ids = np.arange(NO_CLIENTS)
#     random.shuffle(ids)

#     bws = []
#     for idx in ids:
#         bws.append((bw[idx],bw_types[idx]))

#     clients_list = []
#     for idx in range(len(main_label_vectors)):
#         clients_list.append(
#             {
#                 "client_id" : idx,
#                 "bw" : bws[idx][0],
#                 "bw_type" : bws[idx][1],
#                 "distribution" : main_label_vectors[idx],
#                 "group" : None
#             }
#         )

#     clients_df = pd.DataFrame(clients_list)

#     maindf = clients_df.sort_values("bw")
#     maindf["group"] = np.repeat(np.arange(7),int(NO_CLIENTS / NO_GROUPS))


#     for idx,row in maindf.iterrows():
#         maindf.at[idx,"group"] = replace_dict[row.group]
#         # maindf.at[idx,"distribution"] = np.array([float(item) for item in row.distribution.replace("[","").replace("]","").split(" ") if item != ""])
        
#     maindf.reset_index(drop=True)

#     return maindf

def spread_bw_random(main_label_vectors, no_clients, no_groups, tiers):
    NO_GROUPS = no_groups
    NO_CLIENTS = no_clients
    
    bw_types = []
    bw = []
    for k, v in tiers.items():
        for i in range(int(NO_CLIENTS / len(tiers))):
            bw_types.append(k)
            bw.append(random.choice(tiers[k]))

    # print(int(NO_CLIENTS / len(tiers)))
    # print(len(bw))
    # print(len(bw_types))

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
                "distribution" : main_label_vectors[idx],
                "group" : None
            }
        )

    clients_df = pd.DataFrame(clients_list)

    maindf = clients_df.sort_values("bw")
    maindf["group"] = np.repeat(np.arange(7),int(NO_CLIENTS / NO_GROUPS))


    for idx,row in maindf.iterrows():
        maindf.at[idx,"group"] = replace_dict[row.group]
        
    maindf.reset_index(drop=True)

    return maindf

def transfer_op(df_iter, candidate_operations, selected_idx):
    selected_operation = candidate_operations[selected_idx]

    df_dict = df_iter.to_dict(orient="index")
    main_group = df_dict[selected_operation[0]]["group"]
    other_group = df_dict[selected_operation[1]]["group"]

    df_iter.at[selected_operation[0], "group"] = other_group
    df_iter.at[selected_operation[1], "group"] = main_group

    return df_iter

def average_EMD(df_iter, global_dist):
    EMDs = []
    for group_name, group_df in df_iter.groupby("group"):
        EMDs.append(EMD(np.sum(group_df.distribution.values, axis=0), global_dist))
    return np.mean(EMDs)

def NetDag(maindf, no_groups, coeff, max_iter, viz=False):
    global_dist = np.sum(maindf.distribution.values, axis=0)
    df_iter = maindf.copy()

    COEFF = coeff
    NO_GROUPS = no_groups
    down_durs = []
    all_scores = []
    scores = []
    emds = []
    for group_no in range(NO_GROUPS,0,-1):
        group_selected = df_iter[df_iter["group"] == group_no]
        group_dist = np.sum(group_selected.distribution.values, axis=0)
        scores.append(EMD(group_dist, global_dist))
    all_scores.append(scores)


    seq_of_move = np.arange(1,NO_GROUPS+1).tolist()
    total_no_operations = 0
    possible_ops = 0
    candidate_ops = 0
    for iter_no in range(max_iter):
        no_operations_done = 0
        # for group_no in range(NO_GROUPS,0,-1):
        # seq_of_move = sorted(np.arange(1,NO_GROUPS+1), key=lambda k: random.random())
        seq_of_move = seq_of_move[1:] + seq_of_move[: 1]
        # print(seq_of_move)
        for group_no in seq_of_move:
            current_emd = average_EMD(df_iter, global_dist)
            group_selected = df_iter[df_iter["group"] == group_no]
            
            group_friend_next = df_iter[df_iter["group"] == group_no-1]
            group_friend_prev = df_iter[df_iter["group"] == group_no+1]

            group_friend_next_2 = df_iter[df_iter["group"] == group_no-2]
            group_friend_prev_2 = df_iter[df_iter["group"] == group_no+2]

            candidate_groups = [
                group_friend_next,
                group_friend_prev,
                group_friend_next_2,
                group_friend_prev_2
                ]

            group_dist = np.sum(group_selected.distribution.to_numpy(), axis=0)
            score = EMD(group_dist, global_dist)
            candidate_operations = []
            bw_debug = []
            for idx, row in group_selected.iterrows():
                client_id = row.client_id
                group_dist_wo = group_dist - np.array(row.distribution)
                bw_type = row.bw_type

                for candidate_group in candidate_groups:
                    for idx2, row2 in candidate_group.iterrows():
                        possible_ops += 1
                        candidate_dist = group_dist_wo + np.array(row2.distribution)
                        candidate_score = EMD(candidate_dist ,global_dist)
                        bw_type_candidate = row2.bw_type
                        if bw_type == bw_type_candidate: coeff = 1
                        else:
                            # coeff = COEFF**(iter_no+1)
                            coeff = COEFF

                        if candidate_score < score*coeff:
                            candidate_ops += 1
                            df_iter_temp = transfer_op(df_iter.copy(), [(client_id, row2.client_id, candidate_score)], -1)
                            avg_emd = average_EMD(df_iter_temp, global_dist)
                            emds.append(avg_emd)

                            # candidate_operations.append((client_id, row2.client_id, candidate_score))
                            candidate_operations.append((client_id, row2.client_id, avg_emd))
                            bw_debug.append((bw_type, bw_type_candidate))


            # Transfer operation            
            candidate_operations = np.array(candidate_operations)
            if len(candidate_operations) > 0:
                selected_idx = np.argmin(candidate_operations[:,2])
                if current_emd > candidate_operations[selected_idx,2]:
                    # print(candidate_operations[selected_idx], bw_debug[selected_idx])
                    df_iter = transfer_op(df_iter, candidate_operations, selected_idx)
                    no_operations_done += 1

        scores = []
        for group_no in range(NO_GROUPS,0,-1):
            group_selected = df_iter[df_iter["group"] == group_no]
            group_dist = np.sum(group_selected.distribution.to_numpy(), axis=0)
            scores.append(EMD(group_dist, global_dist))
        all_scores.append(scores)

        down_durs.append(download_duration(df_iter))

        total_no_operations += no_operations_done
        if viz: print(f"## ITER {iter_no} | No operations done : {no_operations_done}")
        if no_operations_done == 0:
            break

    all_scores = np.array(all_scores)



    if viz:
        vis_group_dist(df_iter)
        vis_bw_counts(df_iter, len(maindf), NO_GROUPS)

        ### EMD CHANGE
        ax = plt.subplots(figsize=(10,5))
        for group_no in range(NO_GROUPS):
            plt.plot(np.arange(1,len(all_scores)+1), all_scores[:,group_no], label=f"Group {group_no+1}", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("EMD")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,3))
        plt.boxplot(all_scores.T)
        plt.xlabel("Iteration")
        plt.ylabel("EMD")
        plt.show()
    
    return df_iter, all_scores, iter_no, down_durs, total_no_operations, possible_ops, candidate_ops

def download_duration(df_final):
    model_sizes = {
        1: 227108578,
        2: 226817122,
        3: 226559010,
        4: 226595938,
        5: 198579234,
        6: 172803362,
        7: 176866594,
    }

    bw_df = df_final[["bw","group"]].groupby("group").min()
    bw_df["mb"] = model_sizes.values()

    return np.amax(bw_df["mb"] / (bw_df["bw"] / 8))

def main(no_clients, no_groups, alpha, coeff, max_iter, no_classes=10, no_tests=20, viz=False):
    NO_CLIENTS = no_clients
    NO_GROUPS = no_groups
    NO_CLASSES = no_classes
    if isinstance(alpha, float): ALPHA = f"{alpha}".replace(".","")
    elif isinstance(alpha, str): ALPHA = alpha
    COEFF = coeff
    MAX_ITER = max_iter

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

    path = f"./data/alpha/Cifar10_NIID_{NO_CLIENTS}c_a{ALPHA}/config.json"
    conf = json.loads(open(path, "r").read())
    data = [dict(zip(np.array(cli)[:,0], np.array(cli)[:,1])) for cli in conf["Size of samples for labels in clients"]]

    main_label_vectors = np.zeros((NO_CLIENTS,NO_CLASSES))
    for client_id in range(NO_CLIENTS):
        for class_id in range(NO_CLASSES):
            if class_id in data[client_id].keys():
                main_label_vectors[client_id][class_id] = data[client_id][class_id]

    final_scores = []
    durations = []
    for i in range(no_tests):
        s = time()
        # maindf = spread_bw_random(main_label_vectors, NO_CLIENTS, NO_GROUPS)
        maindf = spread_bw_random(main_label_vectors, NO_CLIENTS, NO_GROUPS, tiers)
        # print(maindf)
        df_final, all_scores, iter_no, down_durs, total_no_operations, possible_ops, candidate_ops = NetDag(maindf, NO_GROUPS, COEFF, MAX_ITER, viz=viz)
        final_scores.append(np.mean(all_scores[-1]))
        durations.append(time() - s)

    return df_final, all_scores, final_scores, iter_no, down_durs, np.mean(durations), total_no_operations, possible_ops, candidate_ops

def process(
    no_clients,
    no_groups,
    alpha,
    coeff,
    max_iter,
    no_tests
):
    try:
        df_final, all_scores, final_scores, iter_no, down_durs, avg_elapsed, total_no_operations, possible_ops, candidate_ops = main(
            no_clients=no_clients,
            no_groups=7,
            alpha=alpha,
            coeff=coeff,
            max_iter=max_iter,
            no_tests=1
        )
        avgEMD = np.mean(all_scores[-1])
        lastDur = down_durs[-1]

        log = {
            "alpha": alpha,
            "no_clients" : no_clients,
            "coeff": coeff,
            "max_iter": max_iter,
            "initial_emd": np.mean(all_scores[0]),
            "emd": avgEMD,
            "final_iter": iter_no,
            "total_no_operations": total_no_operations,
            "possible_ops": possible_ops,
            "candidate_ops": candidate_ops,
            "download_duration": lastDur,
            "avg_elapsed_time": avg_elapsed
        }

        print((no_clients, alpha, coeff, max_iter))

        return log

    except Exception as e:
        print(e)

        log = {
            "alpha": alpha,
            "no_clients" : no_clients,
            "coeff": coeff,
            "max_iter": max_iter,
            "initial_emd": None,
            "emd": None,
            "final_iter": None,
            "total_no_operations": None,
            "possible_ops": None,
            "candidate_ops": None,
            "download_duration": None,
            "avg_elapsed_time": None
        }

        return log

if __name__ == "__main__":

    logs = []
    no_tests = 20

    inputs = []
    # for no_clients in [56, 70, 140, 210]:
    for no_clients in np.arange(56,217,14):
        # for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for alpha in np.arange(0.1,1.1,0.1):
            alpha = round(alpha,1)
            # for coeff in [0.1,0.3,0.5,0.7,0.8,0.9,1]:
            for coeff in np.arange(0.1,1.1,0.1):
                for max_iter in [5,6,7,8,9,10,15,20]:
                    for test in range(no_tests):

                        inputs.append((no_clients, 7, alpha, coeff, max_iter, 1))


                        
                        # print(no_clients, alpha, coeff, max_iter, test)
                        
                        # try:
                        #     df_final, all_scores, final_scores, iter_no, down_durs, avg_elapsed, total_no_operations = main(
                        #         no_clients=no_clients,
                        #         no_groups=7,
                        #         alpha=alpha,
                        #         coeff=coeff,
                        #         max_iter=max_iter,
                        #         no_tests=1
                        #     )
                        #     avgEMD = np.mean(all_scores[-1])
                        #     lastDur = down_durs[-1]

                        #     log = {
                        #         "alpha": alpha,
                        #         "no_clients" : no_clients,
                        #         "coeff": coeff,
                        #         "max_iter": max_iter,
                        #         "initial_emd": np.mean(all_scores[0]),
                        #         "emd": avgEMD,
                        #         "final_iter": iter_no,
                        #         "total_no_operations": total_no_operations,
                        #         "download_duration": lastDur,
                        #         "avg_elapsed_time": avg_elapsed
                        #     }
                        #     logs.append(log)

                        #     log_df = pd.DataFrame(logs)
                        #     log_df.to_csv("./data/alpha/test_3.csv")
                            
                        # except Exception as e:
                        #     print(e)

    print(len(inputs))

    pool = Pool(processes=64)
    logs = pool.starmap(process, inputs)
    log_df = pd.DataFrame(logs)
    log_df.to_csv("./data/alpha/test_10.csv")