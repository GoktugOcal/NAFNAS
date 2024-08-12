from argparse import ArgumentParser

import logging
import time
import os
import argparse
import numpy as np
import random
import json

from core.emulator.coreemu import CoreEmu
from core.emulator.data import IpPrefixes
from core.emulator.enumerations import EventTypes
from core.nodes.base import CoreNode
from core.nodes.docker import DockerNode
from core.nodes.network import SwitchNode
from core.emulator.data import LinkOptions

from custom_nodes.dockerserver import DockerServer
from custom_nodes.dockerclient import DockerClient
# from fl_src import config

logging.basicConfig(level=logging.DEBUG)

user = os.getlogin()
if user == "goktug":
    #MODEL_PATH = "projects/define_pretrained_fed_sim_NIID_alpha03/alexnet.pth.tar"
    MODEL_PATH = "projects/test/test-5/worker/iter_1_block_0_model.pth.tar"
    DATASET_PATH = "32_Cifar10_NIID_56c_a03"
else:
    MODEL_PATH = "projects/test/test-5/master/iter_10_best_model.pth.tar"
    DATASET_PATH = "32_Cifar10_NIID_80c_a03"

#WORKING_PATH = "projects/12dec/network_test_8c_iter_10/"
MAX_ITER = 5
NO_CLIENTS = 8
NO_ROUNDS = 1
CPU_PER_CLIENT = 8
MEM_LIMIT_PER_CLIENT = "6g"

WEAK_NETWORK = np.array([2_500_000, 3_500_000])
NORMAL_NETWORK = np.arange(8_000_000, 32_000_000, 1_000_000)
STRONG_NETWORK = np.arange(50_000_000, 100_000_000, 10_000_000)

TIER_11 = np.array([2_500_000, 3_500_000, 500_000])
TIER_12 = np.array([4_000_000, 5_500_000, 500_000])
TIER_13 = np.array([6_000_000, 8_000_000, 500_000])
TIER_21 = np.array([8_500_000, 16_500_000, 1_000_000])
TIER_22 = np.array([17_000_000, 25_000_000, 1_000_000])
TIER_23 = np.array([25_500_000, 32_500_000, 1_000_000])
TIER_31 = np.array([35_000_000, 55_000_000, 1_000_000])
TIER_32 = np.array([56_000_000, 75_000_000, 1_000_000])
TIER_33 = np.array([75_000_000, 100_000_000, 1_000_000])


client_networks = {
    0 : WEAK_NETWORK,
    1 : NORMAL_NETWORK,
    2 : STRONG_NETWORK,
    3 : WEAK_NETWORK,
    4 : NORMAL_NETWORK,
    5 : STRONG_NETWORK,
    6 : WEAK_NETWORK,
    7 : NORMAL_NETWORK,
    8 : STRONG_NETWORK,
    9 : WEAK_NETWORK,
    10 : NORMAL_NETWORK,
    11 : NORMAL_NETWORK,
    12 : STRONG_NETWORK,
    13 : WEAK_NETWORK,
    14 : NORMAL_NETWORK,
    15 : STRONG_NETWORK,
    16 : WEAK_NETWORK,
    17 : NORMAL_NETWORK,
    18 : STRONG_NETWORK,
    19 : WEAK_NETWORK,
}

client_networks_high = {
    0 : STRONG_NETWORK,
    1 : STRONG_NETWORK,
    2 : STRONG_NETWORK,
    3 : STRONG_NETWORK,
    4 : STRONG_NETWORK,
    5 : STRONG_NETWORK,
    6 : STRONG_NETWORK,
    7 : STRONG_NETWORK,
}

client_networks_medium = {
    0 : NORMAL_NETWORK,
    1 : NORMAL_NETWORK,
    2 : NORMAL_NETWORK,
    3 : NORMAL_NETWORK,
    4 : NORMAL_NETWORK,
    5 : NORMAL_NETWORK,
    6 : NORMAL_NETWORK,
    7 : NORMAL_NETWORK,
}

client_networks_low = {
    0 : WEAK_NETWORK,
    1 : WEAK_NETWORK,
    2 : WEAK_NETWORK,
    3 : WEAK_NETWORK,
    4 : WEAK_NETWORK,
    5 : WEAK_NETWORK,
    6 : WEAK_NETWORK,
    7 : WEAK_NETWORK,
}

# client_networks_all = {
#     "block_0" : {
#         0 : STRONG_NETWORK,   1 : STRONG_NETWORK,   2 : STRONG_NETWORK,   3 : STRONG_NETWORK,   4 : STRONG_NETWORK,   5 : STRONG_NETWORK,   6 : NORMAL_NETWORK,   7 : NORMAL_NETWORK,
#     },
#     "block_1" : {
#         0 : STRONG_NETWORK,   1 : NORMAL_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : NORMAL_NETWORK,   7 : NORMAL_NETWORK,
#     },
#     "block_2" : {
#         0 : STRONG_NETWORK,   1 : STRONG_NETWORK,   2 : STRONG_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : NORMAL_NETWORK,   7 : NORMAL_NETWORK,
#     },
#     "block_3" : {
#         0 : NORMAL_NETWORK,   1 : NORMAL_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : NORMAL_NETWORK,   7 : NORMAL_NETWORK,
#     },
#     "block_4" : {
#         0 : WEAK_NETWORK,   1 : WEAK_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : NORMAL_NETWORK,   7 : NORMAL_NETWORK,
#     },
#     "block_5" : {
#         0 : WEAK_NETWORK,   1 : WEAK_NETWORK,   2 : WEAK_NETWORK,   3 : WEAK_NETWORK,   4 : WEAK_NETWORK,   5 : WEAK_NETWORK,   6 : NORMAL_NETWORK,   7 : NORMAL_NETWORK,
#     },
#     "block_6" : {
#         0 : WEAK_NETWORK,   1 : WEAK_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : NORMAL_NETWORK,   7 : NORMAL_NETWORK,
#     },
# }

client_networks_all = {
    "block_0" : {
        0 : STRONG_NETWORK,   1 : STRONG_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : WEAK_NETWORK,   6 : WEAK_NETWORK,   7 : WEAK_NETWORK,
    },
    "block_1" : {
        0 : STRONG_NETWORK,   1 : NORMAL_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : WEAK_NETWORK,   7 : WEAK_NETWORK,
    },
    "block_2" : {
        0 : STRONG_NETWORK,   1 : STRONG_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : WEAK_NETWORK,   7 : WEAK_NETWORK,
    },
    "block_3" : {
        0 : STRONG_NETWORK,   1 : NORMAL_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : WEAK_NETWORK,   7 : WEAK_NETWORK,
    },
    "block_4" : {
        0 : STRONG_NETWORK,   1 : NORMAL_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : WEAK_NETWORK,   7 : WEAK_NETWORK,
    },
    "block_5" : {
        0 : STRONG_NETWORK,   1 : NORMAL_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : WEAK_NETWORK,   7 : WEAK_NETWORK,
    },
    "block_6" : {
        0 : STRONG_NETWORK,   1 : NORMAL_NETWORK,   2 : NORMAL_NETWORK,   3 : NORMAL_NETWORK,   4 : NORMAL_NETWORK,   5 : NORMAL_NETWORK,   6 : WEAK_NETWORK,   7 : WEAK_NETWORK,
    },
}

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


def main(args):

    if args.working_path: WORKING_PATH = args.working_path
    if args.model_path: MODEL_PATH = args.model_path
    if args.dataset_path: DATASET_PATH = args.dataset_path
    if args.cpu_per_client: CPU_PER_CLIENT = args.cpu_per_client
    if args.mem_per_client: MEM_LIMIT_PER_CLIENT = args.mem_per_client
    if args.no_rounds: NO_ROUNDS = args.no_rounds
    if args.no_clients: NO_CLIENTS = args.no_clients

    # if any(block_id in MODEL_PATH for block_id in ["block_0","block_1"]): 
    #     client_networks =  client_networks_high
    # elif any(block_id in MODEL_PATH for block_id in ["block_2","block_3","block_4"]):
    #     client_networks =  client_networks_medium
    # elif any(block_id in MODEL_PATH for block_id in ["block_5","block_6"]):
    #     client_networks =  client_networks_low

    block_id = [block_id for block_id in ["block_0","block_1","block_2","block_3","block_4","block_5","block_6"] if block_id in MODEL_PATH][0]    
    # client_networks = client_networks_all[block_id]

    # client_networks_all = {}
    # with open(os.path.join("./data",DATASET_PATH,"client_groups.json")) as json_file:
    #     bw_data = json.load(json_file)
    # for cid, tier in bw_data[block_id].items():
    #     print(cid, tier)
    #     client_networks_all[cid] = tiers[tier]
    client_networks = client_networks_all


    client_networks = client_networks_all[block_id]
    client_networks = {str(k): v for k,v in client_networks.items()}



    os.makedirs(WORKING_PATH, exist_ok=True)

    config_networks = client_networks.copy()
    for key, value in config_networks.items():
        config_networks[key] = value.tolist()
    config_dict = vars(args)
    config_dict["client_networks"] = config_networks

    json_file_path = os.path.join(WORKING_PATH,"config.json")
    with open(json_file_path, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)

    # create core session
    coreemu = CoreEmu()
    session = coreemu.create_session()
    session.set_state(EventTypes.CONFIGURATION_STATE)

    try:
        prefixes = IpPrefixes(ip4_prefix="10.83.0.0/16")
        # prefixes = IpPrefixes(ip4_prefix="10.0.0.0/24")

        # create switch
        switch = session.add_node(SwitchNode)

        # Server Node
        options = DockerNode.create_options()
        options.image = "pynode"
        options.binds = [
            (os.getcwd() + "/models", "/app/models"),
            (os.getcwd() + "/data", "/app/data"),
            (os.getcwd() + "/projects", "/app/projects")
            ]
        # options.no_cpus = CPU_PER_CLIENT
        # options.mem_limit = MEM_LIMIT_PER_CLIENT
        servernode = session.add_node(DockerServer, options=options)
        iface1_data = prefixes.create_iface(servernode)

        # Client Nodes
        client_options = DockerNode.create_options()
        client_options.image = "pynode"
        client_options.binds = [
            (os.getcwd() + "/models", "/app/models"),
            (os.getcwd() + "/data", "/app/data"),
            (os.getcwd() + "/projects", "/app/projects")
            ]
        client_options.no_cpus = CPU_PER_CLIENT
        client_options.mem_limit = MEM_LIMIT_PER_CLIENT
        clients = []
        client_ifaces = []
        for i in range(NO_CLIENTS):
            new_client = session.add_node(DockerClient, options=client_options)
            clients.append(new_client)
            client_ifaces.append(prefixes.create_iface(new_client))

        # add links
        session.add_link(servernode.id, switch.id, iface1_data)
        # link_options = LinkOptions(bandwidth=55_000_000)
        link_options = None
        if link_options:
            for client, iface in zip(clients, client_ifaces):
                session.add_link(client.id, switch.id, iface, options=link_options)
        else:
            for client, iface in zip(clients, client_ifaces):
                session.add_link(client.id, switch.id, iface)

        # =========== instantiate ===========
        session.instantiate()

        # Execute python scripts
        servernode.host_cmd(
            f"docker exec -td DockerServer2 python pure_server.py "
            f"{WORKING_PATH} "
            f"-nc {NO_CLIENTS} "
            f"-m {MODEL_PATH} "
            f"-nr {NO_ROUNDS}",
            wait=True
        )
        # models/alexnet/alexnet32_a03_server.pth.tar
        
        # time.sleep(5)  # Give server enough time to start

        for i, client in enumerate(clients):
            # DockerClient starts from 3 because 1 is switch and 2 is server
            client.host_cmd(
                f"docker exec -td DockerClient{3 + i} python client.py "
                f"{WORKING_PATH} "
                f"--server_ip {iface1_data.ip4} "
                f"-dn {DATASET_PATH} "
                f"--no {i} "
            )

            # 32_Cifar10_NIID_20c_a03 

        for client in clients:
            new_options = LinkOptions(
                bandwidth=random.choice(client_networks[str(client.id-3)]),
                )
            session.update_link(switch.id, client.id, client.id-2, 0, options=new_options)
        
        time.sleep(10)
        temp_line = ""
        start = time.time()

        # Runtime Operations
        while(True):
            
            # Change BandWidth
            elapsed = time.time() - start
            if elapsed > 300:
                for client in clients:
                    new_options = LinkOptions(
                        bandwidth=random.choice(client_networks[str(client.id-3)]),
                        )
                    session.update_link(switch.id, client.id, client.id-2, 0, options=new_options)
                start = time.time()
                
            # Read Logs
            if not os.path.exists(os.path.join(WORKING_PATH,"logs_debug.txt")):
                time.sleep(30)

            with open(os.path.join(WORKING_PATH,"logs_debug.txt"), "r") as f:
                last_line = f.readlines()[-1]
                # logging.debug(last_line)
                if temp_line != last_line:
                    logging.debug(last_line)
                    temp_line = last_line
                if "DONE" in last_line:
                    raise KeyboardInterrupt


        # servernode.host_cmd(
        #     f"docker exec -td DockerServer2 python pure_server.py "
        #     f"{WORKING_PATH} "
        #     f"-nc {NO_CLIENTS}",
        #     wait=False
        # )


        # Wait for the last client process to end
        # clients[-1].host_cmd(
        #     f"docker exec -t DockerClient{2 + len(clients)} python client.py "
        #     f"{WORKING_PATH} "
        #     f"--server_ip {iface1_data.ip4} "
        #     f"--no {len(clients) - 1} ",
        #     wait=True,
        # )

    except KeyboardInterrupt:
        coreemu.shutdown()
    
    except Exception as e:
        logging.debug(e, exc_info=True)
        print(">>>>>>>>>>>>>>>", e)
        coreemu.shutdown()
        
    finally:
        coreemu.shutdown()

if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-wp", "--working_path", type=str,
                            help="Working path.")
    arg_parser.add_argument("-mp", "--model_path", type=str,
                            help="Path to the model that is used in federated learning.")
    arg_parser.add_argument("-dp", "--dataset_path", type=str,
                            help="Path to the dataset.")
    arg_parser.add_argument("-nc", "--no_clients", type=int,
                            help="Number of clients")
    arg_parser.add_argument("-ng", "--no_groups", type=int,
                            help="Number of groups")
    arg_parser.add_argument("-nr", "--no_rounds", type=int,
                            help="Number of communcation rounds in FL.")                        
    arg_parser.add_argument("-cpc", "--cpu_per_client", type=int,
                            help="CPU per client")
    arg_parser.add_argument("-mpc", "--mem_per_client", type=str,
                            help="Memory per client")
    args = arg_parser.parse_args()

    main(args)
    print("DONE.")