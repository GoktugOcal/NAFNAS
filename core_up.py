import logging
import time
import os
import argparse

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

WORKING_PATH = "models/alexnet/fed/test-test"
MAX_ITER = 5
NO_CLIENTS = 5
CPU_PER_CLIENT = 4
MEM_LIMIT_PER_CLIENT = "4g"

def main():
    parser = argparse.ArgumentParser(description="Core")
    args = parser.parse_args()

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
        options.binds = [(os.getcwd() + "/models", "/app/models")]
        options.no_cpus = CPU_PER_CLIENT
        options.mem_limit = MEM_LIMIT_PER_CLIENT
        servernode = session.add_node(DockerServer, options=options)
        iface1_data = prefixes.create_iface(servernode)

        # Client Nodes
        clients = []
        client_ifaces = []
        for i in range(NO_CLIENTS):
            new_client = session.add_node(DockerClient, options=options)
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
        # # Latency
        # servernode.host_cmd(
        #     f"docker exec -td DockerServer2 python federated_master.py "
        #     f"{WORKING_PATH} "
        #     f"3 224 224 "
        #     f"-im models/alexnet/model_cpu.pth.tar -gp 0 "
        #     f"-mi 10 -bur 0.25 -rt LATENCY  -irr 0.025 -rd 0.96 "
        #     f"-lr 0.001 -st 500 -lt latency_lut/lut_alexnet_pt21.pkl "
        #     f"-dp data/Cifar10/server --arch alexnet "
        #     f"-nc {NO_CLIENTS}"
        # )
        # MAC
        servernode.host_cmd(
            f"docker exec -td DockerServer2 python federated_master.py "
            f"{WORKING_PATH} "
            f"3 224 224 "
            f"-im models/alexnet/model_cpu.pth.tar -gp 0 "
            f"-mi {MAX_ITER} -bur 0.25 -rt FLOPS  -irr 0.025 -rd 0.96 "
            f"-lr 0.001 -st 500 "
            f"-dp data/Cifar10/server --arch alexnet "
            f"-nc {NO_CLIENTS}"
        )
        
        time.sleep(5)  # Give server enough time to start

        for i, client in enumerate(clients[:-1]):
            # DockerClient starts from 3 because 1 is switch and 2 is server
            client.host_cmd(
                f"docker exec -td DockerClient{3 + i} python client.py "
                f"{WORKING_PATH} "
                f"--server_ip {iface1_data.ip4} "
                f"--no {i} "
            )

        # Wait for the last client process to end
        clients[-1].host_cmd(
            f"docker exec -t DockerClient{2 + len(clients)} python client.py "
            f"{WORKING_PATH} "
            f"--server_ip {iface1_data.ip4} "
            f"--no {len(clients) - 1} ",
            wait=True,
        )

    except KeyboardInterrupt:
        coreemu.shutdown()
        
    finally:
        coreemu.shutdown()

if __name__ == "__main__":
    main()
    print("DONE.")