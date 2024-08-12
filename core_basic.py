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
# from fl_src import config

logging.basicConfig(level=logging.DEBUG)

TEST_NAME = "test-cpu-10nc-mac-zzzz"

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
        # options = DockerNode.create_options()
        # options.image = "pynode"
        # options.binds = [(os.getcwd() + "/models", "/app/models")]
        # options.no_cpus = 32
        servernode = session.add_node(CoreNode)
        iface1_data = prefixes.create_iface(servernode)
        session.add_link(servernode.id, switch.id, iface1_data)


        # Client Nodes
        clients = []
        client_ifaces = []
        for i in range(10):
            new_client = session.add_node(CoreNode)
            clients.append(new_client)
            client_ifaces.append(prefixes.create_iface(new_client))

        # add links
        # link_options = LinkOptions(bandwidth=55_000_000)
        for client, iface in zip(clients, client_ifaces):
            session.add_link(client.id, switch.id, iface)

        # =========== instantiate ===========
        session.instantiate()

        # Execute python scripts

        # # Latency
        # servernode.host_cmd(
        #     f"/home/goktug/python_envs/easyfl/bin/python3 /home/goktug/Desktop/thesis/netadapt-x-flower/federated_master.py "
        #     f"models/alexnet/fed/{TEST_NAME} "
        #     f"3 224 224 "
        #     f"-im models/alexnet/model_cpu.pth.tar -gp 0 "
        #     f"-mi 10 -bur 0.25 -rt LATENCY -irr 0.025 -rd 0.96 "
        #     f"-lr 0.001 -st 500 -lt latency_lut/lut_alexnet_pt21.pkl "
        #     f"-dp data/Cifar10/server --arch alexnet "
        #     f"-nc 10",
        #     shell=False, wait=False
        # )

        # MAC
        servernode.host_cmd(
            # f"cd /home/goktug/Desktop/thesis/netadapt-x-flower/ ; "
            f"/home/goktug/python_envs/easyfl/bin/python3 federated_master.py "
            f"models/alexnet/fed/{TEST_NAME} "
            f"3 224 224 "
            f"-im models/alexnet/model_cpu.pth.tar -gp 0 "
            f"-mi 10 -bur 0.25 -rt FLOPS -irr 0.025 -rd 0.96 "
            f"-lr 0.001 -st 500 "
            f"-dp data/Cifar10/server --arch alexnet "
            f"-nc 10",
            shell=False, wait=False
        )
        time.sleep(5)  # Give server enough time to start

        for i, client in enumerate(clients[:-1]):
            # CoreNode starts from 3 because 1 is switch and 2 is server
            client.host_cmd(
                # f"cd /home/goktug/Desktop/thesis/netadapt-x-flower/ ; "
                f"/home/goktug/python_envs/easyfl/bin/python3 client.py "
                f"models/alexnet/fed/{TEST_NAME} "
                f"--server_ip {iface1_data.ip4} "
                f"--no {i}",
                shell=False, wait=False
            )
        
        clients[-1].host_cmd(
            # f"cd /home/goktug/Desktop/thesis/netadapt-x-flower/ ; "
            f"/home/goktug/python_envs/easyfl/bin/python3 client.py "
            f"models/alexnet/fed/{TEST_NAME} "
            f"--server_ip {iface1_data.ip4} "
            f"--no {len(clients) - 1}",
            wait=True,
        )

        input("press enter to shutdown")


    except KeyboardInterrupt:
        coreemu.shutdown()
        
    finally:
        coreemu.shutdown()

if __name__ == "__main__":
    main()
    print("DONE.")