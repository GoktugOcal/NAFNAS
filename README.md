# netadapt-x-flower
A federated NAS study

### Build Docker
```bash
docker build -t pynode -f Dockerfile.python .
```

```bash
sudo iptables --policy FORWARD ACCEPT
```

### Basic Run with Binding
```bash
docker run -it --name gocker -v /home/goktug/Desktop/thesis/netadapt-fl-docker/models/:/app/models/ pynode /bin/bash
```

### Build and run FL-NETADAPT
```bash
sudop python core_up.py
```

```bash
source /home/goktug/python_envs/easyfl/bin/activate; cd /home/goktug/Desktop/thesis/netadapt-x-flower/
```


nohup bash scripts/pure_core/netadapt_56c_fedprox_only_iter1_10.sh > logs/netadapt_56c_fedprox_only_iter1_10.sh &


projects/netadapt/netadapt_1/worker/iter_4_block_4_model_simp.pth.tar

docker run --rm -it --cpus=4 --name gocker -v /home/goktug.ocal/thesis/netadapt-x-flower/logs/:/app/logs/ -v /home/goktug.ocal/thesis/netadapt-x-flower/data/:/app/data/ -v /home/goktug.ocal/thesis/netadapt-x-flower/projects/:/app/projects/ pynode /bin/bash

python pure_server.py projects/pure_core/netadapt_1/netadapt_1_iter_1/block_0/ -nc 8 -m /home/goktug.ocal/thesis/netadapt-x-flower/projects/netadapt/netadapt_1/worker/iter_1_block_0_model_simp.pth.tar -nr 1

python pure_server.py projects/trial/ -nc 8 -m projects/define_pretrained_fed_sim_NIID_alpha03/last_model.pth.tar -nr 1

projects/define_pretrained_fed_sim_NIID_alpha03/last_model.pth.tar

docker run --rm -it \
    -v /home/goktug/Desktop/thesis/netadapt-x-flower/models/:/app/models/ \
    -v /home/goktug/Desktop/thesis/netadapt-x-flower/data/:/app/data/ \
    -v /home/goktug/Desktop/thesis/netadapt-x-flower/projects/:/app/projects/ \
    pynode \
    /bin/bash


docker run -d --rm --cpus=4 --name gocker -v /home/goktug.ocal/thesis/netadapt-x-flower/logs/:/app/logs/ -v /home/goktug.ocal/thesis/netadapt-x-flower/data/:/app/data/ pynode python load_test.py


sudop nohup python pure_core.py > logs/core_NIID_a1_20c_alexnet.log &

srun --immediate --time=0 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --container-name=d_sample0 --container-image=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 --pty bash

docker exec -td DockerServer2 python pure_server.py projects/32_pure_test_NIID_a01_20c_alexnet_3 -nc 20 -m projects/define_pretrained_fed_sim_NIID_alpha03/alexnet.pth.tar