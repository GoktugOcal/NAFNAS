#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --container-mount-home
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=60:00:00
#SBATCH -o fed_master_network_optimized_a03_210c_fedprox.log

source /opt/python3/venv/base/bin/activate
#git clone https://github.com/GoktugOcal/netadapt-x-flower.git
cd /users/goktug.ocal/thesis/netadapt-x-flower
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2
nvidia-smi
sh scripts/fed_master/fed_prox/fed_master_network_optimized_a03_210c_fedprox.sh
