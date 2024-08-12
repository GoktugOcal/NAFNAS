python federated_simulation.py \
    -pf ./projects/federated_simulation/fed_sim_1/ \
    --data data/32_Cifar10_NIID_56c_a03 \
    --global_model_path projects/netadapt/netadapt_1/worker/iter_4_block_4_model_simp.pth.tar \
    -nc 10 \
    -nr 20 \
    --fine_tuning_epochs 20