python federated_simulation.py \
    -pf ./projects/federated_simulation/test_2_iter_1/ \
    --data data/32_Cifar10_NIID_56c_a03 \
    --global_model_path projects/test/test-2/worker/iter_1_block_0_model.pth.tar \
    -nc 56 \
    -nr 20 \
    --fine_tuning_epochs 50