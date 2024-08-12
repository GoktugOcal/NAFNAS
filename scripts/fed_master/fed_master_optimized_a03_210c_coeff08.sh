python test_fed_master.py \
    projects/fed_master/fed_master_optimized_2_a03_210c_coeff08\
    3 32 32 \
    -im ./projects/pretrained_model/predefined_a03_56c_fedavg/last_model.pth.tar \
    -c 10 \
    -gp 0 \
    -mi 15 -bur 0.25 -rt FLOPS -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 \
    -dp ./data/alpha/Cifar10_NIID_210c_a03 \
    --arch alexnet \
    -nc 210 \
    -nr 20 \
    --fine_tuning_epochs 50 \
    --client_selection "optimized"