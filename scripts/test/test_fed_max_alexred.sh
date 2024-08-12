python test_fed_master.py \
    projects/test/mac-fed-2\
    3 32 32 \
    -im projects/define_pretrained_fed_sim_NIID_alpha03/alexnet.pth.tar -gp 0 \
    -mi 15 -bur 0.25 -rt FLOPS -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 \
    -dp data/Cifar10/server \
    --arch alexnet \
    -nc 56 \
    -nr 10 \
    --fine_tuning_epochs 20
    # --client_selection