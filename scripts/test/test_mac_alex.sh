python test_master.py \
    projects/test/test-1-model-size \
    3 224 224 \
    -im projects/define_pretrained_fed_sim_NIID_alpha03/alexnet.pth.tar -gp 0 \
    -mi 15 -bur 0.25 -rt FLOPS -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 \
    -dp data/Cifar10/server --arch alexnet \
    -nc 0