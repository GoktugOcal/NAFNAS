python master_pure.py \
    projects/netadapt/netadapt_1 \
    3 32 32 \
    -im projects/define_pretrained_fed_sim_NIID_alpha03/last_model.pth.tar -gp 0 \
    -mi 10 -bur 0.25 -rt FLOPS -irr 0.05 -rd 0.96 \
    -lr 0.001 -st 0 \
    -dp data/Cifar10/server --arch alexnet \
    -nc 0