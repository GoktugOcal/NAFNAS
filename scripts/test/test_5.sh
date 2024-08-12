python test_master.py \
    models/alexnet/test/test-5 \
    3 224 224 \
    -im models/alexnet/model_cpu.pth.tar -gp 0 \
    -mi 10 -bur 0.25 -rt FLOPS -irr 0.05 -rd 0.96 \
    -lr 0.001 -st 500 \
    -dp data/Cifar10/server --arch alexnet \
    -nc 0