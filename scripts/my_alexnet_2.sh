CUDA_VISIBLE_DEVICES=0 python federated_master.py models/alexnet/fed/prod-1-prune-by-latency 3 224 224 \
    -im models/alexnet/model_pt21.pth.tar -gp 0 \
    -mi 1 -bur 0.25 -rt LATENCY  -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 -lt latency_lut/lut_alexnet_pt21.pkl \
    -dp data/ --arch alexnet