for alpha in 0.1 0.3 0.5 1.0; do
    str_alpha=$(echo "$alpha" | tr -d '.')
    python generate_cifar10.py \
        -dir ./data/alpha/Cifar10_NIID_20c_a${str_alpha}/ \
        -nc 20 \
        -c 10 \
        -niid noniid \
        -b - \
        -p dir \
        -a $alpha
done
