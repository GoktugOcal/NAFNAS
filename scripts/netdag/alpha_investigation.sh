for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for nc in 56 70 84 98 112 126 140 154 168 182 196 210; do
        str_alpha=$(echo "$alpha" | tr -d '.')
        python generate_cifar10.py \
            -dir ./data/alpha/Cifar10_NIID_${nc}c_a${str_alpha}/ \
            -nc $nc \
            -c 10 \
            -niid noniid \
            -b - \
            -p dir \
            -a $alpha
    done
done
