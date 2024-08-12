#!/bin/bash


for iterno in 1 10 15; do
    model_names=(
        "iter_"${iterno[@]}"_block_0_model_simp.pth.tar"
        "iter_"${iterno[@]}"_block_1_model_simp.pth.tar"
        "iter_"${iterno[@]}"_block_2_model_simp.pth.tar"
        "iter_"${iterno[@]}"_block_3_model_simp.pth.tar"
        "iter_"${iterno[@]}"_block_4_model_simp.pth.tar"
        "iter_"${iterno[@]}"_block_5_model_simp.pth.tar"
        "iter_"${iterno[@]}"_block_6_model_simp.pth.tar")

    worker_path="projects/fed_master/fed_master_a03_56c/worker/"
    dataset_path="32_Cifar10_NIID_56c_a03"
    working_path="projects/fed_master/fed_master_cifar10_a03_56c_network_opt_small/iter_"${iterno[@]}"/"

    length=${#model_names[@]}

    model_paths=()
    for ((i=0; i<${#model_names[@]}; i++)); do
        model_paths[$i]="${worker_path[@]}${model_names[$i]}"
    done

    python model_client_assignment.py \
        "${working_path[@]}" \
        -ms "${model_paths[@]}" \
        -bws ./data/32_Cifar10_NIID_56c_a03/bws_groups.csv

    for model_path in ${model_paths[@]}; do
        model_name="$(basename "${model_path[@]%.*.*}")"

        python fed_train_sim.py \
            -pf "${working_path[@]}${model_name[@]}" \
            -dp ./data/"${dataset_path[@]}" \
            -ft "${model_path[@]}" \
            -m alexnet.pth.tar \
            -dn cifar10 \
            -c 10 \
            -nc 56 \
            -nr 20 \
            --fine_tuning_epochs 50 \
            --epochs 200 \
            --lr 0.001 \
            --arch alexnet_reduced \
            --clients_selected "${working_path[@]}"
    done

    


    
    


done




# for ((i=0; i<$length; i++)); do
# # for ((i=0; i<1; i++)); do
#   model_path="${worker_path}${model_names[$i]}"
#   current_path="${working_path}block_${i}/"

#   echo "Model path: ${model_path}"
#   echo "Working path: ${current_path}"

#   python pure_core.py -wp "${current_path}" \
#     -mp "${model_path}" \
#     -dp "${dataset_path}" \
#     -nc 8 \
#     -ng 7 \
#     -nr 1 \
#     -cpc 8 \
#     -mpc "6g"
# done
