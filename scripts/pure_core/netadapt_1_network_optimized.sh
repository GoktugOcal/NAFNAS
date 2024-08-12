#!/bin/bash

model_names=("iter_1_block_0_model_simp.pth.tar"
"iter_1_block_1_model_simp.pth.tar"
"iter_1_block_2_model_simp.pth.tar"
"iter_1_block_3_model_simp.pth.tar"
"iter_1_block_4_model_simp.pth.tar"
"iter_1_block_5_model_simp.pth.tar"
"iter_1_block_6_model_simp.pth.tar")

worker_path="./projects/netadapt/netadapt_1/worker/"
dataset_path="32_Cifar10_NIID_80c_a03"
working_path="projects/pure_core/netadapt_1_network_optimized/netadapt_1_iter_1/"

length=${#model_names[@]}

for ((i=0; i<$length; i++)); do
# for ((i=0; i<1; i++)); do
  model_path="${worker_path}${model_names[$i]}"
  current_path="${working_path}block_${i}/"

  echo "Model path: ${model_path}"
  echo "Working path: ${current_path}"

  python pure_core.py -wp "${current_path}" \
    -mp "${model_path}" \
    -dp "${dataset_path}" \
    -nc 8 \
    -ng 7 \
    -nr 1 \
    -cpc 8 \
    -mpc "6g"
done
