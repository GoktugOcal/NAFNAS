model_paths=(
    "./projects/netadapt/netadapt_1/worker/iter_10_block_0_model_simp.pth.tar"
    "./projects/netadapt/netadapt_1/worker/iter_10_block_1_model_simp.pth.tar"
    "./projects/netadapt/netadapt_1/worker/iter_10_block_2_model_simp.pth.tar"
    "./projects/netadapt/netadapt_1/worker/iter_10_block_3_model_simp.pth.tar"
    "./projects/netadapt/netadapt_1/worker/iter_10_block_4_model_simp.pth.tar"
    "./projects/netadapt/netadapt_1/worker/iter_10_block_5_model_simp.pth.tar"
    "./projects/netadapt/netadapt_1/worker/iter_10_block_6_model_simp.pth.tar"
    )

python model_client_assignment.py \
    projects/fed_master/fed_master_cifar10_a03_56c_network_opt_small \
    -ms "${model_paths[@]}" \
    -bws ./data/32_Cifar10_NIID_56c_a03/bws_groups.csv