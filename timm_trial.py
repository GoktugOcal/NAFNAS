from dotenv import load_dotenv
load_dotenv()

import timm 
import torch
import network_utils as networkUtils

model = timm.create_model('mobilenetv3_small_100')

# model = torch.load("./models/alexnet/model_cpu.pth.tar")

model_arch = "mobilenet"
input_data_shape = [3,224,224]
dataset_path = "./data/"
network_utils = networkUtils.__dict__[model_arch](model, input_data_shape, dataset_path)
network_def = network_utils.get_network_def_from_model(model)
current_resource = network_utils.compute_resource(network_def, "FLOPS", None)

print("num simp blocks:",network_utils.get_num_simplifiable_blocks())
simplified_network_def, simplified_resource = (
        network_utils.simplify_network_def_based_on_constraint(network_def,
                                                               4,
                                                               56510400.0/2,
                                                               "FLOPS",
                                                               None))  

print(simplified_network_def == network_def)
print(current_resource, simplified_resource)

