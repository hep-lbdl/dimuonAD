import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-fid", "--flow_id")
parser.add_argument("-f", "--features")
parser.add_argument("-pid", "--project_id")

args = parser.parse_args()


from numba import cuda 

from helpers.make_flow import *
from helpers.train_flow import *
from helpers.plotting import *
from helpers.evaluation import *

seed = 8

# computing

device = cuda.get_current_device()
device.reset()

# set the number of threads that pytorch will use
torch.set_num_threads(2)

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)


"""
LOAD IN DATA
"""


bands = ["SBL", "SR", "SBH"]
working_dir = "/global/u1/r/rmastand/dimuonAD/projects/standardscaler_0814/"


feature_set = [f for f in args.features.split(" ")]
print(f"Using feature set {feature_set}")

num_features = len(feature_set) - 1 # context doesn't count


with open(f"{working_dir}/processed_data/{args.project_id}_train_band_data", "rb") as ifile:
    proc_dict_s_inj = pickle.load(ifile)
    
data_dict = {}

for b in bands:
    
    num_events_band = proc_dict_s_inj[b]["s_inj_data"]["dimu_mass"].shape[0]
    
    
    data_dict[b] = np.empty((num_events_band, num_features+1))
    for i, feat in enumerate(feature_set):
        data_dict[b][:,i] = proc_dict_s_inj[b]["s_inj_data"][feat].reshape(-1,)
    print("{b} data has shape {length}.".format(b = b, length = data_dict[b].shape))

    
data_dict["SB"] =  np.vstack((data_dict["SBL"], data_dict["SBH"]))
#data_dict["IB"] =  np.vstack((data_dict["IBL"], data_dict["IBH"]))

# train val test split
from sklearn.model_selection import train_test_split

SBL_data_train, SBL_data_val = train_test_split(data_dict["SBL"], test_size=0.2, random_state=42)
SBH_data_train, SBH_data_val = train_test_split(data_dict["SBH"], test_size=0.2, random_state=42)


print(f"SBL train data has shape {SBL_data_train.shape}.")
print(f"SBL val data has shape {SBL_data_val.shape}.")
print(f"SBH train data has shape {SBH_data_train.shape}.")
print(f"SBH val data has shape {SBH_data_val.shape}.")



"""
CREATE THE FLOW
"""

# power scaler
num_layers = 2
num_hidden_features = 32
num_blocks = 4
early_stop_patience = 20


hyperparameters_dict = {"n_epochs":200,
                          "batch_size": 128,
                          "lr": 0.00025,
                          "weight_decay": 0.00}


# standard scaler
num_layers = 2
num_hidden_features = 32
num_blocks = 8
early_stop_patience = 20


hyperparameters_dict = {"n_epochs":300,
                          "batch_size": 128,
                          "lr": 0.0002,
                          "weight_decay": 0.00}


#flow_training_id = f"Masked_PRQ_AR_{num_layers}layers_{num_hidden_features}hidden_{num_blocks}blocks_{seed}seed"
flow_training_id = f"{args.project_id}_{args.flow_id}"

flow_training_dir = os.path.join(f"{working_dir}/models", f"{flow_training_id}/")
os.makedirs(flow_training_dir, exist_ok=True)


test_flow = make_masked_AR_flow(num_features, num_layers, num_hidden_features, num_blocks)

pytorch_total_params = sum(p.numel() for p in test_flow.parameters() if p.requires_grad)
print(f"Numb. trainable params: {pytorch_total_params}")



with open(f"{flow_training_dir}/configs.txt", "w") as param_file:
    param_file.write(f"feature_set = {feature_set}\n")
    param_file.write(f"project_id = {args.project_id}\n")
    param_file.write(f"flow_training_id = {flow_training_id}")

"""
TRAIN THE FLOW
"""



"""
MAKE SAMPLES
"""

# Load in model
config_string = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict["n_epochs"], hyperparameters_dict["lr"], hyperparameters_dict["weight_decay"], hyperparameters_dict["batch_size"])
checkpoint_path = os.path.join(flow_training_dir, f"{config_string}_best_model.pt")
    

data_dict["SBL_samples"] = sample_from_flow(checkpoint_path, device, data_dict["SBL"][:,-1], num_features)
data_dict["SBH_samples"] = sample_from_flow(checkpoint_path, device, data_dict["SBH"][:,-1], num_features)
data_dict["SB_samples"] =  np.vstack((data_dict["SBL_samples"], data_dict["SBH_samples"]))

"""
data_dict["IBL_samples"] = sample_from_flow(checkpoint_path, device, data_dict["IBL"][:,-1], num_features)
data_dict["IBH_samples"] = sample_from_flow(checkpoint_path, device, data_dict["IBH"][:,-1], num_features)
data_dict["IB_samples"] =  np.vstack((data_dict["IBL_samples"], data_dict["IBH_samples"]))
"""
data_dict["SR_samples"] =  sample_from_flow(checkpoint_path, device, data_dict["SR"][:,-1], num_features)
# generate more samples to determine score cutoff at fixed FPR
data_dict["SR_samples_validation"] =  sample_from_flow(checkpoint_path, device, data_dict["SR"][:,-1], num_features)
# generate samples for flow decorrelation
data_dict["SR_samples_decorr"] =  sample_from_flow(checkpoint_path, device, data_dict["SR"][:,-1], num_features)




with open(f"{flow_training_dir}/flow_samples", "wb") as ofile:
    
    pickle.dump(data_dict, ofile)
    
    
print("All done!")
