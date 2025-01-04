import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import torch
import argparse
import yaml

    

parser = argparse.ArgumentParser()
parser.add_argument("-fid", "--flow_id")
parser.add_argument("-f", "--features")
parser.add_argument("-pid", "--project_id")
parser.add_argument("-c", "--configs")

args = parser.parse_args()


from numba import cuda 

from helpers.make_flow import *
from helpers.train_flow import *
from helpers.plotting import *
from helpers.evaluation import *

seed = 8

with open(f"configs/{args.configs}.yml", "r") as file:
    flow_configs = yaml.safe_load(file)
    

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
working_dir = "/pscratch/sd/r/rmastand/dimuonAD/projects/logit_08_22/"


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


test_flow = make_masked_AR_flow(num_features, flow_configs["num_layers"], flow_configs["num_hidden_features"], flow_configs["num_blocks"], flow_configs["spline_type"])



flow_training_dir = os.path.join(f"{working_dir}/models", f"{args.project_id}/{args.flow_id}/{args.configs}")
os.makedirs(flow_training_dir, exist_ok=True)

pytorch_total_params = sum(p.numel() for p in test_flow.parameters() if p.requires_grad)
print(f"Numb. trainable params: {pytorch_total_params}")

with open(f"{flow_training_dir}/configs.txt", "w") as param_file:
    param_file.write(f"feature_set = {feature_set}\n")


"""
TRAIN THE FLOW
"""


"""
epochs, losses, losses_val, best_epoch = train_flow(test_flow, flow_configs["hyperparameters_dict"], device, np.vstack([SBL_data_train, SBH_data_train]), np.vstack([SBL_data_val, SBH_data_val]), flow_training_dir, "flow",  seed = seed, early_stop_patience=flow_configs["early_stop_patience"])

plt.figure()
plt.plot(epochs, losses, label = "loss")
plt.plot(epochs, losses_val, label = "val loss")
plt.legend()
plt.savefig(f"{flow_training_dir}/loss")
"""


"""
MAKE SAMPLES
"""

# Load in model
checkpoint_path = os.path.join(flow_training_dir, f"flow_best_model.pt")
    

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
