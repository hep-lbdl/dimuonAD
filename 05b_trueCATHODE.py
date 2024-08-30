import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import torch
import argparse
import yaml

from helpers.density_estimator import DensityEstimator
from helpers.ANODE_training_utils import train_ANODE, plot_ANODE_losses
    

parser = argparse.ArgumentParser()
parser.add_argument("-fid", "--flow_id")
parser.add_argument("-f", "--features")
parser.add_argument("-pid", "--project_id")
parser.add_argument("-c", "--configs")
parser.add_argument('--no_logit', action="store_true", default=False,
                    help='Turns off the logit transform.')
parser.add_argument('--epochs', default=300)
parser.add_argument('--verbose', default=False)

batch_size = 256

args = parser.parse_args()

print(f"Flow id: {args.flow_id}")
print(f"Project id: {args.project_id}")
print(f"Configs: {args.configs}")



from numba import cuda 

from helpers.make_flow import *
from helpers.train_flow import *
from helpers.plotting import *
from helpers.evaluation import *

seed = 8


# computing
path_to_config_file = f"configs/{args.configs}.yml"

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

feature_set = [f for f in args.features.split(",")]
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

print()
data_dict["SB"] =  np.vstack((data_dict["SBL"], data_dict["SBH"]))


# train val split
from sklearn.model_selection import train_test_split

SBL_data_train, SBL_data_val = train_test_split(data_dict["SBL"], test_size=0.2, random_state=42)
SBH_data_train, SBH_data_val = train_test_split(data_dict["SBH"], test_size=0.2, random_state=42)



print(f"SBL train data has shape {SBL_data_train.shape}.")
print(f"SBL val data has shape {SBL_data_val.shape}.")
print(f"SBH train data has shape {SBH_data_train.shape}.")
print(f"SBH val data has shape {SBH_data_val.shape}.")

train_loader = torch.utils.data.DataLoader(np.vstack([SBL_data_train, SBH_data_train]), batch_size=batch_size, shuffle=True, num_workers = 8, pin_memory = True)
val_loader = torch.utils.data.DataLoader(np.vstack([SBL_data_val, SBH_data_val]), batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    

"""
CREATE THE FLOW
"""
flow_training_dir = os.path.join(f"{working_dir}/models", f"{args.project_id}/{args.flow_id}/{args.configs}")
os.makedirs(flow_training_dir, exist_ok=True)

anode = DensityEstimator(path_to_config_file, num_features, device=device,
                         verbose=args.verbose, bound=args.no_logit)
model, optimizer = anode.model, anode.optimizer

with open(f"{flow_training_dir}/configs.txt", "w") as param_file:
    param_file.write(f"feature_set = {feature_set}\n")
    
"""
TRAIN THE FLOW
"""

train_ANODE(model, optimizer, train_loader, val_loader, "flow",
            args.epochs, savedir=flow_training_dir, device=device, verbose=args.verbose, no_logit=args.no_logit, data_std=None)


# plot losses
train_losses = np.load(os.path.join(flow_training_dir, "flow_train_losses.npy"))
val_losses = np.load(os.path.join(flow_training_dir, "flow_val_losses.npy"))
plot_ANODE_losses(train_losses, val_losses, yrange=None,
savefig=os.path.join(flow_training_dir, "loss_plot"),suppress_show=True)



"""
MAKE SAMPLES
"""

# get epoch of best val loss
best_epoch = np.nanargmin(val_losses) - 1
model_path = f"{flow_training_dir}/flow_epoch_{best_epoch}.par"



eval_model = DensityEstimator(path_to_config_file, num_features,
                         eval_mode=True,
                         load_path=model_path,
                         device=device, verbose=args.verbose,
                         bound=args.no_logit)


def get_samples(masses):

    feats = eval_model.model.sample(num_samples=masses.shape[0], cond_inputs=torch.tensor(masses.reshape(-1,1)).float()).detach().cpu().numpy()
    
    return np.hstack([feats, masses.reshape(-1,1)])


data_dict["SBL_samples"] = get_samples(data_dict["SBL"][:,-1]) 
data_dict["SBH_samples"] = get_samples(data_dict["SBH"][:,-1]) 
data_dict["SB_samples"] =  np.vstack((data_dict["SBL_samples"], data_dict["SBH_samples"]))


data_dict["SR_samples"] =  get_samples(data_dict["SR"][:,-1]) 
# generate more samples to determine score cutoff at fixed FPR
data_dict["SR_samples_validation"] =  get_samples(data_dict["SR"][:,-1]) 
# generate samples for flow decorrelation
data_dict["SR_samples_decorr"] =  get_samples(data_dict["SR"][:,-1]) 



with open(f"{flow_training_dir}/flow_samples", "wb") as ofile:
    pickle.dump(data_dict, ofile)
    
    
print("All done!")
