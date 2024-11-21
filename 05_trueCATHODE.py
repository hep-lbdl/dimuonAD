import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import torch
import argparse
import yaml

from helpers.density_estimator import DensityEstimator
from helpers.ANODE_training_utils import train_ANODE, plot_ANODE_losses
from helpers.data_transforms import clean_data
from helpers.physics_functions import get_bins, get_bins_for_scan, curve_fit_m_inv, bkg_fit_cubic, bkg_fit_quintic, bkg_fit_septic
from helpers.plotting import *
from helpers.evaluation import *
from helpers.flow_sampling import *

parser = argparse.ArgumentParser()
parser.add_argument("-fid", "--flow_id")
parser.add_argument("-f", "--features")
parser.add_argument("-pid", "--project_id", help='ID associated with the dataset')
parser.add_argument("-did", "--dir_id", default='logit_08_22', help='ID associated with the directory')
parser.add_argument("-fit", "--bkg_fit_type", default='quintic')
parser.add_argument("-n_bins", "--num_bins_SR", default=6, type=int)

parser.add_argument("-c", "--configs")
parser.add_argument('-seed', '--seed', default=1)
parser.add_argument('--no_logit', action="store_true", default=False,
                    help='Turns off the logit transform.')
parser.add_argument('--epochs', default=400)
parser.add_argument('--verbose', default=False)
parser.add_argument('--use_inner_bands', action="store_true", default=False)
parser.add_argument('--use_extra_data', action="store_true", default=False)
parser.add_argument('-no_train', '--no_train', action="store_true", default=False)
parser.add_argument('-premade_bins', '--premade_bins', action="store_true", default=False, help='for the lowmass scan, the bin definitions are fixed and should be loaded in')



batch_size = 256

args = parser.parse_args()

print(f"Flow id: {args.flow_id}")
print(f"Project id: {args.project_id}")
print(f"Configs: {args.configs}")




from numba import cuda 



seed = int(args.seed)


# computing
path_to_config_file = f"configs/{args.configs}.yml"

device = cuda.get_current_device()
device.reset()

# set the number of threads that pytorch will use
torch.set_num_threads(2)

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)

torch.manual_seed(seed)
np.random.seed(seed)

"""
LOAD IN DATA
"""

use_inner_bands = args.use_inner_bands
bkg_fit_type = args.bkg_fit_type
if bkg_fit_type == "cubic": bkg_fit_function = bkg_fit_cubic
elif bkg_fit_type == "quintic": bkg_fit_function = bkg_fit_quintic
elif bkg_fit_type == "septic": bkg_fit_function = bkg_fit_septic


if use_inner_bands:
    bands = ["SBL", "IBL", "SR", "IBH", "SBH"]
else:
    bands = ["SBL", "SR", "SBH"]
    

working_dir = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/projects/{args.dir_id}/"

feature_set = [f for f in args.features.split(",")]
print(f"Using feature set {feature_set}")
num_features = len(feature_set) - 1 # context doesn't count


data_dict = {}


"""
LOAD IN DEDICATED TRAIN DATA FOR FLOW
"""

# dataset for the bump hunt only
with open(f"{working_dir}/processed_data/{args.project_id}_test_band_data", "rb") as ifile:
    proc_dict_s_inj_test = pickle.load(ifile)

if args.use_extra_data:
    print("Using supplementary data...")
    # dataset just for flow training (extra data)
    with open(f"{working_dir}/processed_data/{args.project_id}_train_band_data", "rb") as ifile:
        proc_dict_s_inj_train = pickle.load(ifile)

    for b in bands:
        num_events_band = proc_dict_s_inj_train[b]["s_inj_data"]["dimu_mass"].shape[0]+proc_dict_s_inj_test[b]["s_inj_data"]["dimu_mass"].shape[0]
        data_dict[b] = np.empty((num_events_band, num_features+1))
        for i, feat in enumerate(feature_set):
            data_dict[b][:,i] = np.vstack([proc_dict_s_inj_train[b]["s_inj_data"][feat].reshape(-1,1),proc_dict_s_inj_test[b]["s_inj_data"][feat].reshape(-1,1)]).reshape(-1,)
        print("{b} data has shape {length}.".format(b = b, length = data_dict[b].shape))
  
    
else:
    for b in bands:
        num_events_band = proc_dict_s_inj_test[b]["s_inj_data"]["dimu_mass"].shape[0]
        data_dict[b] = np.empty((num_events_band, num_features+1))
        for i, feat in enumerate(feature_set):
            data_dict[b][:,i] = proc_dict_s_inj_test[b]["s_inj_data"][feat].reshape(-1,1).reshape(-1,)
        print("{b} data has shape {length}.".format(b = b, length = data_dict[b].shape))


print()

data_dict["SBL"] = clean_data(data_dict["SBL"])
data_dict["SBH"] = clean_data(data_dict["SBH"])


data_dict["SB"] =  np.vstack((data_dict["SBL"], data_dict["SBH"]))
if use_inner_bands:
    data_dict["IB"] =  np.vstack((data_dict["IBL"], data_dict["IBH"]))

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
flow_training_dir = os.path.join(f"{working_dir}/models", f"{args.project_id}/{args.flow_id}/{args.configs}/seed{args.seed}")
os.makedirs(flow_training_dir, exist_ok=True)

anode = DensityEstimator(path_to_config_file, num_features, device=device,
                         verbose=args.verbose, bound=args.no_logit)
model, optimizer = anode.model, anode.optimizer

with open(f"{flow_training_dir}/configs.txt", "w") as param_file:
    param_file.write(f"feature_set = {feature_set}\n")
    
"""
TRAIN THE FLOW
"""
if not args.no_train:
    train_ANODE(model, optimizer, train_loader, val_loader, f"flow",
                args.epochs, savedir=flow_training_dir, device=device, verbose=args.verbose, no_logit=args.no_logit, data_std=None)


# plot losses
train_losses = np.load(os.path.join(flow_training_dir, f"flow_train_losses.npy"))
val_losses = np.load(os.path.join(flow_training_dir, f"flow_val_losses.npy"))
print(train_losses)
plot_ANODE_losses(train_losses, val_losses, yrange=None,
savefig=os.path.join(flow_training_dir, f"loss_plot"),suppress_show=True)



"""
MAKE SAMPLES
"""
print()
print("Making flow samples...")

# get epoch of best val loss
best_epoch = np.nanargmin(val_losses) - 1
model_path = f"{flow_training_dir}/flow_epoch_{best_epoch}.par"

eval_model = DensityEstimator(path_to_config_file, num_features,
                         eval_mode=True,
                         load_path=model_path,
                         device=device, verbose=args.verbose,
                         bound=args.no_logit)

data_dict["SBL_samples"] = get_flow_samples(eval_model, data_dict["SBL"][:,-1]) 
data_dict["SBH_samples"] = get_flow_samples(eval_model, data_dict["SBH"][:,-1]) 
data_dict["SB_samples"] =  np.vstack((data_dict["SBL_samples"], data_dict["SBH_samples"]))

if use_inner_bands:
    data_dict["IBL_samples"] = get_flow_samples(eval_model, data_dict["IBL"][:,-1]) 
    data_dict["IBH_samples"] = get_flow_samples(eval_model, data_dict["IBH"][:,-1]) 
    data_dict["IB_samples"] =  np.vstack((data_dict["IBL_samples"], data_dict["IBH_samples"]))

# more samples for a high stats validation set
data_dict["SBL_samples_ROC"] =  get_flow_samples(eval_model, data_dict["SBL"][:,-1]) 
data_dict["SBH_samples_ROC"] =  get_flow_samples(eval_model, data_dict["SBH"][:,-1]) 


# to draw peakless samples from the SR, we need to first do a bkg fit in the SB
masses_to_fit = np.hstack((data_dict["SBL"][:,-1], data_dict["SBH"][:,-1]))

if args.premade_bins:
    window_index = int(args.project_id.split("_")[1])
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins_for_scan(f"{working_dir}/processed_data", window_index, scale_bins = True)
    SR_left, SR_right = plot_bins_SR[0], plot_bins_SR[-1]
    SB_left, SB_right = plot_bins_left[0], plot_bins_right[-1]
else:
    # for a single window, we can define the SR / SB bins on the fly
    print("Defining bins on the fly...")
    SB_left, SR_left = np.min(data_dict["SBL"][:,-1].reshape(-1)),  np.max(data_dict["SBL"][:,-1].reshape(-1))
    SR_right, SB_right = np.min(data_dict["SBH"][:,-1].reshape(-1)),  np.max(data_dict["SBH"][:,-1].reshape(-1))
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, binning="linear", num_bins_SR=args.num_bins_SR)
    
x = np.linspace(SB_left, SB_right, 100) # plot curve fit
popt_0, _, _, _, _ = curve_fit_m_inv(masses_to_fit, bkg_fit_type, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)

plt.figure(figsize = (7,5))
plt.plot(x, bkg_fit_function(x, *popt_0), lw = 3, linestyle = "dashed", label = "SB fit")
plt.hist(masses_to_fit, bins = plot_bins_all, lw = 2, histtype = "step", density = False, label = "SB data") 

# estimate number of samples
n_SR_samples = int(np.sum(bkg_fit_function(plot_centers_SR, *popt_0)))
# make samples
mass_samples = get_mass_samples(SR_left, SR_right, bkg_fit_type, n_SR_samples, popt_0)

print(mass_samples)

plt.hist(mass_samples, bins = plot_bins_all, lw = 2, histtype = "step", density = False, label = "samples")    
plt.legend()
plt.savefig(f"{flow_training_dir}/bkg_fit_{bkg_fit_type}_{args.num_bins_SR}")
         


data_dict["SR_samples"] =  get_flow_samples(eval_model, mass_samples) 
# generate more samples to determine score cutoff at fixed FPR
data_dict["SR_samples_validation"] = get_flow_samples(eval_model, mass_samples) 
# get even more samples for a ROC set
data_dict["SR_samples_ROC"] =  get_flow_samples(eval_model, mass_samples) 


with open(f"{flow_training_dir}/flow_samples_{bkg_fit_type}_{args.num_bins_SR}", "wb") as ofile:
    pickle.dump(data_dict, ofile)
    
    
print("All done!")
