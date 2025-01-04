import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import torch
import argparse
import yaml
from numba import cuda

from helpers.density_estimator import DensityEstimator
from helpers.data_transforms import clean_data
from helpers.physics_functions import get_bins, get_bins_for_scan
from helpers.stats_functions import curve_fit_m_inv
#from helpers.plotting import *
from helpers.flow_sampling import get_mass_samples, get_flow_samples
#from helpers.flow_sampling import *

parser = argparse.ArgumentParser()

# project-specific arguments
parser.add_argument("-workflow", "--workflow_path", default="workflow", help='ID associated with the directory')

# data-specific arguments
parser.add_argument("-train_samesign", "--train_samesign", action="store_true")
parser.add_argument("-bootstrap", "--bootstrap", default="bootstrap0")

parser.add_argument("-fit", "--bkg_fit_degree", default=5, type=int)
parser.add_argument("-n_bins", "--num_bins_SR", default=12, type=int)

parser.add_argument('--use_inner_bands', action="store_true", default=False)
#parser.add_argument('--use_extra_data', action="store_true", default=False)

# flow-specific arguments
parser.add_argument("-fid", "--feature_id")
parser.add_argument("-feats", "--feature_list")
parser.add_argument('-seeds', '--seeds', default="6,7,8,9,10")

# training
parser.add_argument("-c", "--configs", default="CATHODE_8")

# boostrapping details
parser.add_argument("-start", "--start", type=int)
parser.add_argument("-stop", "--stop", type=int)
parser.add_argument("-n_events", "--num_events", type=int, help='12332 for OS, 7168 for SS')

args = parser.parse_args()


train_samesign = args.train_samesign
bootstrap_start, bootstrap_stop = args.start, args.stop


use_inner_bands = False
if use_inner_bands: bands = ["SBL", "IBL", "SR", "IBH", "SBH"]
else: bands = ["SBL", "SR", "SBH"]


if train_samesign: samesign_id = "SS"
else: samesign_id = "OS"

    
feature_set = [f for f in args.feature_list.split(",")]
print(f"Using feature set {feature_set}")
num_features = len(feature_set) - 1 # context doesn't count


import yaml
with open(f"{args.workflow_path}.yaml", "r") as file:
    workflow = yaml.safe_load(file)

working_dir = workflow["file_paths"]["working_dir"]
processed_data_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/"+workflow["analysis_keywords"]["name"]+"/processed_data"
path_to_config_file = f"{working_dir}/configs/{args.configs}.yml"


# computing
device = cuda.get_current_device()
device.reset()
torch.set_num_threads(2)
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)


data_dict = {}

# dataset for the bump hunt only
with open(f"{processed_data_dir}/bootstrap0_{samesign_id}_test_band_data", "rb") as ifile:
    test_data_dict = pickle.load(ifile)

print(f"{processed_data_dir}/bootstrap0_{samesign_id}_test_band_data", "rb")
for b in bands:
    num_events_band = test_data_dict[b]["dimu_mass"].shape[0]
    data_dict[b] = np.empty((num_events_band, num_features+1))
    for i, feat in enumerate(feature_set):
        data_dict[b][:,i] = test_data_dict[b][feat].reshape(-1,1).reshape(-1,)
    print("{b} data has shape {length}.".format(b = b, length = data_dict[b].shape))


data_dict["SBL"] = clean_data(data_dict["SBL"])
data_dict["SBH"] = clean_data(data_dict["SBH"])

# to draw peakless samples from the SR, we need to first do a bkg fit in the SB
masses_to_fit = np.hstack((data_dict["SBL"][:,-1], data_dict["SBH"][:,-1]))


# for a single window, we can define the SR / SB bins on the fly
print("Defining bins on the fly...")
SB_left, SR_left = np.min(data_dict["SBL"][:,-1].reshape(-1)),  np.max(data_dict["SBL"][:,-1].reshape(-1))
SR_right, SB_right = np.min(data_dict["SBH"][:,-1].reshape(-1)),  np.max(data_dict["SBH"][:,-1].reshape(-1))
plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, binning="linear", num_bins_SR=args.num_bins_SR)
x = np.linspace(SB_left, SB_right, 100) # plot curve fit

popt_0, _, _, _, _ = curve_fit_m_inv(masses_to_fit, args.bkg_fit_degree, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)





flow_training_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/" + workflow["analysis_keywords"]["name"]+f"/models/bootstrap0_{samesign_id}/{args.feature_id}/{args.configs}/"



sampled_features = {i:[] for i in range(bootstrap_start, bootstrap_stop)}

seeds_list = [int(x) for x in args.seeds.split(",")]

# determine how many events to draw from each network
min_num_events = int(args.num_events/len(seeds_list))
num_samples = [min_num_events for i in range(len(seeds_list))]
remaining_samples = args.num_events % len(seeds_list)
for i in range(remaining_samples):
    num_samples[i] += 1

for i, s in enumerate(seeds_list): # flow seeds to read in


    # load in the best flow model
    loc_flow_training_dir = f"{flow_training_dir}/seed{s}"
    train_losses = np.load(os.path.join(loc_flow_training_dir, f"flow_train_losses.npy"))
    val_losses = np.load(os.path.join(loc_flow_training_dir, f"flow_val_losses.npy"))
    best_epoch = np.nanargmin(val_losses) - 1
    model_path = f"{loc_flow_training_dir}/flow_epoch_{best_epoch}.par"
    eval_model = DensityEstimator(path_to_config_file, num_features, eval_mode=True,load_path=model_path,device=device, verbose=False,bound=False)

    for bs in range(bootstrap_start, bootstrap_stop):
        if bs % 20 == 0:
            print(f"Bootstrapping {bs}")

        torch.manual_seed(bs)
        np.random.seed(bs)
    
        mass_samples = get_mass_samples(SB_left, SB_right, args.bkg_fit_degree, num_samples[i], popt_0)
        feature_samples = get_flow_samples(eval_model, mass_samples) 
        sampled_features[bs].append(feature_samples)

        
for bs in range(bootstrap_start, bootstrap_stop):
    sampled_features[bs] = np.vstack(sampled_features[bs])
  
        
for bs in range(bootstrap_start, bootstrap_stop):  
    in_SR = (sampled_features[bs][:,-1] >= SR_left) & (sampled_features[bs][:,-1] <= SR_right)
    in_SBL = (sampled_features[bs][:,-1] < SR_left)
    in_SBH = (sampled_features[bs][:,-1] > SR_right)

    flow_dict_SR, flow_dict_SBL, flow_dict_SBH = {},{},{}
    
    for i,feat in enumerate(feature_set):
        
        flow_dict_SR[feat] = sampled_features[bs][in_SR][:,i].reshape(-1,1)
        flow_dict_SBL[feat] = sampled_features[bs][in_SBL][:,i].reshape(-1,1)
        flow_dict_SBH[feat] = sampled_features[bs][in_SBH][:,i].reshape(-1,1)

    flow_dict = {"SR":flow_dict_SR, "SBL":flow_dict_SBL, "SBH":flow_dict_SBH}


    with open(f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/projects/upsilon_iso_12_03/processed_data/bkg_samples/bootstrap{bs}_{samesign_id}_test_band_data", "wb") as ofile:
        #pickle.dump(flow_dict, ofile)
        
        
    
            

            
