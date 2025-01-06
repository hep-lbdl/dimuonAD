#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../science.mplstyle")

from sklearn.utils import shuffle

import argparse
import os

from matplotlib.backends.backend_pdf import PdfPages
import pickle

from helpers.data_transforms import clean_data, bootstrap_array, assemble_banded_datasets
from helpers.BDT import run_BDT_bump_hunt
from helpers.evaluation import convert_to_latent_space_true_cathode, get_median_percentiles

parser = argparse.ArgumentParser()

# project-specific arguments
parser.add_argument("-workflow", "--workflow_path", default="workflow", help='ID associated with the directory')

# data-specific arguments
parser.add_argument("-bf", "--bootstrap_flow", type=int, default=0)
parser.add_argument("-bd", "--bootstrap_data", type=int)

parser.add_argument("-train_samesign", "--train_samesign", action="store_true")
parser.add_argument("-fit", "--bkg_fit_degree", default='quintic')
parser.add_argument("-n_bins", "--num_bins_SR", default=6, type=int)

# flow-specific arguments
parser.add_argument("-fid", "--feature_id")
parser.add_argument('-seeds', '--seeds', default="1")
parser.add_argument("-c", "--configs", default="CATHODE_8")

# BDT-specific arguments
parser.add_argument("-ne", "--num_to_ensemble", default=100, type=int) # how many BDTs to train for a single pseudoexperiment
parser.add_argument("-nf", "--n_folds", default=5, type=int) # how many BDTs to train for a single pseudoexperiment
parser.add_argument("-start", "--start", default=0, type=int)  # how many pseudoexperiments to run
parser.add_argument("-stop", "--stop", default=1, type=int) 
parser.add_argument("-run_latent", "--run_latent", action="store_true")
parser.add_argument("-run_null", "--run_null", action="store_true")
parser.add_argument("-use_extra_data", "--use_extra_data", action="store_true")

args = parser.parse_args()

device = "cpu"

import yaml
with open(f"{args.workflow_path}.yaml", "r") as file:
    workflow = yaml.safe_load(file)

if args.train_samesign: samesign_id = "SS"
else: samesign_id = "OS"

# TODO: implement for inner bands
bands = ["SBL", "SR", "SBH"]
data_dict = {}

if args.train_samesign:
    train_data_id = "SS"
    alt_test_data_id = "OS"
else:
    train_data_id = "OS"
    alt_test_data_id = "SS"

working_dir = workflow["file_paths"]["working_dir"]
processed_data_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/"+workflow["analysis_keywords"]["name"]+"/processed_data"
flow_training_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/" + workflow["analysis_keywords"]["name"]+f"/models/bootstrap{args.bootstrap_flow}_{samesign_id}/{args.feature_id}/{args.configs}/"

# make dir to save out pickles
pickle_save_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/" + workflow["analysis_keywords"]["name"]+f"/pickles/bootstrap{args.bootstrap_flow}_{samesign_id}/{args.feature_id}/"
if args.run_null:
    pickle_save_dir += f"bkg_samples/bootstrap{args.bootstrap_data}/"
os.makedirs(pickle_save_dir, exist_ok=True)


plots_dir =  workflow["file_paths"]["working_dir"] +"/plots/" + workflow["analysis_keywords"]["name"]+f"/bootstrap{args.bootstrap_data}_{samesign_id}/"
os.makedirs(plots_dir, exist_ok = True)
if args.run_latent:
    pp = PdfPages(f"{plots_dir}/{args.feature_id}_latent.pdf")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    import torch
    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
else:
    pp = PdfPages(f"{plots_dir}/{args.feature_id}.pdf")
    
    

# load in the flow samples corresponding to the train id
seeds_list = [int(x) for x in args.seeds.split(",")]
train_samples_dict = {'SR_samples_ROC':[], 'SBL_samples_ROC':[], 'SBH_samples_ROC':[], 'SR_samples_validation':[], 'SR_samples':[]}
for seed in seeds_list:
    path_to_samples = f"{flow_training_dir}/seed{seed}/flow_samples_bkg_fit_{args.bkg_fit_degree}_num_bins_{args.num_bins_SR}"
    with open(path_to_samples, "rb") as infile: 
        loc_train_samples_dict = pickle.load(infile)
        for key in train_samples_dict.keys():
            train_samples_dict[key].append(loc_train_samples_dict[key])
for key in train_samples_dict.keys():
    train_samples_dict[key] = np.vstack(train_samples_dict[key])
    print(key, train_samples_dict[key].shape)
           

if not args.run_null:
    # load in the data corresponding to the train id
    # we actually want the "test band" here -- train is just for flow
    with open(f"{processed_data_dir}/bootstrap{args.bootstrap_data}_{train_data_id}_test_band_data", "rb") as ifile:
        test_data_dict = pickle.load(ifile)
    # load in the alternative data
    with open(f"{processed_data_dir}/bootstrap{args.bootstrap_data}_{alt_test_data_id}_test_band_data", "rb") as ifile:
        alt_test_data_dict = pickle.load(ifile)
    
    if args.use_extra_data:
        with open(f"{processed_data_dir}/bootstrap{args.bootstrap_data}_{train_data_id}_train_band_data", "rb") as ifile:
            ROC_test_data_1_dict = pickle.load(ifile)

    print(f"Loading classifier train samples from {processed_data_dir}/bootstrap{args.bootstrap_data}_{train_data_id}_test_band_data")
    print(f"Loading classifier train data from {processed_data_dir}/bootstrap{args.bootstrap_data}_{train_data_id}_test_band_data")
    print(f"Loading alternative test data from {processed_data_dir}/bootstrap{args.bootstrap_data}_{alt_test_data_id}_test_band_dat")
    print()

elif args.run_null:
    with open(f"{processed_data_dir}/bkg_samples/bootstrap{args.bootstrap_data}_{train_data_id}_test_band_data", "rb") as ifile:
        test_data_dict = pickle.load(ifile)
    # load in the alternative data
    with open(f"{processed_data_dir}/bkg_samples/bootstrap{args.bootstrap_data}_{alt_test_data_id}_test_band_data", "rb") as ifile:
        alt_test_data_dict = pickle.load(ifile)

    print(f"Loading classifier train samples from {processed_data_dir}/bkg_samples/bootstrap{args.bootstrap_data}_{train_data_id}_test_band_data")
    print(f"Loading classifier train data from {processed_data_dir}/bkg_samples/bootstrap{args.bootstrap_data}_{train_data_id}_test_band_data")
    print(f"Loading alternative test data from {processed_data_dir}/bkg_samples/bootstrap{args.bootstrap_data}_{alt_test_data_id}_test_band_dat")
    print()
    


with open(f"{flow_training_dir}/seed1/configs.txt", "rb") as infile: 
    configs = infile.readlines()[0].decode("utf-8")
    feature_set = [x.strip() for x in configs.split("'")][1::2]

print(feature_set)

n_features = len(feature_set) - 1


# Assemble the test sets -- consists of both SB and SR
        
# test set events: not used during flow training
banded_test_data = assemble_banded_datasets(test_data_dict, feature_set, bands)

# alt test set events
banded_alt_test_data = assemble_banded_datasets(alt_test_data_dict, feature_set, bands)


num_test_events = banded_test_data["SR"].shape[0]+banded_test_data["SBL"].shape[0]+banded_test_data["SBH"].shape[0]
print(f"Total number of default test events: {num_test_events}.")
num_test_events = banded_alt_test_data["SR"].shape[0]+banded_alt_test_data["SBL"].shape[0]+banded_alt_test_data["SBH"].shape[0]
print(f"Total number of alt test events: {num_test_events}.")
    

# ROC test set events
if args.use_extra_data:
    banded_ROC_test_data = assemble_banded_datasets(ROC_test_data_1_dict, feature_set, bands)
    num_test_events = banded_ROC_test_data["SR"].shape[0]+banded_ROC_test_data["SBL"].shape[0]+banded_ROC_test_data["SBH"].shape[0]
    print(f"Total number of ROC test events (and samples): {num_test_events}.")
    
    
SR_min_rescaled = np.min(banded_test_data["SR"][:,-1])
SR_max_rescaled = np.max(banded_test_data["SR"][:,-1])

# BDT HYPERPARAMETERS 
bdt_hyperparams_dict = workflow["bdt_hyperparameters"]
bdt_hyperparams_dict["n_ensemble"] = args.num_to_ensemble

all_test_data_splits = {pseudo_e:{} for pseudo_e in range(args.start, args.stop)}
all_scores_splits = {pseudo_e:{} for pseudo_e in range(args.start, args.stop)}
all_alt_data_splits = {pseudo_e:{} for pseudo_e in range(args.start, args.stop)}
all_alt_scores_splits = {pseudo_e:{} for pseudo_e in range(args.start, args.stop)}



for pseudo_e in range(args.start, args.stop):
    
    print(f"On pseudoexperiment {pseudo_e} (of {args.start} to {args.stop})...")
    
    if pseudo_e == 0:
        print("Not bootstrapping data")
        loc_alt_test_set = np.vstack([banded_alt_test_data["SR"],banded_alt_test_data["SBL"],banded_alt_test_data["SBH"]])
        if args.use_extra_data:
            loc_ROC_test_events_1 = np.vstack([banded_ROC_test_data["SR"],banded_ROC_test_data["SBL"],banded_ROC_test_data["SBH"]])
        loc_ROC_test_samples_2 = np.vstack([train_samples_dict["SR_samples_ROC"],train_samples_dict["SBL_samples_ROC"],train_samples_dict["SBH_samples_ROC"]])
        loc_SB_test_set = np.vstack([clean_data(banded_test_data["SBL"]),clean_data(banded_test_data["SBH"])])
        loc_SR_data = clean_data(banded_test_data["SR"])
        loc_FPR_val_set = train_samples_dict["SR_samples_validation"]
        
        loc_SR_samples = clean_data(train_samples_dict["SR_samples"])

    else:
         #assemble the bootstrapped datasets
        print("Bootstrapping data")
        
        loc_alt_test_set = bootstrap_array(np.vstack([banded_alt_test_data["SR"],banded_alt_test_data["SBL"],banded_alt_test_data["SBH"]]), pseudo_e)
        if args.use_extra_data:
            loc_ROC_test_events_1 = bootstrap_array(np.vstack([banded_ROC_test_data["SR"],banded_ROC_test_data["SBL"],banded_ROC_test_data["SBH"]]), pseudo_e)
        loc_ROC_test_samples_2 = bootstrap_array(np.vstack([train_samples_dict["SR_samples_ROC"],train_samples_dict["SBL_samples_ROC"],train_samples_dict["SBH_samples_ROC"]]), pseudo_e)

        bootstrapped_loc_test_set = bootstrap_array(clean_data(np.vstack([banded_test_data["SR"],banded_test_data["SBL"],banded_test_data["SBH"]])), pseudo_e)                   
        in_SR = (bootstrapped_loc_test_set[:,-1] >= SR_min_rescaled) & (bootstrapped_loc_test_set[:,-1] <= SR_max_rescaled)
        loc_SR_data = bootstrapped_loc_test_set[in_SR]
        loc_SB_test_set = bootstrapped_loc_test_set[~in_SR]                       
       
        # I think the validation set and the flow samples should NOT be bootstrapped
        loc_FPR_val_set = train_samples_dict["SR_samples_validation"]
        loc_SR_samples = clean_data(train_samples_dict["SR_samples"])
        
    print(banded_test_data["SR"][:10,-1])
    print(loc_SR_data[:10,-1])
    

    if args.run_latent:
        loc_alt_test_set = convert_to_latent_space_true_cathode(loc_alt_test_set, n_features, flow_training_dir, configs_path, device)
        if args.use_extra_data:
            loc_ROC_test_events_1 = convert_to_latent_space_true_cathode(loc_ROC_test_events_1, n_features, flow_training_dir, configs_path, device)
        loc_ROC_test_samples_2 = convert_to_latent_space_true_cathode(loc_ROC_test_samples_2, n_features, flow_training_dir, configs_path, device)
        loc_SB_test_set = convert_to_latent_space_true_cathode(loc_SB_test_set, n_features, flow_training_dir, configs_path, device)
        loc_FPR_val_set = convert_to_latent_space_true_cathode(loc_FPR_val_set, n_features, flow_training_dir, configs_path, device)
        loc_SR_data = convert_to_latent_space_true_cathode(loc_SR_data, n_features, flow_training_dir, configs_path, device)
        loc_SR_samples = convert_to_latent_space_true_cathode(loc_SR_samples, n_features, flow_training_dir, configs_path, device)

    
    loc_alt_test_sets_data = {"FPR_validation":loc_FPR_val_set,
                      "alt":loc_alt_test_set,
                     "ROC_samples":loc_ROC_test_samples_2}
    
    if args.use_extra_data:
        loc_alt_test_sets_data["ROC_data"] = loc_ROC_test_events_1

    
    if pseudo_e==0:
        loc_test_data_splits, loc_scores_splits, loc_alt_data_splits, loc_alt_scores_splits = run_BDT_bump_hunt(loc_SR_samples, loc_SR_data, loc_SB_test_set, args.n_folds, bdt_hyperparams_dict, alt_test_sets_data=loc_alt_test_sets_data, visualize=True, pdf=pp, take_ensemble_avg=True)
        pp.close()
    else:
        loc_test_data_splits, loc_scores_splits, loc_alt_data_splits, loc_alt_scores_splits = run_BDT_bump_hunt(loc_SR_samples, loc_SR_data, loc_SB_test_set, args.n_folds, bdt_hyperparams_dict, alt_test_sets_data=loc_alt_test_sets_data, visualize=False, pdf=None, take_ensemble_avg=True)

    all_test_data_splits[pseudo_e] = loc_test_data_splits
    all_scores_splits[pseudo_e] = loc_scores_splits
    all_alt_data_splits[pseudo_e] = loc_alt_data_splits
    all_alt_scores_splits[pseudo_e] = loc_alt_scores_splits
    
    
    print(10*"*"+"\n")

print("Done training BDTs!")


with open(f"{pickle_save_dir}/all_test_data_splits_bkg_fit_{args.bkg_fit_degree}_num_bins_{args.num_bins_SR}_{args.start}_{args.stop}", "wb") as ofile:
    pickle.dump(all_test_data_splits, ofile)
with open(f"{pickle_save_dir}/all_alt_data_splits_bkg_fit_{args.bkg_fit_degree}_num_bins_{args.num_bins_SR}_{args.start}_{args.stop}", "wb") as ofile:
    pickle.dump(all_alt_data_splits, ofile)
with open(f"{pickle_save_dir}/all_scores_splits_bkg_fit_{args.bkg_fit_degree}_num_bins_{args.num_bins_SR}_{args.start}_{args.stop}", "wb") as ofile:
    pickle.dump(all_scores_splits, ofile)
with open(f"{pickle_save_dir}/all_alt_scores_splits_bkg_fit_{args.bkg_fit_degree}_num_bins_{args.num_bins_SR}_{args.start}_{args.stop}", "wb") as ofile:
    pickle.dump(all_alt_scores_splits, ofile)
      
