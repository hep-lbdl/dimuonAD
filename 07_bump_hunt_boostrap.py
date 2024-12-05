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

from helpers.data_transforms import clean_data
from helpers.BDT import *
from helpers.physics_functions import *
from helpers.plotting import hist_all_features_array
from helpers.evaluation import assemble_banded_datasets, convert_to_latent_space_true_cathode, get_median_percentiles

parser = argparse.ArgumentParser()

# project-specific arguments
parser.add_argument("-run", "--run_id", help='ID associated with the directory')
parser.add_argument("-project", "--project_id", help='ID associated with the dataset')
parser.add_argument("-particle", "--particle_id", help='ID associated with the dataset')
parser.add_argument("-analysis", "--analysis_test_id", help='ID associated with the dataset')

# data-specific arguments
parser.add_argument("-train_samesign", "--train_samesign", action="store_true")
parser.add_argument("-train_jet", "--train_jet", action="store_true")
parser.add_argument("-fit", "--bkg_fit_type", default='quintic')
parser.add_argument("-n_bins", "--num_bins_SR", default=6, type=int)

# flow-specific arguments
parser.add_argument("-fid", "--feature_id")
parser.add_argument('-seeds', '--seeds', default="1")
parser.add_argument("-c", "--configs", default="CATHODE_8")

# BDT-specific arguments
parser.add_argument("-ne", "--num_to_ensemble", default=10, type=int) # how many BDTs to train for a single pseudoexperiment
parser.add_argument("-nf", "--n_folds", default=5, type=int) # how many BDTs to train for a single pseudoexperiment
parser.add_argument("-start", "--start", default=0, type=int)  # how many pseudoexperiments to run
parser.add_argument("-stop", "--stop", default=1000, type=int) 
parser.add_argument("-run_latent", "--run_latent", action="store_true")
parser.add_argument("-use_extra_data", "--use_extra_data", action="store_true")

args = parser.parse_args()

device = "cpu"

import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)

if args.train_samesign: samesign_id = "SS"
else: samesign_id = "OS"
if args.train_jet: jet_id = "jet"
else: jet_id = "nojet"

# TODO: implement for inner bands
bands = ["SBL", "SR", "SBH"]
data_dict = {}

if args.train_samesign:
    train_data_id = "SS"
    alt_test_data_id = "OS"
    train_data_id_title = "SS"
else:
    train_data_id = "OS"
    alt_test_data_id = "SS"
    train_data_id_title = "OS"

working_dir = workflow["file_paths"]["working_dir"]
processed_data_dir = workflow["file_paths"]["data_storage_dir"] +f"/projects/{args.run_id}/processed_data/"
flow_training_dir = workflow["file_paths"]["data_storage_dir"] + f"/projects/{args.run_id}/models/{args.project_id}_{args.particle_id}_{args.analysis_test_id}_{samesign_id}_{jet_id}/{args.feature_id}/{args.configs}"

# make dir to save out pickles
pickle_save_dir = workflow["file_paths"]["data_storage_dir"]+f"/pickles/{args.feature_id}_{args.particle_id}_{args.analysis_test_id}_{train_data_id_title}_{jet_id}"
os.makedirs(pickle_save_dir, exist_ok=True)
os.makedirs(f"{working_dir}/plots", exist_ok = True)
    
if args.run_latent:
    pp = PdfPages(f"plots/{args.feature_id}_{args.particle_id}_{args.analysis_test_id}_{train_data_id_title}_{jet_id}_latent.pdf")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    import torch
    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
else:
    pp = PdfPages(f"plots/{args.feature_id}_{args.particle_id}_{args.analysis_test_id}_{train_data_id_title}_{jet_id}.pdf")
    
    

# load in the flow samples corresponding to the train id
seeds_list = [int(x) for x in args.seeds.split(",")]
train_samples_dict = {'SR_samples_ROC':[], 'SBL_samples_ROC':[], 'SBH_samples_ROC':[], 'SR_samples_validation':[], 'SR_samples':[]}
for seed in seeds_list:
    path_to_samples = f"{flow_training_dir}/seed{seed}/flow_samples_{args.bkg_fit_type}_{args.num_bins_SR}"
    with open(path_to_samples, "rb") as infile: 
        loc_train_samples_dict = pickle.load(infile)
        for key in train_samples_dict.keys():
            train_samples_dict[key].append(loc_train_samples_dict[key])
for key in train_samples_dict.keys():
    train_samples_dict[key] = np.vstack(train_samples_dict[key])
    print(key, train_samples_dict[key].shape)
           


# load in the data corresponding to the train id
# we actually want the "test band" here -- train is just for flow
with open(f"{processed_data_dir}/{args.project_id}_{args.particle_id}_{args.analysis_test_id}_{train_data_id_title}_{jet_id}_test_band_data", "rb") as infile: 
    test_data_dict = pickle.load(infile)

# load in the alternative data
with open(f"{processed_data_dir}/{args.project_id}_{args.particle_id}_{args.analysis_test_id}_{alt_test_data_id}_{jet_id}_test_band_data", "rb") as infile: 
    alt_test_data_dict = pickle.load(infile)

if args.use_extra_data:
    with open(f"{processed_data_dir}/{args.project_id}_{args.particle_id}_{args.analysis_test_id}_{train_data_id_title}_{jet_id}_train_band_data", "rb") as infile: 
        ROC_test_data_1_dict = pickle.load(infile)
    
# ROC set 2 is evaluated on a higher stats version of the flow samples (so may be same or opp sign)

print(f"Loading classifier train samples from {args.project_id}_{args.particle_id}_{args.analysis_test_id}_{train_data_id_title}_{jet_id}")
print(f"Loading classifier train data from {args.project_id}_{args.particle_id}_{args.analysis_test_id}_{train_data_id_title}_{jet_id}")
print(f"Loading alternative test data from {args.project_id}_{args.particle_id}_{args.analysis_test_id}_{alt_test_data_id}_{jet_id}")
print(f"Loading ROC test data from {args.project_id}_{args.particle_id}_{args.analysis_test_id}_{train_data_id_title}_{jet_id}")
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

"""
bdt_hyperparams_dict = {
    "n_estimators": 300, # number of boosting stages
    "max_depth":5, # max depth of individual regression estimators; related to complexity
    "learning_rate":0.1,  # stop training BDT is validation loss doesn't improve after this many rounds
    "subsample":0.7,   # fraction of samples to be used for fitting the individual base learners
    "early_stopping_rounds":10,
    "n_ensemble": num_to_ensemble
    
}
"""


bdt_hyperparams_dict = {
    "n_estimators": 300, # number of boosting stages
    "max_depth":3, # max depth of individual regression estimators; related to complexity
    "learning_rate":0.1,  # stop training BDT is validation loss doesn't improve after this many rounds
    "subsample":0.7,   # fraction of samples to be used for fitting the individual base learners
    "early_stopping_rounds":10,
    "n_ensemble": args.num_to_ensemble
    
}



all_test_data_splits = {pseudo_e:{} for pseudo_e in range(args.start, args.stop)}
all_scores_splits = {pseudo_e:{} for pseudo_e in range(args.start, args.stop)}
all_alt_data_splits = {pseudo_e:{} for pseudo_e in range(args.start, args.stop)}
all_alt_scores_splits = {pseudo_e:{} for pseudo_e in range(args.start, args.stop)}

def bootstrap_array(data_array):
    indices_to_take = np.random.choice(range(data_array.shape[0]), size = data_array.shape[0], replace = True) 
    return data_array[indices_to_take]

for pseudo_e in range(args.start, args.stop):
    
    print(f"On pseudoexperiment {pseudo_e} (of {args.start} to {args.stop})...")
    np.random.seed(pseudo_e) # set seed for data bootstrapping
    
    if pseudo_e == 0:
        print("Not bootstrapping data")
        loc_alt_test_set = np.vstack([banded_alt_test_data["SR"],banded_alt_test_data["SBL"],banded_alt_test_data["SBH"]])
        if args.use_extra_data:
            loc_ROC_test_events_1 = np.vstack([banded_ROC_test_data["SR"],banded_ROC_test_data["SBL"],banded_ROC_test_data["SBH"]])
        loc_ROC_test_samples_2 = np.vstack([train_samples_dict["SR_samples_ROC"],train_samples_dict["SBL_samples_ROC"],train_samples_dict["SBH_samples_ROC"]])
        loc_SB_test_set = np.vstack([clean_data(banded_test_data["SBL"]),clean_data(banded_test_data["SBH"])])
        loc_FPR_val_set = train_samples_dict["SR_samples_validation"]
        loc_SR_data = clean_data(banded_test_data["SR"])
        loc_SR_samples = clean_data(train_samples_dict["SR_samples"])
        
    else:
         #assemble the bootstrapped datasets
        print("Bootstrapping data")
        loc_alt_test_set = np.vstack([bootstrap_array(banded_alt_test_data["SR"]),bootstrap_array(banded_alt_test_data["SBL"]),bootstrap_array(banded_alt_test_data["SBH"])])
        if args.use_extra_data:
            loc_ROC_test_events_1 = np.vstack([bootstrap_array(banded_ROC_test_data["SR"]),bootstrap_array(banded_ROC_test_data["SBL"]),bootstrap_array(banded_ROC_test_data["SBH"])])
        loc_ROC_test_samples_2 = np.vstack([bootstrap_array(train_samples_dict["SR_samples_ROC"]),bootstrap_array(train_samples_dict["SBL_samples_ROC"]),bootstrap_array(train_samples_dict["SBH_samples_ROC"])])
        loc_SB_test_set = np.vstack([bootstrap_array(clean_data(banded_test_data["SBL"])),bootstrap_array(clean_data(banded_test_data["SBH"]))])
        loc_SR_data = bootstrap_array(clean_data(banded_test_data["SR"]))
        
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


with open(f"{pickle_save_dir}/all_test_data_splits_{args.bkg_fit_type}_{args.num_bins_SR}_{args.start}_{args.stop}", "wb") as ofile:
    pickle.dump(all_test_data_splits, ofile)
with open(f"{pickle_save_dir}/all_alt_data_splits_{args.bkg_fit_type}_{args.num_bins_SR}_{args.start}_{args.stop}", "wb") as ofile:
    pickle.dump(all_alt_data_splits, ofile)
with open(f"{pickle_save_dir}/all_scores_splits_{args.bkg_fit_type}_{args.num_bins_SR}_{args.start}_{args.stop}", "wb") as ofile:
    pickle.dump(all_scores_splits, ofile)
with open(f"{pickle_save_dir}/all_alt_scores_splits_{args.bkg_fit_type}_{args.num_bins_SR}_{args.start}_{args.stop}", "wb") as ofile:
    pickle.dump(all_alt_scores_splits, ofile)
      
