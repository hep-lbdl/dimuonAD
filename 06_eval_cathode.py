import numpy as np
import pickle
import argparse

import os

from helpers.evaluation import *
from helpers.data_transforms import clean_data


parser = argparse.ArgumentParser()

# project-specific arguments
parser.add_argument("-workflow", "--workflow_path", default="workflow", help='ID associated with the directory')

# data-specific arguments
parser.add_argument("-bootstrap", "--bootstrap")
parser.add_argument("-train_samesign", "--train_samesign", action="store_true")
parser.add_argument("-fit", "--bkg_fit_type", default='quintic')
parser.add_argument("-n_bins", "--num_bins_SR", default=6, type=int)
parser.add_argument('--use_inner_bands', action="store_true", default=False)

# flow-specific arguments
parser.add_argument("-fid", "--feature_id")
parser.add_argument('-seeds', '--seeds', default="1")
parser.add_argument("-c", "--configs", default="CATHODE_8")

args = parser.parse_args()


use_inner_bands = args.use_inner_bands
if use_inner_bands: data_dict = {'SBL':[], 'SBH':[], 'SB':[], 'SBL_samples':[], 'SBH_samples':[], 'SB_samples':[],'IBL':[], 'IBH':[], 'IB':[], 'IBL_samples':[], 'IBH_samples':[], 'IB_samples':[]}
else: data_dict = {'SBL':[], 'SBH':[], 'SB':[], 'SBL_samples':[], 'SBH_samples':[], 'SB_samples':[]}

if args.train_samesign: samesign_id = "SS"
else: samesign_id = "OS"


import yaml
with open(f"{args.workflow_path}.yaml", "r") as file:
    workflow = yaml.safe_load(file)

working_dir = workflow["file_paths"]["working_dir"]
path_to_config_file = f"{working_dir}/configs/{args.configs}.yml"
processed_data_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/"+workflow["analysis_keywords"]["name"]+"/processed_data"
flow_training_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/" + workflow["analysis_keywords"]["name"]+f"/models/{args.bootstrap}_{samesign_id}/{args.feature_id}/{args.configs}/"


# load in all the flow models across different seeds
seeds_list = [int(x) for x in args.seeds.split(",")]

for seed in seeds_list:
    path_to_samples = f"{flow_training_dir}/seed{seed}/flow_samples_{args.bkg_fit_type}_{args.num_bins_SR}"
    with open(path_to_samples, "rb") as infile: 
        loc_data_dict = pickle.load(infile)
        for key in data_dict.keys():
            if "samples" in key or seed in [1,6]:
                data_dict[key].append(loc_data_dict[key])

for key in data_dict.keys():
    data_dict[key] = np.vstack(data_dict[key])


n_estimators = 100 # number of boosting stages
max_depth = 20 # max depth of individual regression estimators; related to complexity
learning_rate = 0.1
subsample = 0.5 # fraction of samples to be used for fitting the individual base learners
early_stopping_rounds = 10 # stop training BDT is validation loss doesn't improve after this many rounds

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import xgboost as xgb

def run_discriminator(data, samples):
    
    SB_data_train, SB_data_test = train_test_split(data_dict["SB"], test_size=0.1, random_state=42)
    SB_samples_train, SB_samples_test = train_test_split(data_dict["SB_samples"], test_size=0.1, random_state=42)
    
    
    SB_samples_train = clean_data(SB_samples_train)
    SB_samples_test = clean_data(SB_samples_test)

    X_train = np.vstack([SB_data_train, SB_samples_train])

    Y_train = np.vstack([np.ones((SB_data_train.shape[0], 1)), np.zeros((SB_samples_train.shape[0], 1))])

    X_val = np.vstack([SB_data_test, SB_samples_test])
    Y_val = np.vstack([np.ones((SB_data_test.shape[0], 1)), np.zeros((SB_samples_test.shape[0], 1))])

    n_runs = 3
    auc_list = []
    #acc_list = []


    for i in range(n_runs):

        eval_set = [(X_train, Y_train), (X_val, Y_val)]

        bst_i = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, 
                                  subsample=subsample,  early_stopping_rounds=early_stopping_rounds,
                                  objective='binary:logistic', 
                                          random_state = i, eval_metric="logloss")

        bst_i.fit(X_train, Y_train,  eval_set=eval_set,   verbose=False)
        results_f = bst_i.evals_result()
        losses = results_f["validation_0"]["logloss"]
        losses_val = results_f["validation_1"]["logloss"]
        best_epoch = bst_i.best_iteration

        loc_scores =  bst_i.predict_proba(X_val, iteration_range=(0,bst_i.best_iteration))[:,1]

        loc_auc = roc_auc_score(Y_val, loc_scores)
        if loc_auc < 0.5: loc_auc = 1.0 - loc_auc
        #loc_acc = accuracy_score(Y_val, np.round(loc_scores))

        auc_list.append(loc_auc)

    return np.mean(auc_list), np.std(auc_list), bst_i.best_iteration

    

ks_dists_samples = get_kl_dist(data_dict["SB"], data_dict["SB_samples"])
ks_dists_gaussians = get_kl_dist(np.random.normal(size = data_dict["SB"].shape), np.random.normal(size = data_dict["SB_samples"].shape))


validations_dir =  workflow["file_paths"]["working_dir"] +"/flow_training_validations/" + workflow["analysis_keywords"]["name"]+f"/{args.bootstrap}_{samesign_id}/"
os.makedirs(validations_dir, exist_ok = True)

with open(f"{validations_dir}/{args.feature_id}.txt", "w") as ofile:
                                                  
    for i, ks_dist in enumerate(ks_dists_samples):
        ofile.write("Feature {i} KL div: {ks_dist}. (for gaussian: {ks_gauss})\n".format(i=i, ks_dist=ks_dist, ks_gauss=ks_dists_gaussians[i]))
        
    ofile.write("\n")                                  
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SB"], data_dict["SB_samples"])
    ofile.write(f"SB total: auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SBL"], data_dict["SBL_samples"])
    ofile.write(f"SBL: auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SBH"], data_dict["SBH_samples"])
    ofile.write(f"SBH : auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")