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

from helpers.data_transforms import inverse_transform, clean_data
from helpers.BDT import *
from helpers.physics_functions import *
from helpers.plotting import hist_all_features_array
from helpers.evaluation import assemble_banded_datasets


device = "cpu"

import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
    
config_id = "CATHODE_8"

configs_path = f"configs/{config_id}.yml"
with open(configs_path, "r") as file:
    flow_configs = yaml.safe_load(file)
    

    
parser = argparse.ArgumentParser()
parser.add_argument("-fid", "--flow_id")
parser.add_argument("-p", "--particle_type")
parser.add_argument("-t", "--train_samples_id", default="")
parser.add_argument("-ne", "--n_to_ensemble", default=10) # how many BDTs to train for a single pseudoexperiment
parser.add_argument("-nb", "--num_bootstraps",default=20)  # how many pseudoexperiments to run

args = parser.parse_args()

flow_id = args.flow_id
particle_type = args.particle_type
train_samples_id = args.train_samples_id
num_to_ensemble = args.num_to_ensemble
num_bootstraps = args.num_bootstraps

# Train classifier to discriminate train samples from train data in the SR ONLY
# By default, the test set data type is the same as `train_data_id`. There are 2 alternative test sets:
# - If the train data is opp sign, one alt test set will be same sign (and vice versa)
# - The test set used to construct the ROC curve, which is evaluated on a high-stats dataset

# In[4]:


# load in the data

bands = ["SBL", "SR", "SBH"]
data_dict = {}

working_dir = "/global/cfs/cdirs/m3246/rmastand/dimuonAD/projects/logit_08_22/"
#working_dir = "/global/u1/r/rmastand/dimuonAD/projects/powerscaler_0813/"


project_id = "lowmass"



train_data_id = train_samples_id
if train_data_id == "": 
    alt_test_data_id = "_samesign"
    train_data_id_title = "_oppsign"
elif train_data_id == "_samesign": 
    alt_test_data_id = ""
    train_data_id_title = "_samesign"
    
pp = PdfPages(f"plots/{flow_id}_{particle_type}{train_data_id_title}.pdf")

ROC_test_data_1_id = ""

with open(f"{working_dir}/models/{project_id}_{particle_type}{train_samples_id}_nojet/{flow_id}/{config_id}/flow_samples", "rb") as infile: 
    train_samples_dict = pickle.load(infile)

# we actually want the "test band" here -- train is just for flow
with open(f"{working_dir}/processed_data/{project_id}_{particle_type}{train_data_id}_nojet_test_band_data", "rb") as infile: 
    test_data_dict = pickle.load(infile)

with open(f"{working_dir}/processed_data/{project_id}_{particle_type}{alt_test_data_id}_nojet_test_band_data", "rb") as infile: 
    alt_test_data_dict = pickle.load(infile)

    
# ROC set 1 is evaluated on the FULL high stats dataset
#with open(f"{working_dir}/processed_data/{project_id}_{particle_type}{ROC_test_data_id}_nojet_train_band_data", "rb") as infile: 
with open(f"{working_dir}/processed_data/{project_id}_{particle_type}_nojet_train_band_data", "rb") as infile: 
    ROC_test_data_1_dict = pickle.load(infile)
    
# ROC set 2 is evaluated on a higher stats version of the flow samples

print(f"Loading classifier train samples from {project_id}_{particle_type}{train_samples_id}")
print(f"Loading classifier train data from {project_id}_{particle_type}{train_data_id}")
print(f"Loading alternative test data from {project_id}_{particle_type}{alt_test_data_id}")
print(f"Loading ROC test data from {project_id}_upsilon")
print()

with open(f"{working_dir}/models/{project_id}_{particle_type}{train_samples_id}_nojet/{flow_id}/{config_id}/configs.txt", "rb") as infile: 
    configs = infile.readlines()[0].decode("utf-8")
    
    feature_set = [x.strip() for x in configs.split("'")][1::2]

print(configs)
print(feature_set)



# Assemble the test sets -- consists of both SB and SR
        
# test set events: not used during flow training
banded_test_data = assemble_banded_datasets(test_data_dict, feature_set, bands)

# alt test set events
banded_alt_test_data = assemble_banded_datasets(alt_test_data_dict, feature_set, bands)

# ROC test set events
banded_ROC_test_data = assemble_banded_datasets(ROC_test_data_1_dict, feature_set, bands)

num_test_events = banded_test_data["SR"].shape[0]+banded_test_data["SBL"].shape[0]+banded_test_data["SBH"].shape[0]
print(f"Total number of default test events: {num_test_events}.")
num_test_events = banded_alt_test_data["SR"].shape[0]+banded_alt_test_data["SBL"].shape[0]+banded_alt_test_data["SBH"].shape[0]
print(f"Total number of alt test events: {num_test_events}.")
num_test_events = banded_ROC_test_data["SR"].shape[0]+banded_ROC_test_data["SBL"].shape[0]+banded_ROC_test_data["SBH"].shape[0]
print(f"Total number of ROC test events: {num_test_events}.")
num_test_events = banded_ROC_test_data["SR"].shape[0]+banded_ROC_test_data["SBL"].shape[0]+banded_ROC_test_data["SBH"].shape[0]
print(f"Total number of ROC test samples: {num_test_events}.")


SR_min_rescaled = np.min(banded_test_data["SR"][:,-1])
SR_max_rescaled = np.max(banded_test_data["SR"][:,-1])




# BDT HYPERPARAMETERS 


bdt_hyperparams_dict = {
    "n_estimators": 300, # number of boosting stages
    "max_depth":5, # max depth of individual regression estimators; related to complexity
    "learning_rate":0.1,  # stop training BDT is validation loss doesn't improve after this many rounds
    "subsample":0.7,   # fraction of samples to be used for fitting the individual base learners
    "early_stopping_rounds":10,
    "n_ensemble": 20
    
}

all_test_data_splits = {pseudo_e:{} for pseudo_e in range(num_bootstraps)}
all_scores_splits = {pseudo_e:{} for pseudo_e in range(num_bootstraps)}
all_alt_data_splits = {pseudo_e:{} for pseudo_e in range(num_bootstraps)}
all_alt_scores_splits = {pseudo_e:{} for pseudo_e in range(num_bootstraps)}

def bootstrap_array(data_array):
    indices_to_take = np.random.choice(range(data_array.shape[0]), size = data_array.shape[0], replace = True) 
    #return data_array[indices_to_take]
    return data_array

for pseudo_e in range(num_bootstraps):
    
    print(f"On pseudoexperiment {pseudo_e} of {num_bootstraps}...")
    
    # assemble the bootstrapped datasets
    # I think the validation set and the flow samples should NOT be bootstrapped
    
    # boostrapped alt set:
    loc_alt_test_set = np.vstack([bootstrap_array(banded_alt_test_data["SR"]),bootstrap_array(banded_alt_test_data["SBL"]),bootstrap_array(banded_alt_test_data["SBH"])])
    loc_ROC_test_events_1 = np.vstack([bootstrap_array(banded_ROC_test_data["SR"]),bootstrap_array(banded_ROC_test_data["SBL"]),bootstrap_array(banded_ROC_test_data["SBH"])])
    loc_ROC_test_samples_2 = np.vstack([bootstrap_array(train_samples_dict["SR_samples_ROC"]),bootstrap_array(train_samples_dict["SBL_samples_ROC"]),bootstrap_array(train_samples_dict["SBH_samples_ROC"])])
    loc_SB_test_set = np.vstack([bootstrap_array(clean_data(banded_test_data["SBL"])),bootstrap_array(clean_data(banded_test_data["SBH"]))])

    
    loc_alt_test_sets_data = {"FPR_validation":train_samples_dict["SR_samples_validation"],
                      "alt":loc_alt_test_set,
                      "ROC_data":loc_ROC_test_events_1,
                     "ROC_samples":loc_ROC_test_samples_2}
    
    # make sure the input data is also bootstrapped
    loc_test_data_splits, loc_scores_splits, loc_alt_data_splits, loc_alt_scores_splits = run_BDT_bump_hunt(clean_data(train_samples_dict["SR_samples"]), bootstrap_array(clean_data(banded_test_data["SR"])), loc_SB_test_set, n_folds, bdt_hyperparams_dict, alt_test_sets_data=loc_alt_test_sets_data, visualize=True, pdf = pp)
    
    all_test_data_splits[pseudo_e] = loc_test_data_splits
    all_scores_splits[pseudo_e] = loc_scores_splits
    all_alt_data_splits[pseudo_e] = loc_alt_data_splits
    all_alt_scores_splits[pseudo_e] = loc_alt_scores_splits

print("All done!")



with open(f"{working_dir}/processed_data/mass_scaler_{particle_type}", "rb") as ifile:
    scaler = pickle.load(ifile)
    
fpr_thresholds = [1, 0.25, 0.1, 0.05, 0.01]

# determine score cutoffs for each pseudoexperiments
score_cutoffs = {pseudo_e:{i:{threshold:0 for threshold in fpr_thresholds} for i in range(n_folds)} for pseudo_e in range(num_bootstraps)}

for pseudo_e in range(num_bootstraps):
    for i_fold in range(n_folds):
        
        loc_scores_sorted = np.sort(1.0-all_alt_scores_splits[pseudo_e]["FPR_validation"][i_fold])
        
        for threshold in fpr_thresholds:
            
            
            loc_score_cutoff = 1-loc_scores_sorted[min(int(threshold*len(loc_scores_sorted)),len(loc_scores_sorted)-1)]
            score_cutoffs[pseudo_e][i_fold][threshold] = loc_score_cutoff





SB_left = float(workflow[particle_type]["SB_left"])
SR_left = float(workflow[particle_type]["SR_left"])
SR_right = float(workflow[particle_type]["SR_right"])
SB_right = float(workflow[particle_type]["SB_right"])
    
pseudo_e_to_plot= 0


fit_type = "cubic"


"""
PLOT HISTOGRAM ON SMALL TEST SET
"""

plot_histograms_with_fits(fpr_thresholds, all_test_data_splits[pseudo_e_to_plot], all_scores_splits[pseudo_e_to_plot], score_cutoffs, scaler, fit_type,
                          f"{particle_type}{train_data_id_title} (trained on {train_data_id_title})\n"+"  ".join(feature_set[:-1])+"\n", 
                          SB_left, SR_left, SR_right, SB_right)

pp.savefig()


"""
PLOT HISTOGRAM ON ALTERNATIVE TEST SET
"""

plot_histograms_with_fits(fpr_thresholds, all_alt_data_splits[pseudo_e_to_plot]["alt"], all_alt_scores_splits[pseudo_e_to_plot]["alt"], score_cutoffs, scaler, fit_type, f"{particle_type}{alt_test_data_id} (trained on {train_data_id_title})\n"+"  ".join(feature_set[:-1])+"\n", 
                          SB_left, SR_left, SR_right, SB_right)
pp.savefig()



"""
PLOT HISTOGRAM ON ROC TEST DATA
"""


plot_histograms_with_fits(fpr_thresholds, all_alt_data_splits[pseudo_e_to_plot]["ROC_data"], all_alt_scores_splits[pseudo_e_to_plot]["ROC_data"], score_cutoffs, scaler, fit_type,f"ROC data {ROC_test_data_1_id} (trained on {train_data_id_title})\n"+"  ".join(feature_set[:-1])+"\n", SB_left, SR_left, SR_right, SB_right)
pp.savefig()



plot_histograms_with_fits(fpr_thresholds, all_alt_data_splits[pseudo_e_to_plot]["ROC_samples"], all_alt_scores_splits[pseudo_e_to_plot]["ROC_samples"], score_cutoffs, scaler, fit_type,f"ROC samples {train_data_id_title} (trained on {train_data_id_title})\n"+"  ".join(feature_set[:-1])+"\n", SB_left, SR_left, SR_right, SB_right)
pp.savefig()



TPRs, FPRs = [], []

remove_edge = True
      
fit_type = "cubic"
if fit_type == "cubic": fit_function = bkg_fit_cubic
elif fit_type == "quintic": fit_function = bkg_fit_quintic
elif fit_type == "ratio": fit_function = bkg_fit_ratio


fpr_thresholds = np.logspace(0, -3, 20)
#fpr_thresholds = np.linspace(1, 0 , 50)

# first determine score cutoffs
score_cutoffs = {pseudo_e:{i:{threshold:0 for threshold in fpr_thresholds} for i in range(n_folds)} for pseudo_e in range(num_bootstraps)}

for pseudo_e in range(num_bootstraps):
    for i_fold in range(n_folds):
        loc_scores_sorted = np.sort(1.0-all_alt_scores_splits[pseudo_e]["FPR_validation"][i_fold])
        for threshold in fpr_thresholds:
            loc_score_cutoff = 1-loc_scores_sorted[min(int(threshold*len(loc_scores_sorted)),len(loc_scores_sorted)-1)]
            score_cutoffs[pseudo_e][i_fold][threshold] = loc_score_cutoff


S_yield, B_yield = np.empty((fpr_thresholds.shape[0], num_bootstraps)), np.empty((fpr_thresholds.shape[0], num_bootstraps))

for pseudo_e in range(num_bootstraps):
    
    print(f"On pseudo experiment {pseudo_e}...")
    for t, threshold in enumerate(fpr_thresholds):
        
        filtered_masses_bs = []
    
        for i_fold in range(n_folds):
            

            loc_true_masses_bs = scaler.inverse_transform(np.array(all_alt_data_splits[pseudo_e]["ROC"][i_fold][:,-1]).reshape(-1,1))
            
            loc_scores_bs = all_alt_scores_splits[pseudo_e]["ROC"][i_fold]
            # filter top event based on score cutoff
            loc_filtered_masses_bs, _, _, _ = select_top_events_fold(loc_true_masses_bs, loc_scores_bs, score_cutoffs[pseudo_e][i_fold][threshold], plot_bins_left, plot_bins_right, plot_bins_SR)
            filtered_masses_bs.append(loc_filtered_masses_bs)

        filtered_masses_bs = np.concatenate(filtered_masses_bs)

        # get the fit function to SB background
        popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(filtered_masses_bs, fit_type, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_all)
        num_S_expected_in_SR, num_B_expected_in_SR = calc_significance(filtered_masses_bs, fit_function, plot_bins_SR, SR_left, SR_right, popt)
        
        S_yield[t, pseudo_e] = num_S_expected_in_SR
        B_yield[t, pseudo_e] = num_B_expected_in_SR


print("All done!")      
        





# calculate summary stats
TPR = S_yield/S_yield[0,:]
FPR = B_yield/B_yield[0,:]

ROC = 1.0/FPR

SIC = TPR/np.sqrt(FPR)


def get_median_percentiles(x_array):
    
    
    x_median = np.median(x_array, axis = 1)
    x_lower = np.percentile(x_array, 16, axis = 1)
    x_upper = np.percentile(x_array, 84, axis = 1)

    return x_median, x_lower, x_upper


# In[17]:


TPR_median, TPR_lower, TPR_upper = get_median_percentiles(TPR)
FPR_median, FPR_lower, FPR_upper = get_median_percentiles(FPR)
ROC_median, ROC_lower, ROC_upper = get_median_percentiles(ROC)
SIC_median, SIC_lower, SIC_upper = get_median_percentiles(SIC)

print(f"Classifier trained on {feature_set[:-1]}")

plt.figure(figsize = (5, 5))



plt.plot(FPR_median, TPR_median)
plt.fill_between(FPR_median, TPR_lower, TPR_upper, alpha = 0.3 )

plt.plot(FPR_median, FPR_median, linestyle = "dashed", color = "grey")
plt.xlabel("FPR")
plt.ylabel("TPR")


plt.title(f"{particle_type} {ROC_test_data_id} (trained on {train_data_id_title})")
pp.savefig()

fig, ax = plt.subplots(1, 2, figsize = (10, 4))


ax[0].plot(TPR_median, ROC_median)
ax[0].fill_between(TPR_median, ROC_lower, ROC_upper, alpha = 0.3 )
ax[0].plot(TPR_median, 1.0/TPR_median, linestyle = "dashed", color = "grey")
ax[0].set_xlabel("TPR")
ax[0].set_ylabel("1/FPR")
ax[0].set_yscale("log")


    
ax[1].plot(TPR_median, SIC_median)
ax[1].fill_between(TPR_median, SIC_lower, SIC_upper, alpha = 0.3 )
ax[1].plot(TPR_median, TPR_median/np.sqrt(TPR_median), linestyle = "dashed", color = "grey")
ax[1].set_xlabel("TPR")
ax[1].set_ylabel("SIC")

pp.savefig()




# In[19]:


pp.close()

