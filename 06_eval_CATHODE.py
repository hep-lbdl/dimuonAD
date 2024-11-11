import numpy as np
import pickle
import argparse

import os

from helpers.evaluation import *
from helpers.data_transforms import clean_data

parser = argparse.ArgumentParser()
parser.add_argument("-feat", "--feature_set")
parser.add_argument("-p", "--particle_type")
parser.add_argument("-seeds", "--seeds", default="1", help="csv for seeds of flow models to use")
parser.add_argument("-did", "--dir_id", default='logit_08_22', help='ID associated with the directory')
parser.add_argument("-run_jet", "--run_jet", action="store_true")


args = parser.parse_args()


bands = ["SBL", "SR", "SBH"]
data_dict = {}


working_dir = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/projects/{args.dir_id}/"
if args.run_jet:
    project_id = f"lowmass_{args.particle_type}_jet"
else:
    project_id = f"lowmass_{args.particle_type}_nojet"
config_id = "CATHODE_8"

flow_training_dir = os.path.join(f"{working_dir}/models", f"{project_id}/{args.feature_set}/{config_id}")

# load in all the flow models across different seeds
seeds_list = [int(x) for x in args.seeds.split(",")]
data_dict = {'SBL':[], 'SBH':[], 'SB':[], 'SBL_samples':[], 'SBH_samples':[], 'SB_samples':[]}

for seed in seeds_list:
    path_to_samples = f"{flow_training_dir}/seed{seed}/flow_samples"
    with open(path_to_samples, "rb") as infile: 
        loc_data_dict = pickle.load(infile)
        for key in data_dict.keys():
            if "samples" in key or seed == 1:
                data_dict[key].append(loc_data_dict[key])

for key in data_dict.keys():
    data_dict[key] = np.vstack(data_dict[key])
    print(key, data_dict[key].shape)


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



with open(f"flow_training_validations/{args.particle_type}_{args.feature_set}.txt", "w") as ofile:
                                                  
    for i, ks_dist in enumerate(ks_dists_samples):
        ofile.write("Feature {i} KL div: {ks_dist}. (for gaussian: {ks_gauss})\n".format(i=i, ks_dist=ks_dist, ks_gauss=ks_dists_gaussians[i]))
        
    ofile.write("\n")
                                                  
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SB"], data_dict["SB_samples"])
    ofile.write(f"SB total: auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SBL"], data_dict["SBL_samples"])
    ofile.write(f"SBL: auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SBH"], data_dict["SBH_samples"])
    ofile.write(f"SBH : auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")