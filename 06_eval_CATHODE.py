import numpy as np
import pickle
import argparse

import os

from helpers.evaluation import *
from helpers.data_transforms import clean_data

parser = argparse.ArgumentParser()
parser.add_argument("-feat", "--feature_set")
parser.add_argument("-p", "--particle_type")

args = parser.parse_args()


bands = ["SBL", "SR", "SBH"]
data_dict = {}

working_dir = "/global/cfs/cdirs/m3246/rmastand/dimuonAD/projects/logit_08_22/"
project_id = f"lowmass_{args.particle_type}_nojet"
config_id = "CATHODE_8"

flow_training_dir = os.path.join(f"{working_dir}/models", f"{project_id}/{args.feature_set}/{config_id}")


with open(f"{flow_training_dir}/flow_samples", "rb") as infile: 
    data_dict = pickle.load(infile)
    
with open(f"{working_dir}/processed_data/{project_id}_test_band_data", "rb") as infile: 
    test_dict = pickle.load(infile)

with open(f"{flow_training_dir}/configs.txt", "rb") as infile: 
    configs = infile.readlines()[0].decode("utf-8")
    print(configs)
    
    
feature_sets = {
    
    "single_0": ['mu0_pt', 'mu1_pt', 'mu0_eta', 'mu1_eta'] ,
    "single_1": ['mu0_iso04', 'mu1_iso04'] ,
    "single_2": ['mu0_pt', 'mu1_pt'] ,
    "single_3": ['mu0_ip3d', 'mu1_ip3d', 'mu0_jetiso', 'mu1_jetiso'] ,
    "double_0": ['dimu_pt', 'dimu_eta', 'mumu_deltaR', 'mumu_deltapT'] ,
    "double_1": ['dimu_pt', 'dimu_eta'] ,
    "double_2": ['mumu_deltaR', 'mumu_deltapT'] ,
    "mix_0": ['dimu_pt', 'mu0_iso04', 'mu1_iso04'] ,
}


feature_set = feature_sets[args.feature_set]
    
# Assemble the test set -- consists of both SB and SR

num_events_test_SBL = test_dict["SBL"]["s_inj_data"]["dimu_mass"].shape[0]
num_events_test_SBH = test_dict["SBH"]["s_inj_data"]["dimu_mass"].shape[0]

test_events_SBL = np.empty((num_events_test_SBL, len(feature_set)))
test_events_SBH = np.empty((num_events_test_SBH, len(feature_set)))


for i, feat in enumerate(feature_set):
    test_events_SBL[:,i] = test_dict["SBL"]["s_inj_data"][feat].reshape(-1,)
    test_events_SBH[:,i] = test_dict["SBH"]["s_inj_data"][feat].reshape(-1,)

    

test_events_SB = np.vstack([test_events_SBL, test_events_SBH ])
    
print(f"Total number of test events: {test_events_SBL.shape[0]+test_events_SBH.shape[0]} in SB.")



n_estimators = 100 # number of boosting stages
max_depth = 20 # max depth of individual regression estimators; related to complexity
learning_rate = 0.1
subsample = 0.5 # fraction of samples to be used for fitting the individual base learners
early_stopping_rounds = 10 # stop training BDT is validation loss doesn't improve after this many rounds


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import xgboost as xgb

def run_discriminator(data, samples):


    SB_data_train, SB_data_test, SB_samples_train, SB_samples_test = train_test_split(data, samples, 
                                                                                      test_size=0.1, random_state=42)
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
        #loc_acc = accuracy_score(Y_val, np.round(loc_scores))

        auc_list.append(loc_auc)

    return np.mean(auc_list), np.std(auc_list), bst_i.best_iteration

    



with open(f"flow_training_validations/{args.particle_type}_{args.feature_set}.txt", "w") as ofile:
    
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SB"], data_dict["SB_samples"])
    ofile.write(f"SB total: auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SBL"], data_dict["SBL_samples"])
    ofile.write(f"SBL: auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")
    
    auc_mean, auc_std, best_epoch = run_discriminator(data_dict["SBH"], data_dict["SBH_samples"])
    ofile.write(f"SBH : auc {auc_mean} \pm {auc_std}. best epoch {best_epoch} of {n_estimators}.\n")