import awkward as ak
import os
import pickle
import argparse
import yaml

import matplotlib.pyplot as plt
import numpy as np

from helpers.physics_functions import assemble_m_inv, muon_mass, calculate_deltaR


parser = argparse.ArgumentParser()
parser.add_argument("-data_id", "--data_id", default="skimmed_data_2016H_30555", help="which folder in compiled_data to load")
parser.add_argument("-project_id", "--project_id", default="lowmass", help="helpful name for the analysis")
parser.add_argument("-run_jet", "--run_jet", action="store_true")
parser.add_argument("-run_samesign", "--run_samesign", action="store_true")
args = parser.parse_args()


with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)

path_to_data_dir = workflow["file_paths"]["data_storage_dir"]
working_dir = workflow["file_paths"]["working_dir"]

path_to_input = f"{path_to_data_dir}/precompiled_data/{args.data_id}/"
path_to_output = f"{path_to_data_dir}/compiled_data/{args.project_id}"
os.makedirs(path_to_output, exist_ok=True)

num_files_dict = {
    "skimmed_data_2016H_30555":28,
    "SM_SIM":49,
    "BSM_HAA":11,
    "BSM_XYY":2,
}
num_files = num_files_dict[args.data_id]

 
with open(f"{path_to_input}/all_mu_0", "rb") as input_file:
    loc_mu_feature = pickle.load(input_file)
    muon_vars = list(loc_mu_feature.keys())

with open(f"{path_to_input}/all_jet_0", "rb") as input_file:
    loc_jet_feature = pickle.load(input_file)
    jet_vars = list(loc_jet_feature.keys())
    
triggers_HLT = [var for var in muon_vars if "HLT" in var]
muon_vars = [var for var in muon_vars if "HLT" not in var]

print("Muon vars")
print(muon_vars)
print()
print("Trigger vars")
print(triggers_HLT)
print()
print("Jet vars")
print(jet_vars)
print()

all_data = {
      "dimu_pt": [],
      "dimu_eta": [],
     "dimu_phi": [],
     "n_muons": [],
     "n_jets": [],
    "dimu_mass": [],
    }

single_mu_vars_to_add = {
    "ip3d":"Muon_ip3d",
    "jetiso":"Muon_jetRelIso",
    "eta":"Muon_eta",
    "pt": "Muon_pt",
    "phi":"Muon_phi",
    "iso04":"Muon_pfRelIso04_all",
}

for key in single_mu_vars_to_add.keys():
    all_data[f"mu0_{key}"] = []
    all_data[f"mu1_{key}"] = []
    
for key in triggers_HLT:
    all_data[key] = []

# only implemented for hardest jet
hardest_jet_vars_to_add = {
    "hardest_jet_btag":"Jet_btagDeepB",
    "hardest_jet_pt":"Jet_pt",
    "hardest_jet_eta":"Jet_eta",
    "hardest_jet_phi": "Jet_phi",
    "hardest_jet_mass":"Jet_mass",
}
       
if args.run_jet:
    for key in hardest_jet_vars_to_add.keys():
        all_data[key] = []
        
a = list(all_data.keys())
if args.run_samesign:
    for key in a:
        all_data[key+"_samesign"] = []

for i in range(num_files):
    
    print(f"Analyzing file {i+1} of {num_files}...")
    
    # LOAD IN DATA
    
    with open(f"{path_to_input}/all_mu_{i}", "rb") as input_file:
        loc_mu_feature = pickle.load(input_file)
        
    with open(f"{path_to_input}/all_jet_{i}", "rb") as input_file:
        loc_jet_feature = pickle.load(input_file)
    
    if args.run_jet:
        # 2 hard muons that pass tight ID and jet
        event_filter = (np.sum(loc_mu_feature["Muon_tightId"], axis = 1) >= 2) & (ak.count(loc_jet_feature["Jet_mass"], axis = 1) >= 1)
    else:
        # 2 hard muons that pass tight ID
        event_filter = (np.sum(loc_mu_feature["Muon_tightId"], axis = 1) >= 2) 
   
    # helper function to grab tight muons
    def pull_tight_muons(feature):
        return loc_mu_feature[feature][loc_mu_feature["Muon_tightId"]][event_filter]
    
    dimu_mass, dimu_pt, dimu_eta, dimu_phi = assemble_m_inv(muon_mass, pull_tight_muons("Muon_pt")[:,0], pull_tight_muons("Muon_eta")[:,0], pull_tight_muons("Muon_phi")[:,0], 
                                   muon_mass, pull_tight_muons("Muon_pt")[:,1],  pull_tight_muons("Muon_eta")[:,1],  pull_tight_muons("Muon_phi")[:,1])
    
    total_charge = pull_tight_muons("Muon_charge")[:,0] + pull_tight_muons("Muon_charge")[:,1]
    
    # filters for opp-sign and same-sign muons; must apply *after* the event filter
    samesign_filter = np.abs(total_charge) == 2
    oppsign_filter = np.abs(total_charge) == 0

    # variables that have already had the event filter applied
    all_data["dimu_mass"].append(dimu_mass[oppsign_filter].to_numpy(allow_missing = True))
    all_data["dimu_pt"].append(dimu_pt[oppsign_filter].to_numpy(allow_missing = True))
    all_data["dimu_eta"].append(dimu_eta[oppsign_filter].to_numpy(allow_missing = True))
    all_data["dimu_phi"].append(dimu_phi[oppsign_filter].to_numpy(allow_missing = True))
    
    if args.run_samesign:
        all_data["dimu_mass_samesign"].append(dimu_mass[samesign_filter].to_numpy(allow_missing = True))
        all_data["dimu_pt_samesign"].append(dimu_pt[samesign_filter].to_numpy(allow_missing = True))
        all_data["dimu_eta_samesign"].append(dimu_eta[samesign_filter].to_numpy(allow_missing = True))
        all_data["dimu_phi_samesign"].append(dimu_phi[samesign_filter].to_numpy(allow_missing = True))
    # variables that need the event filter
    for mv in triggers_HLT:
        try:
            trigger_data = loc_mu_feature[mv][event_filter]
        except:
            trigger_data = ak.Array([False for i in range(sum(event_filter))])
        all_data[f"{mv}"].append(trigger_data[oppsign_filter].to_numpy(allow_missing = True))
        all_data[f"{mv}_samesign"].append(trigger_data[samesign_filter].to_numpy(allow_missing = True))
    
    all_data["n_jets"].append(ak.count(loc_jet_feature["Jet_mass"][event_filter], axis = 1)[oppsign_filter].to_numpy(allow_missing = True))
    if args.run_samesign:
        all_data["n_jets_samesign"].append(ak.count(loc_jet_feature["Jet_mass"][event_filter], axis = 1)[samesign_filter].to_numpy(allow_missing = True))
    
    all_data["n_muons"].append(ak.count(pull_tight_muons("Muon_charge"), axis = 1)[oppsign_filter].to_numpy(allow_missing = True))
    if args.run_samesign:
        all_data["n_muons_samesign"].append(ak.count(pull_tight_muons("Muon_charge"), axis = 1)[samesign_filter].to_numpy(allow_missing = True))
    
    # jet vars (only implemented for hardest jet)
    if args.run_jet:
        for jet_var in hardest_jet_vars_to_add.keys():          
            all_data[f"{jet_var}"].append(ak.firsts(loc_jet_feature[hardest_jet_vars_to_add[jet_var]][event_filter])[oppsign_filter].to_numpy(allow_missing = True))
            if args.run_samesign:
                all_data[f"{jet_var}_samesign"].append(ak.firsts(loc_jet_feature[hardest_jet_vars_to_add[jet_var]][event_filter][event_filter])[samesign_filter].to_numpy(allow_missing = True))

    # single muon vars        
    for single_mu_var in single_mu_vars_to_add.keys(): 
        all_data[f"mu0_{single_mu_var}"].append(pull_tight_muons(single_mu_vars_to_add[single_mu_var])[oppsign_filter][:,0].to_numpy(allow_missing = True))
        all_data[f"mu1_{single_mu_var}"].append(pull_tight_muons(single_mu_vars_to_add[single_mu_var])[oppsign_filter][:,1].to_numpy(allow_missing = True))
        if args.run_samesign:
            all_data[f"mu0_{single_mu_var}_samesign"].append(pull_tight_muons(single_mu_vars_to_add[single_mu_var])[samesign_filter][:,0].to_numpy(allow_missing = True))
            all_data[f"mu1_{single_mu_var}_samesign"].append(pull_tight_muons(single_mu_vars_to_add[single_mu_var])[samesign_filter][:,1].to_numpy(allow_missing = True))


print("Done loading in files.")
    
all_data["mumu_deltaR"] = calculate_deltaR(all_data["mu0_phi"], all_data["mu1_phi"], all_data["mu0_eta"], all_data["mu1_eta"])
all_data["mumu_deltapT"] = all_data["mu0_pt"] - all_data["mu1_pt"]

if args.run_jet:
    all_data["dimujet_deltaR"] = calculate_deltaR(all_data["dimu_phi"], all_data["hardest_jet_phi"], all_data["dimu_eta"], all_data["hardest_jet_eta"])

if args.run_samesign:
    all_data["mumu_deltaR_samesign"] = calculate_deltaR(all_data["mu0_phi_samesign"], all_data["mu1_phi_samesign"], all_data["mu0_eta_samesign"], all_data["mu1_eta_samesign"])
    all_data["mumu_deltapT_samesign"] = all_data["mu0_pt_samesign"] - all_data["mu1_pt_samesign"]
    if args.run_jet:
        all_data["dimujet_deltaR_samesign"] = calculate_deltaR(all_data["dimu_phi_samesign"], all_data["hardest_jet_phi_samesign"], all_data["dimu_eta_samesign"], all_data["hardest_jet_eta_samesign"])


print("All variables compiled:") 
for key in all_data.keys():
    
    all_data[key] = np.hstack(all_data[key])
    print("   ", key, all_data[key].shape)
print()
        
if args.run_jet:
    save_id = f"{args.data_id}_jet"
else: 
    save_id = f"{args.data_id}_nojet"


with open(f"{path_to_output}/{save_id}", "wb") as output_file:
        pickle.dump(all_data, output_file)

print(f"Data saved out to {path_to_output}/{save_id} .")