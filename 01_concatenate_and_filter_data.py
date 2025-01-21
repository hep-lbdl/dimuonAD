"""
TODO: this function will throw a runtime warning if a negative invariant dimuon mass is calculated or the dimuon pT <= 0. This is taken care of in the code and so the warning can be safely ignored.
"""

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

files_dict = {
    "skimmed_data_2016H_30555":['127C2975-1B1C-A046-AABF-62B77E757A86', '183BFB78-7B5E-734F-BBF5-174A73020F89', '1BE226A3-7A8D-1B43-AADC-201B563F3319', '1DE780E2-BCC2-DC48-815D-9A97B2A4A2CD', '21DA4CE5-4E50-024F-9CE1-50C77254DD4E', '2C6A0345-8E2E-9B41-BB51-DB56DFDFB89A', '3676E287-A650-8F44-BBCB-3B8556966406', '411A019C-7058-FD42-AD50-DE74433E6859', '46A8960A-E58F-4648-9C12-2708FE7C12FB', '4F0B53A7-6440-924B-AF48-B5B61D3CE23F', '790F8A75-8256-3B46-8209-850DE0BE3C77', '7F53D1DE-439E-AD48-871E-D3458DABA798', '8A696857-C147-B04A-905A-F85FB76EDA23', '8B253755-51F2-CB49-A4B6-C79637CAE23F', '9528EA75-1C0B-9047-A9A3-6A47564F7A98', 'A6605227-0B58-864E-8422-B8990D18F622', 'B2DC29E0-8679-1D4F-A5AE-E7D0284A20D4', 'B450B2B3-BEF8-8C43-82BF-7AD0EF2EA7EA', 'B7AA7F04-5D5F-514A-83A6-9A275198852C', 'B93B57BF-4239-A049-9531-4C542C370185', 'C4558F81-9F2C-1349-B528-6B9DD6838D6D', 'C8CFC890-D4B8-8A4F-8699-C6ACCDF1620A', 'CAA285FF-7A12-F945-9183-DC7042178535', 'CD267D88-E57D-3B44-AC45-0712E2E12B87', 'E7C51551-7A75-5C41-B468-46FB922F36A9', 'EBC200F4-C06F-CE45-BAAA-7CAECDD3076F', 'EEB2FE3F-7CF3-BF4A-9F70-3F89FACE698E', 'F5E234F9-1E9C-0042-B395-AB6407E4A336'],
}

num_files = len(files_dict[args.data_id])

 
with open(f"{path_to_input}/all_mu_{files_dict[args.data_id][0]}", "rb") as input_file:
    loc_mu_feature = pickle.load(input_file)
    muon_vars = list(loc_mu_feature.keys())

with open(f"{path_to_input}/all_jet_{files_dict[args.data_id][0]}", "rb") as input_file:
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
    
    print(f"Analyzing file {i+1} of {num_files} (index {files_dict[args.data_id][i]})...")
    
    # LOAD IN DATA
    
    with open(f"{path_to_input}/all_mu_{files_dict[args.data_id][i]}", "rb") as input_file:
        loc_mu_feature = pickle.load(input_file)
        
    with open(f"{path_to_input}/all_jet_{files_dict[args.data_id][i]}", "rb") as input_file:
        loc_jet_feature = pickle.load(input_file)

    """
    EVENT FILTER DEFINITION
    """
    
    if args.run_jet:
        # 2 hard muons that pass tight ID and jet
        event_filter = (np.sum(loc_mu_feature["Muon_tightId"], axis = 1) >= 2) & (ak.count(loc_jet_feature["Jet_mass"], axis = 1) >= 1)
    else:
        # 2 hard muons that pass tight ID
        event_filter = (np.sum(loc_mu_feature["Muon_tightId"], axis = 1) >= 2) 
   
    # helper function to grab tight muons
    def pull_tight_muons(feature):
        return loc_mu_feature[feature][loc_mu_feature["Muon_tightId"]][event_filter]
    
    dimu_mass, dimu_pt, dimu_eta, dimu_phi, good_event_inds = assemble_m_inv(muon_mass, pull_tight_muons("Muon_pt")[:,0], pull_tight_muons("Muon_eta")[:,0], pull_tight_muons("Muon_phi")[:,0], 
                                   muon_mass, pull_tight_muons("Muon_pt")[:,1],  pull_tight_muons("Muon_eta")[:,1],  pull_tight_muons("Muon_phi")[:,1])
    
    total_charge = pull_tight_muons("Muon_charge")[:,0] + pull_tight_muons("Muon_charge")[:,1]
    
    # filters for opp-sign and same-sign muons; must apply *after* the event filter
    samesign_filter = np.logical_and(np.abs(total_charge) == 2, good_event_inds)
    oppsign_filter = np.logical_and(np.abs(total_charge) == 0, good_event_inds)

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
        if args.run_samesign:
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
                all_data[f"{jet_var}_samesign"].append(ak.firsts(loc_jet_feature[hardest_jet_vars_to_add[jet_var]][event_filter])[samesign_filter].to_numpy(allow_missing = True))

    # single muon vars        
    for single_mu_var in single_mu_vars_to_add.keys(): 
        all_data[f"mu0_{single_mu_var}"].append(pull_tight_muons(single_mu_vars_to_add[single_mu_var])[oppsign_filter][:,0].to_numpy(allow_missing = True))
        all_data[f"mu1_{single_mu_var}"].append(pull_tight_muons(single_mu_vars_to_add[single_mu_var])[oppsign_filter][:,1].to_numpy(allow_missing = True))
        if args.run_samesign:
            all_data[f"mu0_{single_mu_var}_samesign"].append(pull_tight_muons(single_mu_vars_to_add[single_mu_var])[samesign_filter][:,0].to_numpy(allow_missing = True))
            all_data[f"mu1_{single_mu_var}_samesign"].append(pull_tight_muons(single_mu_vars_to_add[single_mu_var])[samesign_filter][:,1].to_numpy(allow_missing = True))


print("Done loading in files.")

    
for key in all_data.keys():
    
    all_data[key] = np.hstack(all_data[key])
    print("   ", key, all_data[key].shape)
print()


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

        
if args.run_jet:
    save_id = f"{args.data_id}_jet"
else: 
    save_id = f"{args.data_id}_nojet"


with open(f"{path_to_output}/{save_id}", "wb") as output_file:
        pickle.dump(all_data, output_file)

print(f"Data saved out to {path_to_output}/{save_id} .")