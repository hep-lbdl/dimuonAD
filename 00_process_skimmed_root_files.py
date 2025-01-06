import uproot
import numpy as np
import argparse
import os
import pickle
import yaml

"""
"Process" the OPEN DATA (13 TeV) that has already been skimmed. All this does is select out a set of the features from the root files and save ino smaller directories. 

source: https://opendata.cern.ch/record/30555 for root. 28 files total

"""

parser = argparse.ArgumentParser()
parser.add_argument("-start", "--start", default=0, type=int)
parser.add_argument("-stop", "--stop", default=28, type=int)
args = parser.parse_args()

with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
path_to_root_file_dir = workflow["file_paths"]["data_storage_dir"]
working_dir = workflow["file_paths"]["working_dir"]

path_to_root_input = f"{path_to_root_file_dir}/precompiled_data/skimmed_root_2016H_30555/"
path_to_output = f"{path_to_root_file_dir}/precompiled_data/skimmed_data_2016H_30555/"

os.makedirs(path_to_output, exist_ok=True)


paths_to_root_file_list = [f"{working_dir}/skim_helpers/CMS_Run2016H_DoubleMuon_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_2510000_file_index.txt"]


list_of_root_files = []
for path in paths_to_root_file_list:
    with open(path, "r") as infile:
        root_file_list = infile.readlines()
        list_of_root_files += root_file_list
        
num_jets_to_save = 3

file_paths = []
        
for i in range(args.start, args.stop):
    
    loc_root_file_id = list_of_root_files[i].strip().split("/")[-1][:-5]
    file_paths.append(loc_root_file_id)
    loc_root_file_string = (path_to_root_input+loc_root_file_id)+"_Skim.root"
    print("Analyzing file", i, loc_root_file_string)

    loc_root_file = uproot.open(loc_root_file_string)
    events = loc_root_file["Events;1"]
    triggers_HLT = ["HLT_DoubleMu0", "HLT_DoubleMu18NoFiltersNoVtx", "HLT_DoubleMu23NoFiltersNoVtxDisplaced", "HLT_DoubleMu28NoFiltersNoVtxDisplaced", 
                             "HLT_DoubleMu33NoFiltersNoVtx", "HLT_DoubleMu38NoFiltersNoVtx", "HLT_DoubleMu8_Mass8_PFHT250", "HLT_DoubleMu8_Mass8_PFHT300", 
                             "HLT_L2DoubleMu23_NoVertex", "HLT_L2DoubleMu28_NoVertex_2Cha_Angle2p5_Mass10", "HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_Mass10",
                             "HLT_Mu10_CentralPFJet30_BTagCSV_p13", "HLT_Mu17_Mu8_DZ", "HLT_Mu17_Mu8_SameSign_DZ", "HLT_Mu17_Mu8_SameSign", "HLT_Mu17_Mu8", 
                             "HLT_Mu17_TkMu8_DZ", "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ", "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL", "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",
                             "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL", "HLT_Mu17_TrkIsoVVL", "HLT_Mu17", "HLT_Mu20_Mu10_DZ", "HLT_Mu20_Mu10_SameSign_DZ", 
                             "HLT_Mu20_Mu10_SameSign", "HLT_Mu20_Mu10", "HLT_Mu27_TkMu8", "HLT_Mu30_TkMu11", "HLT_Mu3_PFJet40", "HLT_Mu40_TkMu11", "HLT_Mu8_TrkIsoVVL",
                             "HLT_Mu8", "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ", "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL", "HLT_TripleMu_12_10_5", 
                             "HLT_TripleMu_5_3_3_DZ_Mass3p8", "HLT_TripleMu_5_3_3", "HLT_TrkMu15_DoubleTrkMu5NoFiltersNoVtx", "HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx"]

    muon_vars = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_pfRelIso03_all", "Muon_pfRelIso04_all", "Muon_tightId", "Muon_jetIdx", "Muon_ip3d", "Muon_jetRelIso", "Muon_dxy", "Muon_dz"]
    # store the triggers as extra muons vars
    muon_vars += triggers_HLT
    #electron_vars = ["Electron_pt", "Electron_eta", "Electron_phi", "Electron_charge"]
    jet_vars = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_nConstituents", "Jet_btagCSVV2", "Jet_btagDeepB", "Jet_btagDeepFlavB", "MET_pt", "MET_sumEt", "PV_npvsGood", "Jet_nMuons", "Jet_qgl", "Jet_muEF", "Jet_chHEF", "Jet_chEmEF", "Jet_neEmEF", "Jet_neHEF"]
    

    
    all_muon_data = {}
    #all_electron_data = {}
    all_jet_data = {}
    for mv in muon_vars:
        try:
            all_muon_data[mv] = events[mv].array()
        except:
            print(f"   No variable {mv} in this file")
    #for ev in electron_vars:
    #    all_electron_data[ev] = events[ev].array()
    for jv in jet_vars:
        all_jet_data[jv] = events[jv].array()
        
    # save out
    with open(f"{path_to_output}/all_mu_{loc_root_file_id}", "wb") as output_file:
        pickle.dump(all_muon_data, output_file)
        
    #with open(f"{path_to_output}/all_e_{i}", "wb") as output_file:
     #   pickle.dump(all_electron_data, output_file)

    with open(f"{path_to_output}/all_jet_{loc_root_file_id}", "wb") as output_file:
        pickle.dump(all_jet_data, output_file)
 

print("All done!")
print(f"Saved files {file_paths}")