import uproot
import numpy as np
import argparse
import os
import pickle
import yaml


"""
"Process" simulation, i.e. any files that do not need to be skimmed
"""

parser = argparse.ArgumentParser()
parser.add_argument("-start", "--start", default=0, type=int)
parser.add_argument("-stop", "--stop", default=28, type=int)
parser.add_argument("-code", "--code")

args = parser.parse_args()


with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
path_to_root_file_dir = workflow["file_paths"]["data_storage_dir"]
working_dir = workflow["file_paths"]["working_dir"]

path_to_output = f"{path_to_root_file_dir}/precompiled_data/{args.code}/"

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)

# SM, SIM
# https://opendata.cern.ch/record/44211, 49 files
if args.code == "SM_SIM":
    paths_to_root_file_list = [f"{working_dir}/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_MuMuJet_mll_0to60_LO_EMEnriched_TuneCP5_13TeV-amcatnlo-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_50000_file_index.txt",
                               f"{working_dir}/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_MuMuJet_mll_0to60_LO_EMEnriched_TuneCP5_13TeV-amcatnlo-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_40000_file_index.txt",
                               f"{working_dir}/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_MuMuJet_mll_0to60_LO_EMEnriched_TuneCP5_13TeV-amcatnlo-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt",
                               f"{working_dir}/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_MuMuJet_mll_0to60_LO_EMEnriched_TuneCP5_13TeV-amcatnlo-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2430000_file_index.txt"]

#  USYGluGluToHToAA_AToMuMu_AToBB_M-40
# https://opendata.cern.ch/record/65045, 11 files
elif args.code == "BSM_HAA":
    paths_to_root_file_list = [f"{working_dir}/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_SUSYGluGluToHToAA_AToMuMu_AToBB_M-40_TuneCP5_13TeV_madgraph_pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_40000_file_index.txt"]

#  ggXToYY_YToMuMu_M22p5
# https://opendata.cern.ch/record/36434, 2 files
elif args.code == "BSM_XYY":
    paths_to_root_file_list = [f"{working_dir}/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_ggXToYY_YToMuMu_M22p5_JPCZeroPlusPlus_TuneCP5_13TeV-pythia8-JHUGen_NANOAODSIM_106X_mcRun2_asymptotic_v17-v2_40000_file_index.txt", f"{working_dir}/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_ggXToYY_YToMuMu_M22p5_JPCZeroPlusPlus_TuneCP5_13TeV-pythia8-JHUGen_NANOAODSIM_106X_mcRun2_asymptotic_v17-v2_2430000_file_index.txt"]
    


list_of_root_files = []
for path in paths_to_root_file_list:
    with open(path, "r") as infile:
        root_file_list = infile.readlines()
        list_of_root_files += root_file_list
        
num_jets_to_save = 3
        
print(f"{len(list_of_root_files)} files total to analyze.\n")
        
for i in range(int(args.start), int(args.stop)):
    
    loc_root_file_string = list_of_root_files[i]
    print("Analyzing file", i, loc_root_file_string)

    loc_root_file = uproot.open(loc_root_file_string)
    events = loc_root_file["Events;1"]

    # only store the muons for now
    muon_vars = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_pfRelIso03_all", "Muon_pfRelIso04_all"]
    jet_vars = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_nConstituents", "Jet_btagCSVV2", "Jet_btagDeepB", "Jet_btagDeepFlavB"]
    
    all_muon_data, filtered_muon_data, filtered_amuon_data = {}, {}, {}
    all_jet_data, filtered_jet_data = {}, {}
    for mv in muon_vars:
        all_muon_data[mv] = events[mv].array()
    for jv in jet_vars:
        all_jet_data[jv] = events[jv].array()

    # save out
    with open(f"{path_to_output}/all_mu_{i}", "wb") as output_file:
        pickle.dump(all_muon_data, output_file)

    with open(f"{path_to_output}/all_jet_{i}", "wb") as output_file:
        pickle.dump(all_jet_data, output_file)

print("All done!")