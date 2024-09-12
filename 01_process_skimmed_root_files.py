import uproot
import numpy as np
import argparse
import pickle
import json

"""
"Process" the OPEN DATA (13 TeV) that has already been skimmed. All this does is select out a set of the features from the root files and save ino smaller directories. 
"""

# source: https://opendata.cern.ch/record/30555 for root. 28 files total

parser = argparse.ArgumentParser()
parser.add_argument("-start", "--start", help="increase output verbosity")
parser.add_argument("-stop", "--stop", help="increase output verbosity")
#parser.add_argument("-code", "--code", help="increase output verbosity")

args = parser.parse_args()

path_to_root_input= "/global/cfs/cdirs/m3246/rmastand/dimuonAD/data/skimmed_root_2016H_30555/"
path_to_output = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/data/skimmed_data_2016H_30555/"


paths_to_root_file_list = ["/global/homes/r/rmastand/dimuonAD/file_sources/CMS_Run2016H_DoubleMuon_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_2510000_file_index.txt"]


list_of_root_files = []
for path in paths_to_root_file_list:
    with open(path, "r") as infile:
        root_file_list = infile.readlines()
        list_of_root_files += root_file_list
        
        
num_jets_to_save = 3
        
#print(len(list_of_root_files))
        
for i in range(int(args.start), int(args.stop)):
    
    loc_root_file_string = loc_root_file_string = (path_to_root_input+list_of_root_files[i].strip().split("/")[-1])[:-5]+"_Skim.root"
    print("Analyzing file", i, loc_root_file_string)

    loc_root_file = uproot.open(loc_root_file_string)
    events = loc_root_file["Events;1"]

    # only store the muons for now
    muon_vars = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_pfRelIso03_all", "Muon_pfRelIso04_all", "Muon_tightId", "Muon_jetIdx"]
    electron_vars = ["Electron_pt", "Electron_eta", "Electron_phi", "Electron_charge"]
    jet_vars = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_nConstituents", "Jet_btagCSVV2", "Jet_btagDeepB", "Jet_btagDeepFlavB", "MET_pt", "MET_sumEt"]
    
    all_muon_data, all_electron_data = {}, {}
    all_jet_data = {}
    for mv in muon_vars:
        all_muon_data[mv] = events[mv].array()
    for ev in electron_vars:
        all_electron_data[ev] = events[ev].array()
    for jv in jet_vars:
        all_jet_data[jv] = events[jv].array()
        
        
    # save out
    with open(f"{path_to_output}/all_mu_{i}", "wb") as output_file:
        pickle.dump(all_muon_data, output_file)
        
    with open(f"{path_to_output}/all_e_{i}", "wb") as output_file:
        pickle.dump(all_electron_data, output_file)

    with open(f"{path_to_output}/all_jet_{i}", "wb") as output_file:
        pickle.dump(all_jet_data, output_file)

print("All done!")