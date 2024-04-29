import uproot
import numpy as np
import argparse
import pickle
import json

# source: https://opendata.cern.ch/record/44211 for root. 49 files total

parser = argparse.ArgumentParser()
parser.add_argument("-start", "--start", help="increase output verbosity")
parser.add_argument("-stop", "--stop", help="increase output verbosity")
parser.add_argument("-code", "--code", help="increase output verbosity")

args = parser.parse_args()

path_to_output = "/pscratch/sd/r/rmastand/dimuonAD/post_root_sim/"

# SM, SIM
if args.code == "SM_SIM":
    paths_to_root_file_list = ["/global/homes/r/rmastand/dimuonAD/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_MuMuJet_mll_0to60_LO_EMEnriched_TuneCP5_13TeV-amcatnlo-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_50000_file_index.txt",
                               "/global/homes/r/rmastand/dimuonAD/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_MuMuJet_mll_0to60_LO_EMEnriched_TuneCP5_13TeV-amcatnlo-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_40000_file_index.txt",
                               "/global/homes/r/rmastand/dimuonAD/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_MuMuJet_mll_0to60_LO_EMEnriched_TuneCP5_13TeV-amcatnlo-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt",
                               "/global/homes/r/rmastand/dimuonAD/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_MuMuJet_mll_0to60_LO_EMEnriched_TuneCP5_13TeV-amcatnlo-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2430000_file_index.txt"]

#  SUSYGluGluToHToAA_AToMuMu_AToBB_M-40
elif args.code == "BSM_HAA":
    paths_to_root_file_list = ["/global/homes/r/rmastand/dimuonAD/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_SUSYGluGluToHToAA_AToMuMu_AToBB_M-40_TuneCP5_13TeV_madgraph_pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_40000_file_index.txt"]

#  ggXToYY_YToMuMu_M22p5
elif args.code == "BSM_XYY":
    paths_to_root_file_list = ["/global/homes/r/rmastand/dimuonAD/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_ggXToYY_YToMuMu_M22p5_JPCZeroPlusPlus_TuneCP5_13TeV-pythia8-JHUGen_NANOAODSIM_106X_mcRun2_asymptotic_v17-v2_40000_file_index.txt", "/global/homes/r/rmastand/dimuonAD/file_sources/CMS_mc_RunIISummer20UL16NanoAODv9_ggXToYY_YToMuMu_M22p5_JPCZeroPlusPlus_TuneCP5_13TeV-pythia8-JHUGen_NANOAODSIM_106X_mcRun2_asymptotic_v17-v2_2430000_file_index.txt"]

list_of_root_files = []
for path in paths_to_root_file_list:
    with open(path, "r") as infile:
        root_file_list = infile.readlines()
        list_of_root_files += root_file_list
        
print(len(list_of_root_files))
        
for i in range(int(args.start), int(args.stop)):
    
    loc_root_file_string = list_of_root_files[i]
    print("Analyzing file", i, loc_root_file_string)

    loc_root_file = uproot.open(loc_root_file_string)
    events = loc_root_file["Events;1"]

    # only store the muons for now
    muon_vars = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_pfRelIso03_all", "Muon_pfRelIso04_all"]
    jet_vars = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_nConstituents"]
    
    all_muon_data, filtered_muon_data, filtered_amuon_data = {}, {}, {}
    all_jet_data, filtered_jet_data = {}, {}
    for mv in muon_vars:
        all_muon_data[mv] = events[mv].array()
    for jv in jet_vars:
        all_jet_data[jv] = events[jv].array()

    # make filters for the muons, amuons
    loc_muon_filter = all_muon_data["Muon_charge"] == -1
    loc_amuon_filter = all_muon_data["Muon_charge"] == 1

    # TODO: Official CMS code restricts to events with exactly 2 muons, which is tighter than this cut
    loc_num_muon = np.sum(loc_muon_filter, axis = 1)
    loc_num_amuon = np.sum(loc_amuon_filter, axis = 1)

    loc_event_passes = (loc_num_muon > 0) & (loc_num_amuon > 0)

    # filter the muon data
    for mv in muon_vars:

        # filter the events
        loc_data_mu = all_muon_data[mv][loc_muon_filter][loc_event_passes]
        loc_data_amu = all_muon_data[mv][loc_amuon_filter][loc_event_passes]

        # get data corresponding to the hardest (a)muon
        filtered_muon_data[mv] = [event[0] for event in loc_data_mu]
        filtered_amuon_data[mv] = [event[0] for event in loc_data_amu]
        
    # filter the jet data
    for jv in jet_vars:

        # filter the events
        loc_data_jet = all_jet_data[jv][loc_event_passes]

        # get data corresponding to the hardest (a)muon
        filtered_jet_data[jv] = [event[0] if len(event) > 0 else np.NaN for event in loc_data_jet]

    # save out
    with open(f"{path_to_output}/filtered_mu_{args.code}_{i}", "wb") as output_file:
        pickle.dump(filtered_muon_data, output_file)
        
    with open(f"{path_to_output}/filtered_amu_{args.code}_{i}", "wb") as output_file:
        pickle.dump(filtered_amuon_data, output_file)

    with open(f"{path_to_output}/filtered_jet_{args.code}_{i}", "wb") as output_file:
        pickle.dump(filtered_jet_data, output_file)
       