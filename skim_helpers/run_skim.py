import os
import argparse

# source: https://opendata.cern.ch/record/31305 for root
# https://opendata.cern.ch/record/14220 for valid runs

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--i", help="increase output verbosity")
args = parser.parse_args()


working_dir = "/global/homes/r/rmastand/dimuonAD/" # you will probably need to change this
path_to_root_output = "/pscratch/sd/r/rmastand/dimuonAD/skimmed_root_files/" # and this

path_to_root_file_list = f"{working_dir}/skim_helpers/DoubleMuon_PFNano_29-Feb-24_Run2016G-UL2016_MiniAODv2_PFNanoAODv1_root_file_index.txt"
with open(path_to_root_file_list, "r") as infile:
    root_file_list = infile.readlines()
    
path_to_valid_runs = f"{working_dir}/skim_helpers/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

    
for i in range(int(args.i)):
    root_file_to_run = root_file_list[i]
    print(f"Skimming root file {i} of {args.i}...")
    command = f"{working_dir}/NanoAODTools/scripts/nano_postproc.py -J {path_to_valid_runs} -b {working_dir}/skim_helpers/select_branches.txt {path_to_root_output} {root_file_to_run}"
    os.system(command)
    #print(command)
    print(3*"\n")
    

