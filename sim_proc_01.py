#!/usr/bin/env python
# coding: utf-8

# # Simulation processing: step 1

# Goal: take the simulated hadrons and create an output file for fj clustering

# In[1]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import vector
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-start","--start_read",help="start_read",type=int)
parser.add_argument("-stop","--stop_read",help="stop_read",type=int)

args = parser.parse_args()


# In[2]:


simulation_dir = '/global/cfs/cdirs/m3246/ewitkowski/cmssim'

tower_prefix = "recoPFCandidates_particleFlow__RECO_obj"
tower_vars = ["eta_", "pdgId_", "phi_", "pt_"]


# In[3]:

tower_dict = {}

for tower_var in tower_vars:
    tower_dict[tower_var] = h5py.File(f'{simulation_dir}/{tower_prefix}_{tower_var}.h5', 'r')['values']


# In[4]:


muon_mass = 0.1056583755 # GeV
particles_to_fastjet = [22, 130, 211, 1, 2]

chunk_size = 10000
update_freq = int(chunk_size/10.0)

delta_R_isos = [0.3, 0.5]


# In[6]:


current_chunk_start = args.start_read

while current_chunk_start < args.stop_read:
    
    current_chunk_stop = current_chunk_start + chunk_size
    print(f"Processing chunk from {current_chunk_start} to {current_chunk_stop}...")

    outfile_dimuons = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_post_fj/muons_only_{current_chunk_start}_{current_chunk_stop}_cmssim.dat"
    outfile_dimuons_isolation = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_post_fj/muons_iso_{current_chunk_start}_{current_chunk_stop}_cmssim.dat"
    outfile_hadrons = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_pre_fj/hadrons_only_{current_chunk_start}_{current_chunk_stop}_cmssim.dat"

    ofile_muons = open(outfile_dimuons, "w")
    ofile_iso = open(outfile_dimuons_isolation, "w")
    ofile_hadrons = open(outfile_hadrons, "w")
    num_rejects = 0
   
    for event in range(current_chunk_start, current_chunk_stop):

        if event % update_freq == 0:
            print(f"On event {event}...")

        # get the nonmuons
        loc_particle_exists = (tower_dict["pt_"][event] != 0)
        
        loc_tower = {}
        for tower_var in tower_vars:
            loc_tower[tower_var] = tower_dict[tower_var][event][loc_particle_exists]

        # check that there is at least one muon and one antimuon
        num_muons = np.sum(loc_tower["pdgId_"]==13)
        num_amuons = np.sum(loc_tower["pdgId_"]==-13)
        
        if (num_muons < 1) or (num_amuons < 1):
            num_rejects += 1
            
        else:
            # write out the event
            
            # find the hardest muon / antimuon
            loc_mu_pt = loc_tower["pt_"][loc_tower["pdgId_"]==13]
            max_muon_pt = np.max(loc_mu_pt)
            hardest_muon_id = np.where(loc_tower["pt_"]==max_muon_pt)[0][0]
            
            loc_amu_pt = loc_tower["pt_"][loc_tower["pdgId_"]==-13]
            max_amuon_pt = np.max(loc_amu_pt)
            hardest_amuon_id = np.where(loc_tower["pt_"]==max_amuon_pt)[0][0]
      
            
            # sanity check -- just in case there are 2 particles with the same pt...                
            if (loc_tower["pdgId_"][hardest_muon_id] != 13):
                for candidate_id in np.where(loc_tower["pt_"]==max_muon_pt)[0]:
                    if loc_tower["pdgId_"][candidate_id] == 13:
                        hardest_muon_id = candidate_id
                        pass
                        
            if (loc_tower["pdgId_"][hardest_amuon_id] != -13):
                for candidate_id in np.where(loc_tower["pt_"]==max_amuon_pt)[0]:
                    if loc_tower["pdgId_"][candidate_id] == -13:
                        print("passed")
                        hardest_amuon_id = candidate_id
                        pass
            
            
            ofile_muons.write("#BEGIN\n")
            ofile_iso.write("#BEGIN\n")
            ofile_hadrons.write("#BEGIN\n")

            # get the muons
            # construct the muon 4-vector
            mu_1 = vector.obj(pt = loc_tower["pt_"][hardest_muon_id], eta = loc_tower["eta_"][hardest_muon_id], phi = loc_tower["phi_"][hardest_muon_id], M = muon_mass)
            mu_2 = vector.obj(pt = loc_tower["pt_"][hardest_amuon_id], eta = loc_tower["eta_"][hardest_amuon_id], phi = loc_tower["phi_"][hardest_amuon_id], M = muon_mass)
            dimu_system = mu_1 + mu_2

            ofile_muons.write(f"{dimu_system.pt} {dimu_system.eta} {dimu_system.phi} {dimu_system.M}\n")

            # get the hadrons and calculate muon isolation
            isolations_mu, isolations_amu = {R:0 for R in delta_R_isos}, {R:0 for R in delta_R_isos}

            for particle_i in range(len(loc_tower["pt_"])):
                if np.abs(loc_tower["pdgId_"][particle_i]) in particles_to_fastjet:
                                        
                    # write out the particle
                    particle_vector = vector.obj(pt = loc_tower["pt_"][particle_i], eta = loc_tower["eta_"][particle_i], phi = loc_tower["phi_"][particle_i], M = 0)
                    ofile_hadrons.write(f"{particle_vector.px} {particle_vector.py} {particle_vector.pz} {particle_vector.E}\n")
                    
                    # calculate the isolation contribution to the hardest (a)muon
                    delta_R_mu = mu_1.deltaR(particle_vector)
                    delta_R_amu = mu_2.deltaR(particle_vector)
                    
                    for R in delta_R_isos:
                        if delta_R_mu <= R:
                            isolations_mu[R] += (particle_vector.pt)/(mu_1.pt)
                        if delta_R_amu <= R:
                            isolations_amu[R] += (particle_vector.pt)/(mu_2.pt)
                            
            iso_muons_line = ""
            for R in delta_R_isos:
                iso_muons_line += str(isolations_mu[R]) + " " 
            for R in delta_R_isos:
                iso_muons_line += str(isolations_amu[R]) + " " 
                   
            ofile_iso.write(f"{iso_muons_line}\n")

            ofile_muons.write("#END\n")
            ofile_iso.write("#END\n")
            ofile_hadrons.write("#END\n")


    ofile_muons.close()  
    ofile_iso.close() 
    ofile_hadrons.close()
    
    print(f"Done processing chunk.")
    print(f"{num_rejects} events without 2 muons")
    print("\n")
    
    current_chunk_start += chunk_size

print("Done completely!")



                                         