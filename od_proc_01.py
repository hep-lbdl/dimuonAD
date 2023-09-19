#!/usr/bin/env python
# coding: utf-8

# # Open data processing: step 1

# Take Ed's processed data files and split the PFCs by their ids.
# 
# PDG codes (from https://cms-opendata-workshop.github.io/workshop2023-lesson-advobjects/02-particleflow/index.html):
# 
# - 11, 13 = electron, muon
# - 22 = photon
# - 130 = neutral hadron
# - 211 = charged hadron
# - 1 = hadronic particle reconstructed in the forward calorimeters
# - 2 = electromagnetic particle reconstructed in the forward calorimeters
# 
# We will split as:
# - Muons (2 / event)
# - Hadrons + photons, to be routed to fastjet for clustering
# - Electrons, to trash

# In[6]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import vector


# In[7]:


selected_data_dir = '/global/cfs/cdirs/m3246/ewitkowski/selected_data'
selected_pT = h5py.File(f'{selected_data_dir}/recoPFCandidates_particleFlow__RECO_obj_pt_.h5', 'r')['values']
selected_eta = h5py.File(f'{selected_data_dir}/recoPFCandidates_particleFlow__RECO_obj_eta_.h5', 'r')['values']
selected_phi = h5py.File(f'{selected_data_dir}/recoPFCandidates_particleFlow__RECO_obj_phi_.h5', 'r')['values']
selected_pdgId = h5py.File(f'{selected_data_dir}/recoPFCandidates_particleFlow__RECO_obj_pdgId_.h5', 'r')['values']


# In[8]:


muon_mass = 0.1056583755 # GeV
particles_to_fastjet = [22, 130, 211, 1, 2]

start_read, stop_read, chunk_size = 0, 500000, 10000
update_freq = int(chunk_size/10.0)


# Muons file:  pt eta phi M
# 
# Hadrons file: px py pz E

# In[ ]:


current_chunk_start = start_read

while current_chunk_start < stop_read:
    
    current_chunk_stop = current_chunk_start + chunk_size
    print(f"Processing chunk from {current_chunk_start} to {current_chunk_stop}...")

    outfile_dimuons = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_post_fj/muons_only_{current_chunk_start}_{current_chunk_stop}_od.dat"
    outfile_hadrons = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_pre_fj/hadrons_only_{current_chunk_start}_{current_chunk_stop}_od.dat"

    ofile_muons = open(outfile_dimuons, "w")
    ofile_hadrons = open(outfile_hadrons, "w")
   
    for event in range(current_chunk_start, current_chunk_stop):

        if event % update_freq == 0:
            print(f"On event {event}...")

        # get the nonzero entries
        loc_pid = selected_pdgId[event]
        loc_pt = selected_pT[event][loc_pid != 0]
        loc_eta = selected_eta[event][loc_pid != 0]
        loc_phi = selected_phi[event][loc_pid != 0]
        loc_pid = loc_pid[loc_pid != 0]    
        num_particles = len(loc_pid)
        #print(f"Event {event} has {num_particles} particles.")

        ofile_muons.write("#BEGIN\n")
        ofile_hadrons.write("#BEGIN\n")

        # get the muons
        muon_ids = np.where(np.abs(loc_pid) == 13)[0]
        # construct the muon 4-vector
        mu_1 = vector.obj(pt = loc_pt[muon_ids[0]], eta = loc_eta[muon_ids[0]], phi = loc_phi[muon_ids[0]], M = muon_mass)
        mu_2 = vector.obj(pt = loc_pt[muon_ids[1]], eta = loc_eta[muon_ids[1]], phi = loc_phi[muon_ids[1]], M = muon_mass)
        dimu_system = mu_1 + mu_2

        ofile_muons.write(f"{dimu_system.pt} {dimu_system.eta} {dimu_system.phi} {dimu_system.M}\n")

        # get the hadrons
        for particle_i in range(len(loc_pt)):
            if np.abs(loc_pid[particle_i]) in particles_to_fastjet:
                particle_vector = vector.obj(pt = loc_pt[particle_i], eta = loc_eta[particle_i], phi = loc_phi[particle_i], M = 0)
                ofile_hadrons.write(f"{particle_vector.px} {particle_vector.py} {particle_vector.pz} {particle_vector.E}\n")

        ofile_muons.write("#END\n")
        ofile_hadrons.write("#END\n")

    ofile_muons.close()  
    ofile_hadrons.close()
    
    print(f"Done processing chunk.")
    print("\n")
    
    current_chunk_start += chunk_size

print("Done completely!")

