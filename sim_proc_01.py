#!/usr/bin/env python
# coding: utf-8

# # Simulation processing: step 1

# Goal: take the simulated hadrons and create an output file for fj clustering

# In[1]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import vector


# In[2]:


simulation_dir = '/global/cfs/cdirs/m3246/ewitkowski/delphes_data'
#sim_code = "wp_wzpythia_forcms_1k-mz50.0-mw40.0-mwp1000_8000199"
#sim_code = "wp_wzpythia_forcms_1k-mz70.0-mw60.0-mwp1200_8000200"
sim_code = "zmm_forcms_1k-mz90.1-mw80.4_full"


# In[3]:


selected_tower_ET = h5py.File(f'{simulation_dir}/{sim_code}_Tower_ET.h5', 'r')['values']
selected_tower_eta = h5py.File(f'{simulation_dir}/{sim_code}_Tower_Eta.h5', 'r')['values']
selected_tower_phi = h5py.File(f'{simulation_dir}/{sim_code}_Tower_Phi.h5', 'r')['values']
selected_tower_E = h5py.File(f'{simulation_dir}/{sim_code}_Tower_E.h5', 'r')['values']

selected_muon_pT = h5py.File(f'{simulation_dir}/{sim_code}_Muon_PT.h5', 'r')['values']
selected_muon_eta = h5py.File(f'{simulation_dir}/{sim_code}_Muon_Eta.h5', 'r')['values']
selected_muon_phi = h5py.File(f'{simulation_dir}/{sim_code}_Muon_Phi.h5', 'r')['values']


# In[4]:


muon_mass = 0.1056583755 # GeV

start_read, stop_read, chunk_size = 0, 50000, 10000
update_freq = int(chunk_size/10.0)




# In[6]:


current_chunk_start = start_read

while current_chunk_start < stop_read:
    
    current_chunk_stop = current_chunk_start + chunk_size
    print(f"Processing chunk from {current_chunk_start} to {current_chunk_stop}...")

    outfile_dimuons = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_post_fj/muons_only_{current_chunk_start}_{current_chunk_stop}_{sim_code}.dat"
    outfile_hadrons = f"/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_pre_fj/hadrons_only_{current_chunk_start}_{current_chunk_stop}_{sim_code}.dat"

    ofile_muons = open(outfile_dimuons, "w")
    ofile_hadrons = open(outfile_hadrons, "w")
    num_rejects = 0
   
    for event in range(current_chunk_start, current_chunk_stop):

        if event % update_freq == 0:
            print(f"On event {event}...")

        # get the nonmuons
        loc_E = selected_tower_E[event]
        loc_eta = selected_tower_eta[event][loc_E != 0]
        loc_phi = selected_tower_phi[event][loc_E != 0]
        loc_ET = selected_tower_ET[event][loc_E != 0]
        loc_E = loc_E[loc_E != 0]    

        # get the muons
        loc_mu_pt = selected_muon_pT[event]
        loc_mu_eta = selected_muon_eta[event][loc_mu_pt != 0]
        loc_mu_phi = selected_muon_phi[event][loc_mu_pt != 0]
        loc_mu_pt = loc_mu_pt[loc_mu_pt != 0]
        
        
        if len(loc_mu_pt) != 2:
            num_rejects += 1
        
        else:
            ofile_muons.write("#BEGIN\n")
            ofile_hadrons.write("#BEGIN\n")

            # get the muons
            # construct the muon 4-vector
            mu_1 = vector.obj(pt = loc_mu_pt[0], eta = loc_mu_eta[0], phi = loc_mu_phi[0], M = muon_mass)
            mu_2 = vector.obj(pt = loc_mu_pt[1], eta = loc_mu_eta[1], phi = loc_mu_phi[1], M = muon_mass)
            dimu_system = mu_1 + mu_2

            ofile_muons.write(f"{dimu_system.pt} {dimu_system.eta} {dimu_system.phi} {dimu_system.M}\n")

            # get the hadrons
            for particle_i in range(len(loc_E)):
                particle_vector = vector.obj(E = loc_E[particle_i], eta = loc_eta[particle_i], phi = loc_phi[particle_i], pt = loc_ET[particle_i])
                ofile_hadrons.write(f"{particle_vector.px} {particle_vector.py} {particle_vector.pz} {particle_vector.E}\n")

            ofile_muons.write("#END\n")
            ofile_hadrons.write("#END\n")

    ofile_muons.close()  
    ofile_hadrons.close()
    
    print(f"Done processing chunk.")
    print(f"{num_rejects} events without 2 muons")
    print("\n")
    
    current_chunk_start += chunk_size

print("Done completely!")



                                         


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




