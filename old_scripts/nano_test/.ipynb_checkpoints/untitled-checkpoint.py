import uproot
import awkward as ak
import vector

import matplotlib.pyplot as plt
import numpy as np
plt.style.use("science.mplstyle")



file_sources = ["root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/127C2975-1B1C-A046-AABF-62B77E757A86.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/183BFB78-7B5E-734F-BBF5-174A73020F89.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/1BE226A3-7A8D-1B43-AADC-201B563F3319.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/1DE780E2-BCC2-DC48-815D-9A97B2A4A2CD.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/21DA4CE5-4E50-024F-9CE1-50C77254DD4E.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/2C6A0345-8E2E-9B41-BB51-DB56DFDFB89A.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/3676E287-A650-8F44-BBCB-3B8556966406.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/411A019C-7058-FD42-AD50-DE74433E6859.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/46A8960A-E58F-4648-9C12-2708FE7C12FB.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/4F0B53A7-6440-924B-AF48-B5B61D3CE23F.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/790F8A75-8256-3B46-8209-850DE0BE3C77.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/7F53D1DE-439E-AD48-871E-D3458DABA798.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/8A696857-C147-B04A-905A-F85FB76EDA23.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/8B253755-51F2-CB49-A4B6-C79637CAE23F.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/9528EA75-1C0B-9047-A9A3-6A47564F7A98.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/A6605227-0B58-864E-8422-B8990D18F622.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/B2DC29E0-8679-1D4F-A5AE-E7D0284A20D4.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/B450B2B3-BEF8-8C43-82BF-7AD0EF2EA7EA.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/B7AA7F04-5D5F-514A-83A6-9A275198852C.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/B93B57BF-4239-A049-9531-4C542C370185.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/C4558F81-9F2C-1349-B528-6B9DD6838D6D.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/C8CFC890-D4B8-8A4F-8699-C6ACCDF1620A.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/CAA285FF-7A12-F945-9183-DC7042178535.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/CD267D88-E57D-3B44-AC45-0712E2E12B87.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/E7C51551-7A75-5C41-B468-46FB922F36A9.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/EBC200F4-C06F-CE45-BAAA-7CAECDD3076F.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/EEB2FE3F-7CF3-BF4A-9F70-3F89FACE698E.root","root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/F5E234F9-1E9C-0042-B395-AB6407E4A336.root"]




total_num_events = 0
passed_events = 0

for i, root_file in enumerate(file_sources):
    

    print(root_file)
    
    events = uproot.open(root_file)['Events;1']
    
    loc_pt = events["Muon_pt"].array()
    loc_eta = events["Muon_eta"].array()
    loc_phi = events["Muon_phi"].array()
    loc_charge = events["Muon_charge"].array()
    
    total_num_events += len(loc_pt)
    
    loc_muon_filter = loc_charge == -1
    loc_amuon_filter = loc_charge == 1
    
    loc_num_muon = np.sum(loc_muon_filter, axis = 1)
    loc_num_amuon = np.sum(loc_amuon_filter, axis = 1)
    
    loc_event_passes = (loc_num_muon > 0) & (loc_num_amuon > 0)
    
    
    def loc_get_hardest_mu_amu(data_array):
    
        # filter the events
        loc_data_mu = data_array[loc_muon_filter][loc_event_passes]
        loc_data_amu = data_array[loc_amuon_filter][loc_event_passes]

        # get data corresponding to the hardest (a)muon
        loc_data_mu = [event[0] for event in loc_data_mu]
        loc_data_amu = [event[0] for event in loc_data_amu]

        return loc_data_mu, loc_data_amu
 
    loc_filtered_mu_pt, loc_filtered_amu_pt = loc_get_hardest_mu_amu(loc_pt)

    loc_filtered_mu_eta, loc_filtered_amu_eta = loc_get_hardest_mu_amu(loc_eta)
    loc_filtered_mu_phi, loc_filtered_amu_phi = loc_get_hardest_mu_amu(loc_phi)
    
    passed_events += len(loc_filtered_mu_pt)
    
    np.save(f"nano_test/filtered_mu_pt_{i}.npy", loc_filtered_mu_pt)
    np.save(f"nano_test/filtered_amu_pt_{i}.npy", loc_filtered_amu_pt)
    np.save(f"nano_test/filtered_mu_eta_{i}.npy", loc_filtered_mu_eta)
    np.save(f"nano_test/filtered_amu_eta_{i}.npy", loc_filtered_amu_eta)
    np.save(f"nano_test/filtered_mu_phi_{i}.npy", loc_filtered_mu_phi)
    np.save(f"nano_test/filtered_amu_phi_{i}.npy", loc_filtered_amu_phi)
    
    print(i, "total events", total_num_events, "passed events", passed_events)

