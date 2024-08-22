import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""
PLOTTING KWARGS
"""
kwargs_dict_bands = {
               "SB":  {"density": True,"histtype": "step", "color":"blue", "label": "SB"},
               
               "SBL":  {"density": True,"histtype": "step", "color":"green","label": "SBL"},
               
               "SBH":  {"density": True, "histtype": "step", "color":"purple","label": "SBH"},
               
                "IBL":  {"density": True,"histtype": "step", "color":"orange","label": "IBL"},
               
               "IBH":  {"density": True, "histtype": "step", "color":"pink","label": "IBH"},
               
              "SR":  {"density": True, "histtype": "step",  "color":"red", "label": "SR"},
               
              "SB_samples":  {"density": True,   "histtype": "stepfilled",   "color":"blue",  "label": "SB samples", "alpha": 0.3},
               
               "SBL_samples":  {"density": True,  "histtype": "stepfilled",   "color":"green",  "label": "SBL samples",   "alpha": 0.3},
               
               "SBH_samples":  {"density": True,   "histtype": "stepfilled",   "color":"purple",    "label": "SBH samples",  "alpha": 0.3},
               
               "IBL_samples":  {"density": True,  "histtype": "stepfilled",   "color":"orange",  "label": "IBL samples",   "alpha": 0.3},
               
               "IBH_samples":  {"density": True,   "histtype": "stepfilled",   "color":"pink",    "label": "IBH samples",  "alpha": 0.3},
               
              "SR_samples":  {"density": True,  "histtype": "stepfilled",   "color":"red",    "label": "SR samples",  "alpha": 0.3}
}

kwargs_dict_dtype = {
                "DATA_nojet":  {"density": True, "histtype": "step", "color":"blue", "label": "DATA_nojet"},
      "DATA_jet":  {"density": True, "histtype": "step", "color":"blue", "label": "DATA_jet"},
    "DATA":  {"density": True, "histtype": "step", "color":"blue", "label": "DATA"},
               
              "SM_SIM":  {"density": True,  "histtype": "step", "color":"red", "label": "CMS SIM"},
            "cmssim":  {"density": True,  "histtype": "step", "color":"red", "label": "CMS SIM"},
    
     "s_inj_data":  {"density": True,  "histtype": "step", "color":"red", "label": "s_inj_data"},

             "BSM_XYY":  {"density": True,  "histtype": "step", "color":"green", "label": "BSM_XYY"},
    "wp_wzpythia_forcms_charge-mz50.0-mw40.0-mwp1000_full":  {"density": True,  "histtype": "step", "color":"green", "label": "BSM_XYY"},

             "BSM_HAA":  {"density": True,  "histtype": "step", "color":"orange", "label": "BSM_HAA"},
              }



feature_labels = {
                  "muon_pt": "$\mu$ $p_T$",
                  "amuon_pt": "$\overline{\mu}$ $p_T$",
                  "mu0_pt": "$\mu_0$ $p_T$",
                  "mu1_pt": "$\mu_1$ $p_T$",
                  "muon_eta": "$\mu$ $\eta$",
                  "amuon_eta": "$\overline{\mu}$ $\eta$",
                  "mu0_eta": "$\mu_0$ $\eta$",
                  "mu1_eta": "$\mu_1$ $\eta$",
                  "mu0_phi": "$\mu_0$ $\phi$",
                  "mu1_phi": "$\mu_1$ $\phi$",
                  "mu0_iso03": "$\mu_0$ isoR03",
                  "mu1_iso03": "$\mu_1$ isoR03",
                  "mu0_iso04": "$\mu_0$ isoR04",
                  "mu1_iso04": "$\mu_1$ isoR04",
                  "muon_iso03": "$\mu$ iso R03",
                  "amuon_iso03": "$\overline{\mu}$ iso R03",
                  "muon_iso04": "$\mu$ iso R04",
                  "amuon_iso04": "$\overline{\mu}$ iso R04",
                  "dijet_pt": "Dijet $p_T$",
                  "dijet_eta": "Dijet $\eta$",
                  "dijet_mass": "Dijet $M$",
                  "hardest_jet_pt": "Jet $p_T$",
                  "hardest_jet_eta": "Jet $\eta$",
                  "hardest_jet_phi": "Jet $\phi$",
                  "hardest_jet_mass": "Jet $M$",
                  "jet0_btag": "Hardest jet Jet_btagDeepB",
                  "hardest_jet_btag": "Hardest jet Jet_btagDeepB",
                  "jet1_btag": "Second jet Jet_btagDeepB",
                  "higgs_pt": "$h$ $p_T$",
                  "higgs_eta": "$h$ $\eta$",
                  "higgs_mass": "$h$ $M$",
                  "dimu_pt": "Dimu $p_T$",
                  "dimu_eta": "Dimu $\eta$",
                  "dimu_phi": "Dimu $\phi$",
                  "dimu_mass": "Dimu $M$",
                  "n_electrons": "Num. electrons",
                  "n_muons": "Num. muons",
                  "n_jets": "Num. Jets",
                  "mumu_deltaR": "$\mu\mu$ $\Delta R$",
                  "mumu_deltapT": "$\mu\mu$ $\Delta p_T$",
                  "dimujet_deltaR": "$\mu\mu$ Jet $\Delta R$",
    
        }


n_bins = 60
feature_bins = {
                "muon_pt": np.linspace(0, 120, n_bins), 
                "amuon_pt": np.linspace(0, 120, n_bins), 
                "mu0_pt": np.linspace(0, 120, n_bins), 
                "mu1_pt":np.linspace(0, 120, n_bins), 
                "muon_eta": np.linspace(-3, 3, n_bins), 
                "amuon_eta": np.linspace(-3, 3, n_bins), 
                "mu0_eta": np.linspace(-3, 3, n_bins), 
                "mu1_eta":np.linspace(-3, 3, n_bins), 
                "mu0_phi":np.linspace(-3.2, 3.2, n_bins), 
                "mu1_phi":np.linspace(-3.2, 3.2, n_bins), 
                "mu0_iso03": np.linspace(0, 1, n_bins),
                "mu1_iso03": np.linspace(0, 1, n_bins),
                "mu0_iso04": np.linspace(0, 1, n_bins),
                "mu1_iso04":np.linspace(0, 1, n_bins),
                "dijet_pt": np.linspace(0, 300, n_bins), 
                "dijet_eta": np.linspace(-6, 6, n_bins), 
                "dijet_mass": np.linspace(0, 100, n_bins), 
                "hardest_jet_pt": np.linspace(0, 300, n_bins), 
                "hardest_jet_eta": np.linspace(-6, 6, n_bins), 
                "hardest_jet_phi": np.linspace(-3.2, 3.2, n_bins), 
                "hardest_jet_mass": np.linspace(0, 300, n_bins), 
                "jet0_btag": np.linspace(0, 1, n_bins),
                "hardest_jet_btag": np.linspace(0, 1, n_bins),
                "jet1_btag": np.linspace(0, 1, n_bins), 
                "higgs_pt": np.linspace(0, 300, n_bins), 
                "higgs_eta": np.linspace(-6, 6, n_bins), 
                "higgs_mass": np.linspace(0, 300, n_bins),
                "dimu_pt": np.linspace(0, 300, n_bins), 
                "dimu_eta": np.linspace(-6, 6, n_bins), 
                "dimu_phi":np.linspace(-3.2, 3.2, n_bins), 
                "dimu_mass": np.linspace(0, 120, n_bins),
                "n_electrons": np.linspace(0, 10, 11),
                "n_muons":np.linspace(0, 10, 11),
                "n_jets":np.linspace(0, 10, 11),
                "mumu_deltaR":np.linspace(0, 2, n_bins),
                "mumu_deltapT":np.linspace(0, 100, n_bins),
                "dimujet_deltaR":np.linspace(0, 2, n_bins),
      }



"""
FUNCTIONS
"""

def hist_all_features_dict(data_dicts, data_labels, feature_set, kwargs_dict, scaled_features=False, plot_bound=3, image_path=None, yscale_log=False, nice_labels=True):
    
        
    scaled_feature_bins = [np.linspace(-plot_bound, plot_bound, n_bins) for i in range(len(feature_set))]   
        
    if image_path:
        p = PdfPages(f"{image_path}.pdf")

    for i, feat in enumerate(feature_set):
        fig = plt.figure(figsize = (5, 3))
        

        for j, data_dict in enumerate(data_dicts):
            if scaled_features:
                plt.hist(data_dict[feat], bins = scaled_feature_bins[i], **kwargs_dict[data_labels[j]])
            else:
                plt.hist(data_dict[feat], bins = feature_bins[feat], **kwargs_dict[data_labels[j]])
                
        if yscale_log:
            plt.yscale("log")
        
        if nice_labels:
            plt.xlabel(feature_labels[feat])
        plt.legend()
        plt.ylabel("Density")
        plt.tight_layout()
        if image_path:
            fig.savefig(p, format='pdf') 
        plt.show()
        
    if image_path: 
        p.close()
        
        
        
def hist_all_features_array(samples, labels, feature_set, plot_bound=3, yscale_log=False):
    scaled_feature_bins = [np.linspace(-plot_bound, plot_bound, 40) for i in range(len(feature_set))]   
    
    n_features = len(feature_set)
    fig, ax = plt.subplots(1, n_features, figsize = (3*n_features, 3))
        

    for i, feat in enumerate(feature_set):
        for j, samp in enumerate(samples):
            ax[i].hist(samp[:,i], bins = scaled_feature_bins[i], histtype = "step", density = True, label = labels[j])
         
        if yscale_log:
            ax[i].set_yscale("log")
        ax[i].set_xlabel(feat)
        ax[i].set_yticks([])
    plt.legend(loc = (0, 1))
    ax[0].set_ylabel("Density")
    plt.subplots_adjust(wspace=0)
    plt.show()
  