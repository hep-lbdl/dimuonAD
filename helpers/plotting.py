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
                "od":  {"density": True, "histtype": "step", "color":"blue", "label": "OD"},
               
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
                  "muon_eta": "$\mu$ $\eta$",
                  "amuon_eta": "$\overline{\mu}$ $\eta$",
                  "muon_iso03": "$\mu$ iso R03",
                  "amuon_iso03": "$\overline{\mu}$ iso R03",
                  "muon_iso04": "$\mu$ iso R04",
                  "amuon_iso04": "$\overline{\mu}$ iso R04",
                  "dijet_pt": "Dijet $p_T$",
                  "dijet_eta": "Dijet $\eta$",
                  "dijet_mass": "Dijet $M$",
                  "jet_pt": "Jet $p_T$",
                  "jet_eta": "Jet $\eta$",
                  "jet_mass": "Jet $M$",
                  "jet0_btag": "Hardest jet Jet_btagDeepB",
                  "jet1_btag": "Second jet Jet_btagDeepB",
                  "higgs_pt": "$h$ $p_T$",
                  "higgs_eta": "$h$ $\eta$",
                  "higgs_mass": "$h$ $M$",
                  "dimu_pt": "Dimu $p_T$",
                  "dimu_eta": "Dimu $\eta$",
                  "dimu_mass": "Dimu $M$",
        }

n_bins = 60
feature_bins = {
                "muon_pt": np.linspace(0, 120, n_bins), 
                "amuon_pt": np.linspace(0, 120, n_bins), 
                "muon_eta": np.linspace(-3, 3, n_bins), 
                "amuon_eta": np.linspace(-3, 3, n_bins), 
                "muon_iso03": np.linspace(0, 1, n_bins),
                "amuon_iso03": np.linspace(0, 1, n_bins),
                "muon_iso04": np.linspace(0, 1, n_bins),
                "amuon_iso04":np.linspace(0, 1, n_bins),
                "dijet_pt": np.linspace(0, 300, n_bins), 
                "dijet_eta": np.linspace(-6, 6, n_bins), 
                "dijet_mass": np.linspace(0, 100, n_bins), 
                "jet_pt": np.linspace(0, 300, n_bins), 
                "jet_eta": np.linspace(-6, 6, n_bins), 
                "jet_mass": np.linspace(0, 300, n_bins), 
                "jet0_btag": np.linspace(0, 1, n_bins),
                "jet1_btag": np.linspace(0, 1, n_bins), 
                "higgs_pt": np.linspace(0, 300, n_bins), 
                "higgs_eta": np.linspace(-6, 6, n_bins), 
                "higgs_mass": np.linspace(0, 300, n_bins),
                "dimu_pt": np.linspace(0, 300, n_bins), 
                "dimu_eta": np.linspace(-6, 6, n_bins), 
                "dimu_mass": np.linspace(0, 120, n_bins),
      }



"""
FUNCTIONS
"""

def hist_all_features(codes_to_plot, data_dict, feature_set, kwargs_dict, scaled_features=False, plot_bound=3, image_path=None, yscale_log=False):
    
        
    scaled_feature_bins = [np.linspace(-plot_bound, plot_bound, n_bins) for i in range(len(feature_set))]   
        
    if image_path:
        p = PdfPages(f"{image_path}.pdf")

    for i, feat in enumerate(feature_set):
        fig = plt.figure(figsize = (5, 3))
        
        
        for code in codes_to_plot:
            if scaled_features:
                plt.hist(data_dict[code][feat], bins = scaled_feature_bins[i], **kwargs_dict[code])
            else:
                plt.hist(data_dict[code][feat], bins = feature_bins[feat], **kwargs_dict[code])
                
        if yscale_log:
            plt.yscale("log")
            
        plt.xlabel(feature_labels[feat])
        plt.legend()
        plt.ylabel("Density")
        plt.tight_layout()
        if image_path:
            fig.savefig(p, format='pdf') 
        plt.show()
        
    if image_path: 
        p.close()