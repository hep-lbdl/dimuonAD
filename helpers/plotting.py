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
               
              "cmssim":  {"density": True,  "histtype": "step", "color":"red", "label": "CMS SIM"},
              }



feature_labels = {
                  0: "jet $p_T$",
                  1: "jet $\eta$",
                  2: "jet $\phi$",
                  3: "jet $M$",
                  4: "jet $\\tau_{21}$",
                  5: "$\mu$ iso $R$ = 0.3",
                  6: "$\mu$ iso $R$ = 0.5",
                  7: "$\overline{\mu}$ iso $R$ = 0.3",
                  8: "$\overline{\mu}$ iso $R$ = 0.5",
                  9: "$\mu\mu$ $p_T$",
                  10: "$\mu\mu$ $\eta$",
                  11: "$\mu\mu$ $\phi$",
                  12: "$\mu\mu$ $M$",
        }

n_bins = 60

feature_bins = {
                0: np.linspace(0, 300, n_bins), # jet pt
                1: np.linspace(-6, 6, n_bins), # jet eta
                2: np.linspace(-3.2, 3.2, n_bins), # jet phi
                3: np.linspace(20, 120, n_bins), # jet M
                4: np.linspace(0, 1, n_bins), # jet tau21
                5: np.linspace(0, 2.5, n_bins), # mu R = 0.3
                6: np.linspace(0, 2.5, n_bins), # mu R = 0.5
                7: np.linspace(0, 2.5, n_bins), # amu R = 0.3
                8: np.linspace(0, 2.5, n_bins), # amu R = 0.5
                9: np.linspace(0, 800, n_bins), # dimuon pt
                10: np.linspace(-6, 6, n_bins), # dimuon eta
                11: np.linspace(-3.2, 3.2, n_bins), # dimuon phi
                12: np.linspace(20, 120, n_bins), # dimuon M
      }

scaled_feature_bins = [np.linspace(-4, 4, n_bins) for i in range(13)]

"""
FUNCTIONS
"""

def hist_all_features(codes_to_plot, data_dict, feature_set, kwargs_dict, scaled_features = False, image_path=None):
    
    """
    subset: True if the feature set is a true subset of the provided data. I.e. the data contains all the features, but you only want to plot a few
            False if the dataset contains all of the features to plot
    """
    if data_dict[codes_to_plot[0]].shape[-1] != len(feature_set):
        subset = True
    else:
        subset = False
    
        
    if image_path:
        p = PdfPages(f"{image_path}.pdf")

    for i, feat in enumerate(feature_set):
        fig = plt.figure(figsize = (5, 3))
        
        if subset:
            plotting_index = feat
        else:
            plotting_index = i
        
        for code in codes_to_plot:
            if scaled_features:
                plt.hist(data_dict[code][:,plotting_index], bins = scaled_feature_bins[feat], **kwargs_dict[code])
            else:
                plt.hist(data_dict[code][:,plotting_index], bins = feature_bins[feat], **kwargs_dict[code])
        plt.xlabel(feature_labels[feat])
        plt.legend()
        plt.ylabel("Density")
        if image_path:
            fig.savefig(p, format='pdf') 
        plt.show()
        
    if image_path: 
        p.close()