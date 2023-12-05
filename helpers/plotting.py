import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""
PLOTTING KWARGS
"""
kwargs_dict = {"SB":  {"density": True,"histtype": "step", "color":"blue", "label": "SB"},
               
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
               
              "SR_samples":  {"density": True,  "histtype": "stepfilled",   "color":"red",    "label": "SR samples",  "alpha": 0.3}}


"""
FUNCTIONS
"""

def hist_all_features(codes_to_plot, data_dict, kwargs_dict, feature_bins, image_path=None):
    
    # these feature labels will almost certainly not change
    feature_labels = ["jet $p_T$", "jet $\eta$", "jet $\phi$", "jet $M$", "jet $\\tau_{21}$","$\mu\mu$ $p_T$", "$\mu\mu$ $\eta$", "$\mu\mu$ $\phi$", "$\mu\mu$ $M$"]
    
    if image_path:
        p = PdfPages(f"{image_path}.pdf")

    for i in range(9):
        fig = plt.figure()
        
        for code in codes_to_plot:
            plt.hist(data_dict[code][:,i], bins = feature_bins[i], **kwargs_dict[code])

        plt.xlabel(feature_labels[i])
        plt.legend()
        plt.ylabel("Density")
        if image_path:
            fig.savefig(p, format='pdf') 
        plt.show()
        
    if image_path: 
        p.close()