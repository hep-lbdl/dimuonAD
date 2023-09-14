import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def hist_all_features(codes_to_plot, data_dict, kwargs_dict, feature_labels, feature_bins, image_path=None):
    

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