import numpy as np

def logit_transform(x, all_min, all_max):
    
    x_norm = (x-all_min)/(all_max-all_min)
    x_norm = 0.98*x_norm + 0.01
    #x_norm = x_norm[(x_norm != 0) & (x_norm != 1)]
    logit = np.log(x_norm/(1-x_norm+1e-10)+1e-10)
    #print(np.sum(np.isnan(logit)))
    #logit = logit[~np.isnan(logit)]
    return logit


def inverse_transform(x, preproc_info):
    
    unscaled_x =  x*preproc_info["std"] + preproc_info["mean"]
    tmp = np.exp(unscaled_x) / (1.0 + np.exp(unscaled_x))
    return tmp*(preproc_info["max"]-preproc_info["min"]) + preproc_info["min"]


def clean_data(x):
    
    remove_nan =  x[~np.isnan(x).any(axis=1)]
    remove_inf = remove_nan[~np.isinf(remove_nan).any(axis=1)]
    
    return remove_inf