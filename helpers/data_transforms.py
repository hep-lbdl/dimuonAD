import numpy as np

epsilon = 1e-10

def logit_transform(x, all_min, all_max):
    
    x_norm = (x-all_min)/(all_max-all_min)
    x_norm = 0.98*x_norm + 0.01
    #x_norm = x_norm[(x_norm != 0) & (x_norm != 1)]
    logit = np.log(x_norm/(1.0-x_norm+epsilon)+epsilon)
    #print(np.sum(np.isnan(logit)))
    #logit = logit[~np.isnan(logit)]
    return logit


def scaled_to_physical_transform(scaled_x, preproc_info):
    
    unscaled_x =  scaled_x*preproc_info["std"] + preproc_info["mean"]
    #inverse logit
    x_norm = np.exp(unscaled_x) / (1.0 + np.exp(unscaled_x))
    x_norm = (x_norm - 0.01) / 0.98
    
    return x_norm*(preproc_info["max"]-preproc_info["min"]) + preproc_info["min"]

def unscale_mass(scaled_x, SB_left, SB_right):
    
    unscaled_x =  scaled_x*preproc_info["std"] + preproc_info["mean"]
    #inverse logit
    x_norm = np.exp(unscaled_x) / (1.0 + np.exp(unscaled_x))
    x_norm = (x_norm - 0.01) / 0.98
    
    return x_norm*(preproc_info["max"]-preproc_info["min"]) + preproc_info["min"]


def clean_data(x):
    
    remove_nan =  x[~np.isnan(x).any(axis=1)]
    remove_inf = remove_nan[~np.isinf(remove_nan).any(axis=1)]
    
    return remove_inf

def bootstrap_array(data_array, seed):
    np.random.seed(seed)
    indices_to_take = np.random.choice(range(data_array.shape[0]), size = data_array.shape[0], replace = True) 
    return data_array[indices_to_take]



