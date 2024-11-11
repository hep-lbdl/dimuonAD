import numpy as np
import torch

import os

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from helpers.density_estimator import DensityEstimator
from scipy.stats import ks_2samp

def get_kl_dist(samp0, samp1):
    
    divs = []
    for i in range(samp0.shape[1]):
        divs.append(ks_2samp(samp0[:,i], samp1[:,i])[0])
    return divs

def get_median_percentiles(x_array):
    
    x_median = np.median(x_array, axis = 1)
    x_lower = np.percentile(x_array, 16, axis = 1)
    x_upper = np.percentile(x_array, 84, axis = 1)

    return x_median, x_lower, x_upper

def sample_from_flow(model_path, device, context, num_features):
    
    print(f"Loading in the best flow model ...")
    flow_best = torch.load(model_path)
    flow_best.to(device)

    # freeze the trained model
    for param in flow_best.parameters():
        param.requires_grad = False
    flow_best.eval()

    context_masses = torch.tensor(context.reshape(-1,1)).float().to(device)
    SB_samples = flow_best.sample(1, context=context_masses).detach().cpu().numpy()
    SB_samples = SB_samples.reshape(SB_samples.shape[0], num_features)

    SB_samples = np.hstack((SB_samples, np.reshape(context, (-1, 1))))
    
    return SB_samples

"""
def convert_to_latent_space(samples_to_convert, flow_training_dir, training_config_string, device):


    checkpoint_path = os.path.join(flow_training_dir, f"flow_best_model.pt")
    flow_best = torch.load(checkpoint_path)
    flow_best.to(device)

    # freeze the trained model
    for param in flow_best.parameters():
        param.requires_grad = False
    flow_best.eval()

    context_masses = torch.tensor(samples_to_convert[:,-1].reshape(-1,1)).float().to(device)
    outputs_normal_target, _ = flow_best._transform(torch.tensor(samples_to_convert[:,:-1]).float().to(device), context=context_masses)
    
    # note that the mass is saved out as well. Necessary for evaluating the test set
    return np.hstack([outputs_normal_target.detach().cpu().numpy(), samples_to_convert[:,-1].reshape(-1,1)])

"""

def convert_to_latent_space_true_cathode(samples_to_convert, num_inputs, flow_training_dir, config_file, device):
    
    val_losses = np.load(os.path.join(flow_training_dir, "flow_val_losses.npy"))

    # get epoch of best val loss
    best_epoch = np.argmin(val_losses) - 1
    model_path = f"{flow_training_dir}/flow_epoch_{best_epoch}.par"

    eval_model = DensityEstimator(config_file, num_inputs, eval_mode=True, load_path=model_path, device=device, verbose=False,bound=False)
    
    
    context_masses = torch.tensor(samples_to_convert[:,-1].reshape(-1,1)).float().to(device)
    
    outputs_normal_target = eval_model.model.forward(torch.tensor(samples_to_convert[:,:-1]).float().to(device), context_masses, mode='direct')[0]
    return np.hstack([outputs_normal_target.detach().cpu().numpy(), samples_to_convert[:,-1].reshape(-1,1)])
    


def assemble_banded_datasets(data_dict, feature_set, bands):
    
    banded_data = {}
    
    for b in bands:
        num_events_band = data_dict[b]["s_inj_data"]["dimu_mass"].shape[0]
        events_band = np.empty((num_events_band, len(feature_set)))
        for i, feat in enumerate(feature_set):
            # default test set
            events_band[:,i] = data_dict[b]["s_inj_data"][feat].reshape(-1,)
        banded_data[b] = events_band
        
    return banded_data





