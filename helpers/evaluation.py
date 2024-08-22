import numpy as np
import torch

import os

from scipy.stats import wasserstein_distance

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score



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


def convert_to_latent_space(samples_to_convert, flow_training_dir, training_config_string, device):


    checkpoint_path = os.path.join(flow_training_dir, f"{training_config_string}_best_model.pt")
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



def get_1d_wasserstein_distances(samples_1, samples_2):
    
    distances_1d = []
    for i in range(samples_1.shape[1]):
        distances_1d.append(wasserstein_distance(samples_1[:,i] , samples_2[:,i]))
    return distances_1d
