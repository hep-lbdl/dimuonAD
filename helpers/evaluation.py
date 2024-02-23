import numpy as np
import torch

from scipy.stats import wasserstein_distance


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


def get_1d_wasserstein_distances(samples_1, samples_2):
    
    distances_1d = []
    for i in range(samples_1.shape[1]):
        distances_1d.append(wasserstein_distance(samples_1[:,i] , samples_2[:,i]))
    return distances_1d

