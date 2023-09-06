import numpy as np

from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal

from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.lu import LULinear


def assemble_masked_AR_transforms(num_features, num_layers, num_hidden_features, num_blocks, num_bins = 10, spline_type = "PiecewiseRationalQuadratic", perm_type = "Reverse", tail_bound = 1.5, tails = "linear", dropout = 0.05):
    
    transforms = []
    
    for _ in range(num_layers):
        
        if spline_type == "PiecewiseRationalQuadratic":
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features = num_features, hidden_features = num_hidden_features, num_blocks = num_blocks, tail_bound = tail_bound, context_features = 1, tails = tails, num_bins = num_bins, dropout_probability = dropout))     
            
        if perm_type == "Reverse":
            transforms.append(ReversePermutation(features = num_features)) 
        else:
            transforms.append(LULinear(features = num_features)) 
            
    return transforms

def make_masked_AR_flow(num_features, num_layers, num_hidden_features, num_blocks):

    # Define a flow architecture
    transforms = assemble_masked_AR_transforms(num_features, num_layers, num_hidden_features, num_blocks)
    base_dist = StandardNormal(shape=[num_features])
    
    flow = Flow(CompositeTransform(transforms), base_dist)
    
    return flow



