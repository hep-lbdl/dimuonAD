import zfit
from zfit import z
import numpy as np
import torch
import tensorflow as tf

def get_flow_samples(model, masses):
    with torch.no_grad():
        feats = model.model.sample(num_samples=masses.shape[0], cond_inputs=torch.tensor(masses.reshape(-1,1)).float()).detach().cpu().numpy()
    return np.hstack([feats, masses.reshape(-1,1)])


def get_mass_samples(SR_left, SR_right, bkg_fit_degree, n_SR_samples, popt_0):
    
    obs = zfit.Space("mass_inv", limits=(SR_left, SR_right))
    
    class CustomPDF(zfit.pdf.ZPDF):
        _PARAMS = tuple([f"a{i}" for i in range(bkg_fit_degree+1)])
        def _unnormalized_pdf(self, x):  # implement function
            data = z.unstack_x(x)
            y = tf.zeros_like(data)
            for i in range(bkg_fit_degree + 1):
                y += self.params[f"a{i}"] * (data)**i
            return y

    extra_args = {f"a{i}":popt_0[i] for i in range(bkg_fit_degree+1)}
    custom_pdf = CustomPDF(obs=obs, **extra_args)
    
    return custom_pdf.sample(n=n_SR_samples)["mass_inv"].numpy()
    
    