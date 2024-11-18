import zfit
from zfit import z
import numpy as np

import torch

from helpers.physics_functions import bkg_fit_cubic, bkg_fit_quintic, bkg_fit_septic

def get_flow_samples(model, masses):
    with torch.no_grad():
        feats = model.model.sample(num_samples=masses.shape[0], cond_inputs=torch.tensor(masses.reshape(-1,1)).float()).detach().cpu().numpy()
    return np.hstack([feats, masses.reshape(-1,1)])

class CustomPDFCubic(zfit.pdf.ZPDF):
    _PARAMS = ("a0","a1","a2","a3")  # specify which parameters to take
    def _unnormalized_pdf(self, x):  # implement function
        data = z.unstack_x(x)
        return bkg_fit_cubic(data, self.params["a0"],self.params["a1"],self.params["a2"],self.params["a3"])

class CustomPDFQuintic(zfit.pdf.ZPDF):
    _PARAMS = ("a0","a1","a2","a3","a4","a5")  # specify which parameters to take
    def _unnormalized_pdf(self, x):  # implement function
        data = z.unstack_x(x)
        return bkg_fit_quintic(data, self.params["a0"],self.params["a1"],self.params["a2"],self.params["a3"],self.params["a4"],self.params["a5"])

class CustomPDFSeptic(zfit.pdf.ZPDF):
    _PARAMS = ("a0","a1","a2","a3","a4","a5","a6","a7")  # specify which parameters to take
    def _unnormalized_pdf(self, x):  # implement function
        data = z.unstack_x(x)
        return bkg_fit_septic(data, self.params["a0"],self.params["a1"],self.params["a2"],self.params["a3"],self.params["a4"],self.params["a5"],self.params["a6"],self.params["a7"])


def get_mass_samples(SR_left, SR_right, bkg_fit_type, n_SR_samples, popt_0):
    
    obs = zfit.Space("mass_inv", limits=(SR_left, SR_right))
    
    if bkg_fit_type == "cubic":
        custom_pdf = CustomPDFCubic(obs=obs,a0=popt_0[0],a1=popt_0[1],a2=popt_0[2],a3=popt_0[3])
    elif bkg_fit_type == "quintic":
        custom_pdf = CustomPDFQuintic(obs=obs,a0=popt_0[0],a1=popt_0[1],a2=popt_0[2],a3=popt_0[3],a4=popt_0[4],a5=popt_0[5])
    elif bkg_fit_type == "septic":
        custom_pdf = CustomPDFSeptic(obs=obs,a0=popt_0[0],a1=popt_0[1],a2=popt_0[2],a3=popt_0[3],a4=popt_0[4],a5=popt_0[5],a6=popt_0[6],a7=popt_0[7])

    return custom_pdf.sample(n=n_SR_samples)["mass_inv"].numpy()
    
    