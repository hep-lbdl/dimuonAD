import numpy as np


muon_mass = 0.1056583755 # GeV

def assemble_m_inv(a_M, a_pt, a_eta, a_phi, b_M, b_pt, b_eta, b_phi):
    # computes system of mother particle
    
    a_E = np.sqrt(a_M**2 + (a_pt*np.cosh(a_eta))**2)
    b_E = np.sqrt(b_M**2 + (b_pt*np.cosh(b_eta))**2)

    a_px = a_pt*np.cos(a_phi)
    b_px = b_pt*np.cos(b_phi)

    a_py = a_pt*np.sin(a_phi)
    b_py = b_pt*np.sin(b_phi)

    a_pz = a_pt*np.sinh(a_eta)
    b_pz = b_pt*np.sinh(b_eta)

    mother_E = a_E + b_E
    mother_px = a_px + b_px
    mother_py = a_py + b_py
    mother_pz = a_pz + b_pz

    mother_M = np.sqrt(mother_E**2 - mother_px**2 - mother_py**2 - mother_pz**2)
    mother_pt = np.sqrt(mother_px**2 + mother_py**2)
    mother_eta = np.arcsinh(mother_pz/mother_pt)
    mother_phi = np.arctan(mother_py/mother_px)
    

    return mother_M, mother_pt, mother_eta, mother_phi