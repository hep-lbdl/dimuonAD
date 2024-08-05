import numpy as np

from scipy.optimize import curve_fit


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


def calculate_deltaR(phi_0, phi_1, eta_0, eta_1):
    
    delta_phi = np.abs(phi_0 - phi_1)
    # adjust to be in the range(-pi, pi)
    delta_phi = np.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
    delta_R = np.sqrt(delta_phi**2 + (eta_0 - eta_1)**2)
    
    return delta_R


"""
STATS AND CURVE FITTING
"""

def bkg_fit_cubic(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*x**2 + a3*x**3

def bkg_fit_quintic(x, a0, a1, a2, a3, a4, a5):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5


def calculate_chi2(y_fit, y_true):
    return np.sum((y_fit - y_true)**2/y_true)
                 
def curve_fit_m_inv(masses, fit_function, left_bound, right_bound, SR_left, SR_right, width, p0, remove_edge = True):
        
        # get left SB data
        loc_bkg_left = masses[masses < SR_left]

        plot_bins_left = np.arange(SR_left, left_bound-width,  -width)[::-1]
        if remove_edge:
            plot_bins_left = plot_bins_left[1:]
    
        plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
        y_vals_left, _ = np.histogram(loc_bkg_left, bins = plot_bins_left, density = False)

        # get right SB data
        loc_bkg_right = masses[masses > SR_right]
        
        plot_bins_right = np.arange(SR_right, right_bound+width, width)
        if remove_edge:
            plot_bins_right = plot_bins_right[:-1]
       
        plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])
        y_vals_right, _ = np.histogram(loc_bkg_right, bins = plot_bins_right, density = False)

        # concatenate the SB data
        y_vals = np.concatenate((y_vals_left, y_vals_right))
        plot_centers = np.concatenate((plot_centers_left, plot_centers_right))

        # fit the SB data
        popt, pcov = curve_fit(fit_function, plot_centers, y_vals, p0)

        # get chi2 in the SB
        chi2 = calculate_chi2(fit_function(plot_centers, *popt), y_vals)
        
        return popt, chi2, len(y_vals), plot_bins_left, plot_bins_right
    
    
def calc_significance(masses, fit_function, plot_bins_SR, SR_left, SR_right, popt):

    x_SR_center = 0.5*(plot_bins_SR[1:] + plot_bins_SR[:-1])
    num_B_expected_in_SR = sum(fit_function(x_SR_center, *popt))
    num_total_in_SR = len(masses[(masses >= SR_left) & (masses <= SR_right)])

    num_S_expected_in_SR = num_total_in_SR - num_B_expected_in_SR
    
    return num_S_expected_in_SR, num_B_expected_in_SR