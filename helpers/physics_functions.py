import numpy as np
import matplotlib.pyplot as plt
import numdifftools
import pickle

from scipy.special import erfcinv
from scipy.optimize import curve_fit
from scipy import stats


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

def bkg_fit_septic(x, a0, a1, a2, a3, a4, a5, a6, a7):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6 + a7*x**7

# https://github.com/jackhcollins/CWoLa-Hunting/blob/master/code/pvalues_analysis.ipynb
def bkg_fit_ratio(x,p1,p2,p3):
        #see the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf.
        xi = 0.
        y = x/13000.
        return p1*(1.-y)**(p2-xi*p3)*y**-p3
    
 #The following code is used to get the bin errors by propagating the errors on the fit params


def get_errors_bkg_fit_ratio(popt, pcov, xdata, bkg_fit_type):
    
    
    def bkg_fit_array(parr):
        
        if bkg_fit_type == "cubic":
            a0, a1, a2, a3 = parr
            return np.array([bkg_fit_cubic(x, a0, a1, a2, a3) for x in xdata])
        
        elif bkg_fit_type == "quintic":
            a0, a1, a2, a3, a4, a5 = parr
            return np.array([bkg_fit_quintic(x, a0, a1, a2, a3, a4, a5) for x in xdata])
        
        elif bkg_fit_type == "septic":
            a0, a1, a2, a3, a4, a5, a6, a7 = parr
            return np.array([bkg_fit_septic(x, a0, a1, a2, a3, a4, a5, a6, a7) for x in xdata])
        
        if bkg_fit_type == "ratio":
            p1, p2, p3 = parr
            return np.array([bkg_fit_ratio(x, p1, p2, p3) for x in xdata])

    jac = numdifftools.core.Jacobian(bkg_fit_array)
    x_cov = np.dot(np.dot(jac(popt),pcov),jac(popt).T)
    #For plot, take systematic error band as the diagonal of the covariance matrix
    y_unc=np.sqrt([row[i] for i, row in enumerate(x_cov)])
    
    return y_unc



def calculate_chi2(y_fit, y_true):
    return np.sum((y_fit - y_true)**2/np.sqrt(y_true+1)**2)



def get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR=6, binning="linear"):
    
    if binning == "linear":
        plot_bins_SR = np.linspace(SR_left, SR_right, num_bins_SR)
        plot_centers_SR = 0.5*(plot_bins_SR[1:] + plot_bins_SR[:-1])
        width = plot_bins_SR[1] - plot_bins_SR[0]

        plot_bins_left = np.arange(SR_left, SB_left-width,  -width)[::-1]
        if plot_bins_left[0] < SB_left:
            plot_bins_left = plot_bins_left[1:]
        plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])

        plot_bins_right = np.arange(SR_right, SB_right+width, width)
        if plot_bins_right[-1] > SB_right:
            plot_bins_right = plot_bins_right[:-1]
        plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])
        
    elif binning == "log":
        plot_bins_SR = np.logspace(np.log10(SR_left), np.log10(SR_right), num_bins_SR)
        plot_centers_SR = [np.sqrt(plot_bins_SR[i]*plot_bins_SR[i+1]) for i in range(len(plot_bins_SR)-1)]
        ratio = plot_bins_SR[1]/plot_bins_SR[0]
        
        # SBL
        current_endpoint = plot_bins_SR[0]
        plot_bins_left = [current_endpoint]
        while current_endpoint > SB_left:
            next_endpoint = current_endpoint/ratio
            plot_bins_left.insert(0, next_endpoint)
            current_endpoint = next_endpoint
        if plot_bins_left[0] < SB_left:
            plot_bins_left = plot_bins_left[1:]
        plot_centers_left = [np.sqrt(plot_bins_left[i]*plot_bins_left[i+1]) for i in range(len(plot_bins_left)-1)]
        
        # SBR
        current_endpoint = plot_bins_SR[-1]
        plot_bins_right = [current_endpoint]
        while current_endpoint < SB_right:
            next_endpoint = current_endpoint*ratio
            plot_bins_right.append(next_endpoint)
            current_endpoint = next_endpoint
        if plot_bins_right[-1] > SB_right:
            plot_bins_right = plot_bins_right[:-1]
        plot_centers_right = [np.sqrt(plot_bins_right[i]*plot_bins_right[i+1]) for i in range(len(plot_bins_right)-1)]
    
    plot_centers_all = np.concatenate((plot_centers_left, plot_centers_SR, plot_centers_right))
    plot_centers_SB =np.concatenate((plot_centers_left, plot_centers_right))
    plot_bins_all = np.concatenate([plot_bins_left[:-1], plot_bins_SR, plot_bins_right[1:]])
    
    
    return plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB
             
def get_bins_for_scan(path_to_bin_defs_folder, window_index, scale_bins=False):
    
    """
    If this code is run on preprocessed data, then the bin definitions need to be modified with a scalar
    """
    
    print(f"Loading in bin definitions from {path_to_bin_defs_folder}/bin_definitions...")
    with open(f"{path_to_bin_defs_folder}/bin_definitions", "rb") as infile:
        bin_definitions = pickle.load(infile)
    window_bin_definitions = bin_definitions[window_index]
    
    plot_bins_SR = window_bin_definitions["SR"]
    plot_bins_left = window_bin_definitions["SBL"]
    plot_bins_right = window_bin_definitions["SBH"]
    
    plot_centers_SR = [np.sqrt(plot_bins_SR[i]*plot_bins_SR[i+1]) for i in range(len(plot_bins_SR)-1)]
    plot_centers_left = [np.sqrt(plot_bins_left[i]*plot_bins_left[i+1]) for i in range(len(plot_bins_left)-1)]
    plot_centers_right = [np.sqrt(plot_bins_right[i]*plot_bins_right[i+1]) for i in range(len(plot_bins_right)-1)]

    plot_centers_all = np.concatenate((plot_centers_left, plot_centers_SR, plot_centers_right))
    plot_centers_SB = np.concatenate((plot_centers_left, plot_centers_right))
    plot_bins_all = np.concatenate([plot_bins_left[:-1], plot_bins_SR, plot_bins_right[1:]])
    
    if scale_bins:
        with open(f"{path_to_bin_defs_folder}/mass_scaler_{window_index}", "rb") as infile:
            mass_scaler = pickle.load(infile)
        plot_bins_all = mass_scaler.transform(np.array(plot_bins_all).reshape(-1,1)).reshape(-1,)
        plot_bins_SR = mass_scaler.transform(np.array(plot_bins_SR).reshape(-1,1)).reshape(-1,)
        plot_bins_left = mass_scaler.transform(np.array(plot_bins_left).reshape(-1,1)).reshape(-1,)
        plot_bins_right = mass_scaler.transform(np.array(plot_bins_right).reshape(-1,1)).reshape(-1,)
        plot_centers_all = mass_scaler.transform(np.array(plot_centers_all).reshape(-1,1)).reshape(-1,)
        plot_centers_SR = mass_scaler.transform(np.array(plot_centers_SR).reshape(-1,1)).reshape(-1,)
        plot_centers_SB = mass_scaler.transform(np.array(plot_centers_SB).reshape(-1,1)).reshape(-1,)
            
    return plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB
    

def select_top_events_fold(true_masses, scores, score_cutoff, plot_bins_left, plot_bins_right, plot_bins_SR):
    
    """
    true_masses: unpreprocessed masses
    """
    # get the events that pass the score threshold
    pass_scores = scores >= score_cutoff

    # correct for diff efficiency in the SB
    SBL_counts_passed, _ = np.histogram(true_masses[pass_scores], bins = plot_bins_left)
    SBL_counts_all, _ = np.histogram(true_masses, bins = plot_bins_left)

    SBH_counts_passed, _ = np.histogram(true_masses[pass_scores], bins = plot_bins_right)
    SBH_counts_all, _ = np.histogram(true_masses, bins = plot_bins_right)

    SR_counts_passed, _ = np.histogram(true_masses[pass_scores], bins = plot_bins_SR)
    SR_counts_all, _ = np.histogram(true_masses, bins = plot_bins_SR)
    
    return true_masses[pass_scores], SBL_counts_passed/SBL_counts_all, SBH_counts_passed/SBH_counts_all, SR_counts_passed/SR_counts_all

   
def curve_fit_m_inv(masses, fit_type, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, SBL_rescale=None, SBH_rescale=None):
    

    if fit_type == "cubic":
        p0  = [5000, -20000, 30000, -10000]
        fit_function = bkg_fit_cubic
        n_dof_fit = 4

    elif fit_type == "quintic":
        p0  = [5000, -20000, 30000, -10000, 0, 0]
        fit_function = bkg_fit_quintic
        n_dof_fit = 6
        
    elif fit_type == "septic":
        p0  = [5000, -20000, 30000, -10000, 0, 0, 0, 0]
        fit_function = bkg_fit_septic
        n_dof_fit = 8

    elif fit_type == "ratio":
        p0  = [100,1000,.1]
        fit_function = bkg_fit_ratio
        n_dof_fit = 3
        
    # get left SB data
    loc_bkg_left = masses[masses < SR_left]
    y_vals_left, _ = np.histogram(loc_bkg_left, bins = plot_bins_left, density = False)

    # get right SB data
    loc_bkg_right = masses[masses > SR_right]
    y_vals_right, _ = np.histogram(loc_bkg_right, bins = plot_bins_right, density = False)

    # concatenate the SB data
    if SBL_rescale is not None:
        y_vals = np.concatenate((SBL_rescale*y_vals_left, SBH_rescale*y_vals_right))
    else: 
        y_vals = np.concatenate((y_vals_left, y_vals_right))

    # fit the SB data
    y_err = np.sqrt(y_vals + 1)
    popt, pcov = curve_fit(fit_function, plot_centers_SB, y_vals, p0, sigma = y_err, maxfev=10000)

    # get chi2 in the SB
    chi2 = calculate_chi2(fit_function(plot_centers_SB, *popt), y_vals)

    return popt, pcov, chi2, y_vals, len(y_vals) - n_dof_fit







def calc_significance(masses, fit_function, plot_bins_SR, plot_centers_SR, SR_left, SR_right, popt, pcov = None, ONE_SIDED = False, TWO_SIDED = False):

    num_B_expected_in_SR = sum(fit_function(plot_centers_SR, *popt))
    num_total_in_SR = len(masses[(masses >= SR_left) & (masses <= SR_right)])
    
    num_S_expected_in_SR = num_total_in_SR - num_B_expected_in_SR

    # ONE-SIDED CONSTRAINT
    if ONE_SIDED:

        # Get hist of just B 
        B_function = fit_function(plot_centers_SR, *popt)

        # Get hist of SR data (S+B)
        S_plus_B_function, _ = np.histogram(masses, bins = plot_bins_SR, density = False)

        # If S+B is less than B, then raise the S+B to the B level
        S_plus_B_function = np.where(S_plus_B_function < B_function, B_function, S_plus_B_function)

        # Get the expected number of S events in SR
        num_S_expected_in_SR = sum(S_plus_B_function) - num_B_expected_in_SR


    if TWO_SIDED:

        if pcov is None:
            raise ValueError("Need to provide covariance matrix for two-sided significance calculation")

        # Get the chi2 between the fit and the data in the SR
        B_function = fit_function(plot_centers_SR, *popt) 

        # Get the fit errors using resampling trick
        n = 1000
        temp_params = np.random.multivariate_normal(popt, pcov, n)
        y = np.array([fit_function(plot_centers_SR, *p) for p in temp_params])
        B_error = np.std(y, axis = 0)

        S_plus_B_function, _ = np.histogram(masses, bins = plot_bins_SR, density = False)
        B_error = np.sqrt(B_error**2 + np.sqrt(S_plus_B_function + 1)**2)

        # Get the chi2 between the data and the fit
        chi2 = np.sum((S_plus_B_function - B_function)**2 / B_error**2)
        N_DOF = len(S_plus_B_function) - len(popt)
        chi2_ndof = chi2/N_DOF

        # p_value = 1 - stats.chi2.cdf(chi2, N_DOF)
        log_p_value = stats.chi2.logsf(chi2, N_DOF)
        significance =  np.sqrt(2) * erfcinv(1 * np.exp(log_p_value))
        # significance = stats.norm.ppf(1 - np.exp(log_p_value) / 2)
        # approx_significance = (chi2 - N_DOF) /  np.sqrt(2*N_DOF)

        # print("chi2/ndof:", chi2_ndof, "p_value:", np.exp(log_p_value), "significance:", significance)

        return num_S_expected_in_SR, num_B_expected_in_SR, significance



    return num_S_expected_in_SR, num_B_expected_in_SR



def plot_histograms_with_fits(fpr_thresholds, data_dict_by_fold, scores_dict_by_fold, score_cutoffs_by_fold, mass_scalar, fit_type, title, SB_left, SR_left, SR_right, SB_right, n_folds= 5, take_score_avg=True):
    
    if fit_type == "cubic": fit_function = bkg_fit_cubic
    elif fit_type == "quintic": fit_function = bkg_fit_quintic
    elif fit_type == "septic": fit_function = bkg_fit_septic
    elif fit_type == "ratio": fit_function = bkg_fit_ratio

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right)


    plt.figure(figsize = (12, 9))
    for t, threshold in enumerate(fpr_thresholds):

        # corrections to SR / SB efficiencies
        filtered_masses = []

        for i_fold in range(n_folds):
            loc_true_masses = mass_scalar.inverse_transform(np.array(data_dict_by_fold[i_fold][:,-1]).reshape(-1,1))
            if take_score_avg:
                loc_scores = np.mean(scores_dict_by_fold[i_fold], axis = 1)
            else:
                loc_scores = scores_dict_by_fold[i_fold]
            loc_filtered_masses, loc_SBL_eff, loc_SBH_eff, loc_SR_eff = select_top_events_fold(loc_true_masses, loc_scores, score_cutoffs_by_fold[i_fold][threshold],plot_bins_left, plot_bins_right, plot_bins_SR)
            filtered_masses.append(loc_filtered_masses)

        filtered_masses = np.concatenate(filtered_masses)

        # get the fit function to SB background
        popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(filtered_masses, fit_type, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_all)
        #print("chi2/dof:", chi2/n_dof)
        # plot the fit function
        plt.plot(plot_centers_all, fit_function(plot_centers_all, *popt), lw = 2, linestyle = "dashed", color = f"C{t}")    

        # calculate significance of bump
        num_S_expected_in_SR, num_B_expected_in_SR = calc_significance(filtered_masses, fit_function, plot_bins_SR, SR_left, SR_right, popt)

        y_err = get_errors_bkg_fit_ratio(popt, pcov, plot_centers_SR, fit_type)
        B_error = np.sqrt(np.sum(y_err**2))
        
        S_over_B = num_S_expected_in_SR/num_B_expected_in_SR
        significance = num_S_expected_in_SR/np.sqrt(num_B_expected_in_SR+B_error**2)

        label_string = str(round(100*threshold, 2))+"% FPR: $S/B$: "+str(round(S_over_B,4))+", $S/\sqrt{B}$: "+str(round(significance,4))

        plt.hist(filtered_masses, bins = plot_bins_all, lw = 3, histtype = "step", color = f"C{t}",label = label_string)
        plt.scatter(plot_centers_SB, y_vals, color = f"C{t}")


    plt.legend(loc = (1, 0), fontsize = 24)


    plt.axvline(SR_left, color= "k", lw = 3, zorder = -10)
    plt.axvline(SR_right, color= "k", lw = 3, zorder = -10)

    plt.xlabel("$M_{\mu\mu}$ [GeV]", fontsize = 24)
    plt.ylabel("Counts", fontsize = 24)

    plt.title(title, fontsize = 24)
    

