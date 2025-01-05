import numpy as np
import matplotlib.pyplot as plt
import numdifftools
import pickle

from scipy.special import erfcinv
from scipy.optimize import curve_fit, minimize
from scipy import stats

from helpers.physics_functions import get_bins


"""
CURVE FITTING
"""

def parametric_fit(x, *theta):

    degree = len(theta) - 1
    y = np.zeros_like(x)
    for i in range(degree + 1):
        y += theta[i] * (x)**i

    return y

def integral(lower, upper, bin_width, *theta):

    degree = len(theta) - 1
    integral = 0
    for i in range(degree + 1):
        integral += theta[i] / (i + 1) * ((upper)**(i + 1) - (lower)**(i + 1))

    return integral / bin_width


def calculate_chi2(y_fit, y_true):
    return np.sum((y_fit - y_true)**2/np.sqrt(y_true+1)**2)


def curve_fit_m_inv(masses, fit_degree, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, weights = None):
    

    if weights is None:
        weights = np.ones_like(masses)
        
    # get left SB data
    loc_bkg_left = masses[masses < SR_left]
    weights_left = weights[masses < SR_left]
    y_vals_left, _bins = np.histogram(loc_bkg_left, bins = plot_bins_left, density = False, weights = weights_left)

    digits = np.digitize(loc_bkg_left, _bins)
    left_err = np.asarray([np.linalg.norm(weights_left[digits==i]) for i in range(1, len(_bins))])

    # get right SB data
    loc_bkg_right = masses[masses > SR_right]
    weights_right = weights[masses > SR_right]
    y_vals_right, _bins = np.histogram(loc_bkg_right, bins = plot_bins_right, density = False, weights = weights_right)

    digits = np.digitize(loc_bkg_right, _bins)
    right_err = np.asarray([np.linalg.norm(weights_right[digits==i]) for i in range(1, len(_bins))])

    # concatenate the SB data
    y_vals = np.concatenate((y_vals_left, y_vals_right))
    errs = np.concatenate((left_err, right_err))

    average_bin_count = np.mean(y_vals)
    
    p0 = [average_bin_count] + [0 for i in range(fit_degree)]
    n_dof_fit = fit_degree + 1


    # fit the SB data
    y_err = np.sqrt(errs**2 + 1)
    popt, pcov = curve_fit(parametric_fit, plot_centers_SB, y_vals, p0, sigma = y_err, maxfev=10000)

    # get chi2 in the SB
    chi2 = calculate_chi2(parametric_fit(plot_centers_SB, *popt), y_vals)

    return popt, pcov, chi2, y_vals, len(y_vals) - n_dof_fit




"""
STATS
"""

def ReLU(x):
    return np.maximum(0, x)



def likelihood(data, s, SR_left, SR_right, SB_left, SB_right, *theta):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right)
    plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])


    # get left SB data
    loc_bkg_left = data[data < SR_left]
    y_vals_left, _ = np.histogram(loc_bkg_left, bins = plot_bins_left, density = False)

    # get right SB data
    loc_bkg_right = data[data > SR_right]
    y_vals_right, _ = np.histogram(loc_bkg_right, bins = plot_bins_right, density = False)

    # Log poisson likelihood for the SB bins
    log_likelihood = 0
    fit_vals_left = parametric_fit(plot_centers_left, *theta)
    fit_vals_right = parametric_fit(plot_centers_right, *theta)
    for i in range(len(y_vals_left)):
        log_likelihood += stats.poisson.logpmf(y_vals_left[i], fit_vals_left[i])
    for j in range(len(y_vals_right)):
        log_likelihood += stats.poisson.logpmf(y_vals_right[j], fit_vals_right[j])

    # get SR data
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    loc_data = data[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = len(loc_data)
    num_bkg = integral(SR_left, SR_right, bin_width, *theta)

    # Log poisson likelihood for the SR
    s_prime = s * (s > 0) # Ensure positive signal. If negative, this will cancel out in the likelihood ratio
    log_likelihood += stats.poisson.logpmf(num_SR, num_bkg + s_prime)

    return -2 * log_likelihood
    
def cheat_likelihood(data, SR_left, SR_right, SB_left, SB_right, *theta):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right)
    plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])
    

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right)


    # get left SB data
    loc_bkg_left = data[data < SR_left]
    y_vals_left, _ = np.histogram(loc_bkg_left, bins = plot_bins_left, density = False)

    # get right SB data
    loc_bkg_right = data[data > SR_right]
    y_vals_right, _ = np.histogram(loc_bkg_right, bins = plot_bins_right, density = False)

    # Log poisson likelihood for the SB bins
    log_likelihood = 0
    fit_vals_left = parametric_fit(plot_centers_left, *theta)
    fit_vals_right = parametric_fit(plot_centers_right, *theta)
    for i in range(len(y_vals_left)):
        log_likelihood += stats.poisson.logpmf(y_vals_left[i], fit_vals_left[i])
    for j in range(len(y_vals_right)):
        log_likelihood += stats.poisson.logpmf(y_vals_right[j], fit_vals_right[j])

    return -2 * log_likelihood
    
def null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, *theta):
    return likelihood(data, 0, SR_left, SR_right, SB_left, SB_right, *theta)


def calculate_test_statistic(data, SR_left, SR_right, SB_left, SB_right , degree = 5, starting_guess = None, verbose_plot = False, return_popt = False):

    # We want to determine the profiled log likelihood ratio: -2 * [L(s, theta_hat_hat) - L(s_hat, theta_hat)]
    # for s = 0

    # Set up 
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right)
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    if starting_guess is None:
        average_bin_count = len(data) / len(plot_centers_all)
        starting_guess = [average_bin_count, 0, 0, 0, 0, 0, 0, 0, 0, 0][:degree + 1]


    # Fit the s = 0 hypothesis
    lambda_null = lambda theta: null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, *theta)
    fit = minimize(lambda_null , x0 = starting_guess, method = 'Nelder-Mead', options = {'maxiter': 15000, "disp": verbose_plot})
    theta_hat_hat = fit.x
    null_fit_likelihood = null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, *theta_hat_hat)


        # Fit the s = float hypothesis
    lambda_cheat = lambda theta: cheat_likelihood(data, SR_left, SR_right, SB_left, SB_right, *theta)
    fit = minimize(lambda_cheat , x0 = theta_hat_hat, method = 'Nelder-Mead', options = {'maxiter': 15000, "disp": verbose_plot})
    theta_hat = fit.x
    integrated_background = integral(SR_left, SR_right, bin_width, *theta_hat)
    num_SR = len(data[np.logical_and(data > SR_left, data < SR_right)])
    integrated_signal = num_SR - integrated_background
    best_fit_likelihood = likelihood(data, integrated_signal, SR_left, SR_right, SB_left, SB_right, *theta_hat)


    # Calculate the test statistic
    test_statistic = (null_fit_likelihood - best_fit_likelihood)
    if integrated_signal < 0:
        test_statistic = 0
    if test_statistic < 0:
        test_statistic = 0

    if verbose_plot:
        print('Best fit:', best_fit_likelihood)
        print('Null fit:', null_fit_likelihood)
        print('Test statistic:', null_fit_likelihood - best_fit_likelihood)
        print('Integrated signal:', integrated_signal)
        print('Integrated background:', integrated_background)


    if verbose_plot:
        plt.hist(data, bins = plot_bins, histtype = 'step', color = 'black', label = 'Data')
        plt.plot(plot_centers, parametric_fit(plot_centers, *theta_hat), label = 'Fit')
        plt.plot(plot_centers, parametric_fit(plot_centers, *theta_hat_hat), label = 'Null')
        plt.legend()


    if return_popt:
        return integrated_signal, integrated_background, test_statistic, theta_hat

    return integrated_signal, integrated_background, test_statistic


    