import numpy as np
import matplotlib.pyplot as plt
import numdifftools
import pickle

from scipy.special import erfcinv, loggamma
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


def binned_likelihood(yvals, ycounts, weights, fit_vals):

    log_likelihood = 0


    for i in range(len(yvals)):

        expval_weights = np.mean(weights[i])
        expval_weights2 = np.mean(weights[i]**2)
        len_weights = len(weights[i])

        if len_weights == 0 or (np.abs(expval_weights - 1) < 1e-3 and np.abs(expval_weights2 - 1) < 1e-3):
            log_likelihood += stats.poisson.logpmf(yvals[i], fit_vals[i])
        
        else:


            scale_factor = expval_weights2 / expval_weights
            lambda_prime = fit_vals[i] / scale_factor

            n_prime = yvals[i] / scale_factor

            if True:
                log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)

    return log_likelihood

def build_histogram(data, weights, bins):

    y_vals, _bins = np.histogram(data, bins = bins, density = False, weights = weights)
    y_counts, _ = np.histogram(data, bins = bins, density = False)

    digits = np.digitize(data, _bins)
    bin_weights = [weights[digits==i] for i in range(1, len(_bins))]
    bin_err = np.asarray([np.linalg.norm(weights[digits==i]) for i in range(1, len(_bins))])

    return y_vals, y_counts, bins, bin_weights, bin_err


def likelihood(data, s, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR = num_bins)
    plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])


    if weights is None:
        weights = np.ones_like(data)

    # get left SB data
    loc_bkg_left = data[data < SR_left]
    weights_left = weights[data < SR_left]
    y_vals_left, y_counts_left, _bins, left_weights, left_err = build_histogram(loc_bkg_left, weights_left, plot_bins_left)


    # get right SB data
    loc_bkg_right = data[data > SR_right]
    weights_right = weights[data > SR_right]
    y_vals_right, y_counts_right, _bins, right_weights, right_err = build_histogram(loc_bkg_right, weights_right, plot_bins_right)

    # Log poisson likelihood for the SB bins
    log_likelihood = 0
    fit_vals_left = parametric_fit(plot_centers_left, *theta)
    fit_vals_right = parametric_fit(plot_centers_right, *theta)
    
    log_likelihood += binned_likelihood(y_vals_left, y_counts_left, left_weights, fit_vals_left)
    log_likelihood += binned_likelihood(y_vals_right, y_counts_right, right_weights, fit_vals_right)
      

    # get SR data
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    loc_data = data[np.logical_and(data > SR_left, data < SR_right)]
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    err = np.sqrt(np.sum(loc_weights**2))
    num_bkg = integral(SR_left, SR_right, bin_width, *theta)
    s_prime = s * (s > 0) # Ensure positive signal. If negative, this will cancel out in the likelihood ratio

    expval_weights = np.mean(loc_weights)
    expval_weights2 = np.mean(loc_weights**2)

    # If weights are trivial
    if len(loc_weights) == 0 or (np.abs(expval_weights) < 1e-3 and np.abs(expval_weights2) < 1e-3):

        log_likelihood += stats.poisson.logpmf(num_SR, num_bkg + s_prime)

    else:
        scale_factor = expval_weights2 / expval_weights
        lambda_prime = (num_bkg + s) / scale_factor
        n_prime = num_SR / scale_factor
        log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)
    return -2 * log_likelihood
    
def cheat_likelihood(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR = num_bins)
    plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])
    

    if weights is None:
            weights = np.ones_like(data)

    # get left SB data
    loc_bkg_left = data[data < SR_left]
    weights_left = weights[data < SR_left]
    y_vals_left, y_counts_left, _bins, left_weights, left_err = build_histogram(loc_bkg_left, weights_left, plot_bins_left)

    # get right SB data
    loc_bkg_right = data[data > SR_right]
    weights_right = weights[data > SR_right]
    y_vals_right, y_counts_right, _bins, right_weights, right_err = build_histogram(loc_bkg_right, weights_right, plot_bins_right)
    
    # Log poisson likelihood for the SB bins
    log_likelihood = 0
    fit_vals_left = parametric_fit(plot_centers_left, *theta)
    fit_vals_right = parametric_fit(plot_centers_right, *theta)
   
    log_likelihood += binned_likelihood(y_vals_left, y_counts_left, left_weights, fit_vals_left)
    log_likelihood += binned_likelihood(y_vals_right, y_counts_right, right_weights, fit_vals_right)

    return -2 * log_likelihood
    
def null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta):
    return likelihood(data, 0, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta)


def calculate_test_statistic(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights = None, degree = 5, starting_guess = None, verbose_plot = False, return_popt = False):

    # We want to determine the profiled log likelihood ratio: -2 * [L(s, theta_hat_hat) - L(s_hat, theta_hat)]
    # for s = 0

    # Set up 
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins)
    if weights is None:
        weights = np.ones_like(data)

    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    if starting_guess is None:
        average_bin_count = len(data) / len(plot_centers_all)
        starting_guess = [average_bin_count, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    # Fit the s = 0 hypothesis
    lambda_null = lambda theta: null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta)
    fit = minimize(lambda_null , x0 = starting_guess, method = 'Nelder-Mead', options = {'maxiter': 15090, "disp": verbose_plot})
    theta_hat_hat = fit.x
    null_fit_likelihood = null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta_hat_hat)


        # Fit the s = float hypothesis
    lambda_cheat = lambda theta: cheat_likelihood(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta)
    fit = minimize(lambda_cheat , x0 = theta_hat_hat, method = 'Nelder-Mead', options = {'maxiter': 15090, "disp": verbose_plot})
    theta_hat = fit.x
    integrated_background = integral(SR_left, SR_right, bin_width, *theta_hat)
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    integrated_signal = num_SR - integrated_background
    best_fit_likelihood = likelihood(data, integrated_signal, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta_hat)


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

    