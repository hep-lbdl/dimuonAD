#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from scipy.special import erfcinv, loggamma
from scipy.optimize import curve_fit, minimize
from scipy import stats



from helpers.stats_functions import  curve_fit_m_inv

# from helpers.physics_functions import bkg_fit_cubic, bkg_fit_septic, bkg_fit_quintic, get_bins, select_top_events_fold, curve_fit_m_inv, calc_significance, get_errors_bkg_fit_ratio, calculate_test_statistic

import sys


# In[ ]:


SR_left = 9.0
SR_right = 10.6
SB_left = 5.0
SB_right = 16.0
num_bins_SR = 12

try:
    start_index = int(sys.argv[1])
except:
    print('Please provide an FPR index, Defaulting to 0')
    start_index = 0

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


def binned_likelihood(yvals, ycounts, weights, fit_vals):

    log_likelihood = 0
    for i in range(len(yvals)):

        if False:
            log_likelihood += stats.poisson.logpmf(yvals[i], fit_vals[i])
        
        else:
            expval_weights = np.mean(weights[i])
            expval_weights2 = np.mean(weights[i]**2)

            scale_factor = expval_weights2 / expval_weights
            lambda_prime = fit_vals[i] / scale_factor

            n_prime = yvals[i] / scale_factor

            if n_prime > 0 and expval_weights > 0:
                log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)


    return log_likelihood

def build_histogram(data, weights, bins):

    y_vals, _bins = np.histogram(data, bins = bins, density = False, weights = weights)
    y_counts, _ = np.histogram(data, bins = bins, density = False)

    digits = np.digitize(data, _bins)
    bin_weights = [weights[digits==i] for i in range(1, len(_bins))]
    bin_err = np.asarray([np.linalg.norm(weights[digits==i]) for i in range(1, len(_bins))])

    return y_vals, y_counts, bins, bin_weights, bin_err


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

plot_bins = np.concatenate((plot_bins_left, plot_bins_SR, plot_bins_right))
plot_centers = np.concatenate((plot_centers_left, plot_centers_SR, plot_centers_right))

plot_centers_SB = np.concatenate((plot_centers_left, plot_centers_right))

def likelihood(data, s, weights, *theta):

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

    scale_factor = expval_weights2 / expval_weights
    lambda_prime = (num_bkg + s) / scale_factor
    n_prime = num_SR / scale_factor
    log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)
    
    # log_likelihood += num_SR * np.log(num_bkg + s_prime) - (num_bkg + s_prime) - loggamma(num_SR + 1)
    # log_likelihood += stats.norm.logpdf(num_SR, num_bkg + s_prime, np.sqrt(err**2 + 1))

    return -2 * log_likelihood
    
def cheat_likelihood(data, weights, *theta):

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
    
def null_hypothesis(data, weights, *theta):
    return likelihood(data, 0, weights, *theta)

def calculate_test_statistic(data, weights = None, degree = 5, starting_guess = None, verbose_plot = False, return_popt = False):

    # We want to determine the profiled log likelihood ratio: -2 * [L(s, theta_hat_hat) - L(s_hat, theta_hat)]
    # for s = 0

    if weights is None:
        weights = np.ones_like(data)

    # Set up 
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    average_bin_count = len(data) / len(plot_bins)
    if starting_guess is None:
        starting_guess = [average_bin_count, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    # Fit the s = 0 hypothesis
    lambda_null = lambda theta: null_hypothesis(data, weights, *theta)
    fit = minimize(lambda_null , x0 = starting_guess, method = 'Nelder-Mead', options = {'maxiter': 1500, "disp": verbose_plot})
    theta_hat_hat = fit.x
    null_fit_likelihood = null_hypothesis(data, weights, *theta_hat_hat)


        # Fit the s = float hypothesis
    lambda_cheat = lambda theta: cheat_likelihood(data, weights, *theta)
    fit = minimize(lambda_cheat , x0 = theta_hat_hat, method = 'Nelder-Mead', options = {'maxiter': 1500, "disp": verbose_plot})
    theta_hat = fit.x
    integrated_background = integral(SR_left, SR_right, bin_width, *theta_hat)
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    integrated_signal = num_SR - integrated_background
    best_fit_likelihood = likelihood(data, integrated_signal, weights, *theta_hat)


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


data = np.random.uniform(5.0, 16.0, 1000)

likelihood(data, 0, np.ones_like(data), *np.ones(10))


# q0s = []

# for i in tqdm(range(10)):
#     data = np.random.uniform(5.0, 16.0, 1000)
#     verbose_plot = i == 0
#     s, b, q0 = calculate_test_statistic(data, weights = np.ones_like(data), degree = 5, verbose_plot = verbose_plot)
#     q0s.append(q0)
#     print(i, q0, popt)


# In[49]:


# q0s = np.array(q0s) 
# q0s = q0s * (q0s > 0)
# z_scores = np.sqrt(np.array(q0s))
# print(q0s)

# # Print the median, 1 sigma, and 2 sigma values
# print('Median:', np.median(z_scores))
# print('1 sigma:', np.percentile(z_scores, 84))
# print('2 sigma:', np.percentile(z_scores, 97.5))

# # Plot the distribution of q0 
# plt.hist(q0s, bins = 10, histtype = 'step', color = 'blue', alpha = 0.5, label = 'q0', density = True)

# # Asymptotic formula
# def f(q):
#     return 1/(2*np.sqrt(q * 2 * np.pi)) * np.exp(-q/2)

# q = np.linspace(0, 10, 1000)
# plt.plot(q, f(q), label = 'Asymptotic formula', color = 'red')


# In[ ]:





# In[50]:


import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file) 
    
feature_id = "mix_2"
bootstrap_flow = 0 # don't change this from 0

train_samesign = False
# somewhat complicated code to set naming conventions
if train_samesign:
    train_data_id = "SS"
else:
    train_data_id = "OS"

# train on opp sign means alt test set is samesign
if train_data_id == "OS": 
    alt_test_data_id = "SS"
elif train_data_id == "SS": 
    alt_test_data_id = "OS"

configs = "CATHODE_8"


# pickles contain all the results from the BDT training
working_dir = workflow["file_paths"]["working_dir"]
processed_data_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/"+workflow["analysis_keywords"]["name"]+"/processed_data"
flow_training_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/" + workflow["analysis_keywords"]["name"]+f"/models/bootstrap{bootstrap_flow}_{train_data_id}/{feature_id}/{configs}/"
pickle_save_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/" + workflow["analysis_keywords"]["name"]+f"/pickles/bootstrap{bootstrap_flow}_{train_data_id}/{feature_id}/"
plot_data_dir = "plot_data/"

# basically hard-coded for the PRL 
num_pseudoexperiments = 1001
n_folds = 5

num_bins_SR = 12 # 16, 12, 8


def bkg_fit_cubic(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3 

def bkg_fit_quintic(x, a, b, c, d, e, f):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5

def bkg_fit_septic(x, a, b, c, d, e, f, g, h):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7


pseudo_e_to_plot = 0 # this plots the actual data (not a boostrapped version)
fit_type = "quintic" # "cubic", "quintic", septic
if fit_type == "cubic": fit_function = bkg_fit_cubic
if fit_type == "quintic": fit_function = bkg_fit_quintic
if fit_type == "septic": fit_function = bkg_fit_septic


# In[51]:


# if train_samesign = False, this loads in the OS test data
# test 

bkg_fit_degree = 5
fit_types = {3: "cubic", 5: "quintic", 7: "septic"}

def load_in_pseudoexperiments(file_string, num_pseudoexps):

    master_dict = {}

    with open(f"{pickle_save_dir}/{file_string}_bkg_fit_{bkg_fit_degree}_num_bins_{num_bins_SR}_0_1", "rb") as ifile:
        loc_dict = pickle.load(ifile)
    master_dict = {**loc_dict}
    # load in the bootstraps
    for i in range(1, num_pseudoexps):
        with open(f"{pickle_save_dir}/bkg_samples/bootstrap{i}/{file_string}_{fit_types[bkg_fit_degree]}_{num_bins_SR}_0_1", "rb") as ifile:
            loc_dict = pickle.load(ifile)
            master_dict[i] = loc_dict[0]
    return master_dict

num_to_plot = num_pseudoexperiments

all_test_data_splits = load_in_pseudoexperiments("all_test_data_splits", num_to_plot)
print(len(all_test_data_splits.keys())==num_pseudoexperiments)

# test scores
all_scores_splits = load_in_pseudoexperiments("all_scores_splits", num_to_plot)
print(len(all_scores_splits.keys())==num_pseudoexperiments)

# alt data
# if train_samesign = False, this loads in the SS test data, OS high-stats data, and OS flow samples
# if train_samesign = True, this loads in the OS test data, SS high-stats data, and SS flow samples
all_alt_data_splits = load_in_pseudoexperiments("all_alt_data_splits", num_to_plot)
print(len(all_alt_data_splits.keys())==num_pseudoexperiments)
# alt scores
all_alt_scores_splits = load_in_pseudoexperiments("all_alt_scores_splits", num_to_plot)

print(len(all_alt_scores_splits.keys())==num_pseudoexperiments)

with open(f"{processed_data_dir}/mass_scaler_bootstrap{bootstrap_flow}", "rb") as ifile:
    scaler = pickle.load(ifile)
    
with open(f"{processed_data_dir}/preprocessing_info_bootstrap{bootstrap_flow}", "rb") as ifile:
     preprocessing_info = pickle.load(ifile)


# In[52]:


SB_left = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SB_left"])
SR_left = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SR_left"])
SR_right = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SR_right"])
SB_right = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SB_right"])

print(SB_left, SB_right)


data_prefix = f"upsilon_{train_data_id}"
print(data_prefix)


# In[53]:


pseudo = 0

# Load in scores
data_dict_by_fold = all_test_data_splits[pseudo]
scores_dict_by_fold = all_scores_splits[pseudo]

all_data = np.vstack([data_dict_by_fold[i] for i in range(n_folds)])
all_scores = np.vstack([scores_dict_by_fold[i].reshape(-1,1) for i in range(n_folds)]) 
all_masses = scaler.inverse_transform(all_data[:,-1].reshape(-1,1))

in_SR = (all_masses >= SR_left ) & (all_masses <= SR_right)
in_SBL = (all_masses < SR_left )
in_SBH = (all_masses > SR_right )

mass_SBL = all_masses[in_SBL]
mass_SR = all_masses[in_SR]
mass_SBH = all_masses[in_SBH]

scores_SBL = all_scores[in_SBL]
scores_SR = all_scores[in_SR]
scores_SBH = all_scores[in_SBH]

all_likelihood_ratios = (all_scores) / (1 - all_scores)
likelihood_ratios_SBL = (scores_SBL) / (1 - scores_SBL)
likelihood_ratios_SR = (scores_SR) / (1 - scores_SR)
likelihood_ratios_SBH = (scores_SBH) / (1 - scores_SBH)

log_likelihoods_SBL = (scores_SBL / (1 - scores_SBL))
log_likelihoods_SR = np.log(scores_SR / (1 - scores_SR))
log_likelihoods_SBH = np.log(scores_SBH / (1 - scores_SBH))

plt.hist(likelihood_ratios_SBL, bins = 50, histtype = 'step', color = 'black', label = 'SBL')
plt.hist(likelihood_ratios_SR, bins = 50, histtype = 'step', color = 'red', label = 'SR')
plt.hist(likelihood_ratios_SBH, bins = 50, histtype = 'step', color = 'blue', label = 'SBH')

print(np.mean(log_likelihoods_SBL), np.mean(log_likelihoods_SR), np.mean(log_likelihoods_SBH))


# Get the likelihood assuming BCE
likelihood_ratios = (all_scores) / (1 - all_scores)
log_likelihoods = np.log(likelihood_ratios)
log_likelihood = np.sum(log_likelihoods)

# plt.hist(log_likelihood, bins = 50, histtype = 'step', color = 'black', label = 'Data')


# In[54]:


# Mass histograms

popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(all_masses, 5, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)
s, b, q0, popt = calculate_test_statistic(all_masses, degree = 5, verbose_plot = False, starting_guess = popt, return_popt = True)
mu = (s) / (s + b)
plt.hist(all_masses, plot_bins, histtype = 'step', color = 'black', label = r'Unweighted: $Z_0 = $' + str(np.sqrt(q0)))
y_vals = parametric_fit(plot_centers, *popt)
plt.plot(plot_centers, y_vals, color = 'black')
print(s, b, np.sqrt(q0), mu)

# dummy_weights = np.ones_like(all_masses) * 2
# popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(all_masses, 5, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, weights = dummy_weights)
# s, b, q0, popt = calculate_test_statistic(all_masses, degree = 5, verbose_plot = False, starting_guess = popt, return_popt = True, weights = dummy_weights)
# mu = (s) / (s + b)
# plt.hist(all_masses, plot_bins, histtype = 'step', color = 'Grey', label = r'Weight = 2: $Z_0 = $' + str(np.sqrt(q0)), weights = dummy_weights)
# y_vals = parametric_fit(plot_centers, *popt)
# plt.plot(plot_centers, y_vals, color = 'Grey')
# print(s, b, np.sqrt(q0), mu)

# # Same thing but with the likelihood ratios as a weight
# popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(all_masses, 5, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, weights = likelihood_ratios)
# s, b, q0, popt = calculate_test_statistic(all_masses, weights = likelihood_ratios, degree = 5, verbose_plot = False, starting_guess = popt, return_popt = True)
# plt.hist(all_masses, plot_bins, histtype = 'step', color = 'red', label = r'Weight = $\ell$: $Z_0 = $' + str(np.sqrt(q0)), weights = likelihood_ratios)
# y_vals = parametric_fit(plot_centers, *popt)
# plt.plot(plot_centers, y_vals, color = 'red')
# print(s, b, np.sqrt(q0), mu)

# # Same thing but with the likelihood ratios as a weight
# popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(all_masses, 5, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, weights = 0.5 * likelihood_ratios)
# s, b, q0, popt = calculate_test_statistic(all_masses, weights = 0.5*likelihood_ratios , degree = 5, verbose_plot = False, starting_guess = popt, return_popt = True)
# plt.hist(all_masses, plot_bins, histtype = 'step', color = 'purple', label = r'Weight = $0.5 \ell$: $Z_0 = $' + str(np.sqrt(q0)), weights = 0.5 * likelihood_ratios)
# y_vals = parametric_fit(plot_centers, *popt)
# plt.plot(plot_centers, y_vals, color = 'purple')
# print(s, b, np.sqrt(q0), mu)

weights = (likelihood_ratios - (1-mu)) / mu
weights = np.clip(weights, 0, 1e6)
popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(all_masses, 5, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, weights = weights)
y_vals = parametric_fit(plot_centers, *popt)
plt.plot(plot_centers, y_vals, color = 'blue', ls = '--')
s, b, q0, popt = calculate_test_statistic(all_masses, weights = weights, degree = 5, verbose_plot = False, starting_guess = popt, return_popt = True)
plt.hist(all_masses, plot_bins, histtype = 'step', color = 'blue', label = r'Weight = $\ell - (1-\mu)/\mu$: $Z_0 = $' + str(np.sqrt(q0)), weights = weights)
y_vals = parametric_fit(plot_centers, *popt)
plt.plot(plot_centers, y_vals, color = 'blue')
print(s, b, np.sqrt(q0), mu)

plt.legend(title = "mu = " + str(mu))


# In[55]:


plt.hist(weights, bins = 50, histtype = 'step', color = 'black', label = 'Data')
plt.yscale('log')

print(np.mean(weights), np.std(weights))
print(np.max(weights), np.min(weights))


# In[56]:


print(*popt)
plt.hist(weights, bins = 50, histtype = 'step', color = 'black', density = True)
# plt.hist(likelihood_ratios, bins = 50, histtype = 'step', color = 'red', density = True)
null = null_hypothesis(all_masses, weights, *popt)
print(null)

like = likelihood(all_masses, s, weights, *popt)
print(like)
print(2 * (null - like))

n_prime = 11.0
lambda_prime = 11.72
print(stats.poisson.logpmf(n_prime, lambda_prime))
print(n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1))


# In[57]:


q0s = []
mus = []
ss = []
bs = []

start_index = start_index
end_index = start_index + 100

for pseudo in range(start_index, end_index):

    # Load in scores
    data_dict_by_fold = all_test_data_splits[pseudo]
    scores_dict_by_fold = all_scores_splits[pseudo]

    all_data = np.vstack([data_dict_by_fold[i] for i in range(n_folds)])
    all_scores = np.vstack([scores_dict_by_fold[i].reshape(-1,1) for i in range(n_folds)]) 
    all_masses = scaler.inverse_transform(all_data[:,-1].reshape(-1,1))

    # epsilon = 1e-3
    # all_scores = np.random.normal(0.5, epsilon, (len(all_scores),1))
    # print(all_scores.shape)

    in_SR = (all_masses >= SR_left ) & (all_masses <= SR_right)
    in_SBL = (all_masses < SR_left )
    in_SBH = (all_masses > SR_right )

    mass_SBL = all_masses[in_SBL]
    mass_SR = all_masses[in_SR]
    mass_SBH = all_masses[in_SBH]


    # Get the likelihood assuming BCE
    likelihood_ratios = (all_scores) / (1 - all_scores)
    
    # Get mu
    popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(all_masses, 5, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)
    s, b, q0, popt = calculate_test_statistic(all_masses, degree = 5, verbose_plot = False, starting_guess = popt, return_popt = True)
    mu = (s) / (s + b)

    mus.append(mu)
    ss.append(s)
    bs.append(b)


    weights = (likelihood_ratios - (1-mu)) / mu
    weights = np.clip(weights, 0, 1e6)
    popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(all_masses, 5, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, weights = weights)
    s, b, q0, popt = calculate_test_statistic(all_masses, weights = weights, degree = 5, verbose_plot = False, starting_guess = popt, return_popt = True)

    print("Experiment: ", pseudo, "Signal: ", s, "Background: ", b, "mu: ", mu, "q0: ", np.sqrt(q0))

    q0s.append(q0)

q0s = np.array(q0s)
mus = np.array(mus)
ss = np.array(ss)
bs = np.array(bs)


# In[ ]:


np.save(f"junk/q0s_{start_index}_{end_index}", q0s)


# In[58]:


start_indices = [1, 101, 201, 301, 401, 501, 601, 701, 801, 901]
end_indices = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]

all_q0s = []
for start_index, end_index in zip(start_indices, end_indices):
    all_q0s.append(np.load(f"junk/q0s_{start_index}_{end_index}.npy"))

all_q0s = np.concatenate(all_q0s)
# filter out nans
all_q0s = all_q0s[~np.isnan(all_q0s)]
print(all_q0s.shape)

z_scores = np.sqrt(all_q0s)

z_scores = np.sort(z_scores)

# Print the median, 1 sigma, and 2 sigma values
med = np.median(z_scores)
print('Median:', med)
print('1 sigma:', np.percentile(z_scores, 84))
print('2 sigma:', np.percentile(z_scores, 97.5))
print('3 sigma:', np.percentile(z_scores, 99.7))
print('4 sigma:', np.percentile(z_scores, 99.99))



counts, bins = np.histogram(z_scores**2, bins = 50, density = True)
centers = (bins[:-1] + bins[1:]) / 2

plt.hist(z_scores**2, bins = 50, histtype = 'step', color = 'black', density = True)

def f(q):
    return 1/(2*np.sqrt(q * 2 * np.pi)) * np.exp(-q/2)

plt.plot(centers, f(centers), label = "Asymptotic", color = "red")

plt.yscale("log")
plt.xlabel("Test statistic $q_0$")

# # Print the median, 1sigma, and 2sigma values
# print("Median: ", np.median(z_score))
# print("1sigma: ", np.percentile(z_score, 16), np.percentile(z_score, 84))
# print("2sigma: ", np.percentile(z_score, 2.5), np.percentile(z_score, 97.5))

# Print the percentile of z = 0, 1, 2
index = np.searchsorted(z_scores, [0, 1, 2])
print("Percentile of 0: ", index[0] / len(z_scores))
print("Percentile of Z = 1: ", index[1] / len(z_scores))
print("Percentile of Z = 2: ", index[2] / len(z_scores))

# number of q0 == 0
print("Number of q0 == 0: ", np.mean(z_scores == 0))


# In[ ]:





# In[28]:


plt.hist(mus[1:], bins = 15, histtype = 'step', color = 'black')

# print quantiles of mu
print(np.percentile(mus[1:], 50))


# In[ ]:





# In[46]:


z_scores = np.sqrt(np.array(q0s[1:]))
z_scores = np.sort(z_scores)

# Print the median, 1 sigma, and 2 sigma values
med = np.median(z_scores)
print('Median:', med)
print('1 sigma:', np.percentile(z_scores, 84))
print('2 sigma:', np.percentile(z_scores, 97.5))
print('3 sigma:', np.percentile(z_scores, 99.7))



counts, bins = np.histogram(z_scores**2, bins = 50, density = True)
centers = (bins[:-1] + bins[1:]) / 2

plt.hist(z_scores**2, bins = 50, histtype = 'step', color = 'black', density = True)

def f(q):
    return 1/(2*np.sqrt(q * 2 * np.pi)) * np.exp(-q/2)

plt.plot(centers, f(centers), label = "Asymptotic", color = "red")

plt.yscale("log")
plt.xlabel("Test statistic $q_0$")

# # Print the median, 1sigma, and 2sigma values
# print("Median: ", np.median(z_score))
# print("1sigma: ", np.percentile(z_score, 16), np.percentile(z_score, 84))
# print("2sigma: ", np.percentile(z_score, 2.5), np.percentile(z_score, 97.5))

# Print the percentile of z = 0, 1, 2
index = np.searchsorted(z_scores, [0, 1, 2])
print("Percentile of 0: ", index[0] / len(z_scores))
print("Percentile of Z = 1: ", index[1] / len(z_scores))
print("Percentile of Z = 2: ", index[2] / len(z_scores))

# number of q0 == 0
print("Number of q0 == 0: ", np.mean(z_scores == 0))


# In[32]:


likelihoods = np.array([-2 * bdt_log_likelihood(pseudo) for pseudo in range(num_to_plot)])
pseudoexperiment_likelihoods = likelihoods[1:]
data_likelihood = likelihoods[0]

print(np.mean(pseudoexperiment_likelihoods), data_likelihood)
print(np.std(pseudoexperiment_likelihoods)**2 )

dof = 176

# plt.hist(pseudoexperiment_likelihoods[0], bins = 50, density = True, alpha = 0.5, color = "blue", label = "Pseudoexperiments")
# plt.hist(data_likelihood, bins = 50, density = True, alpha = 0.5, color = "red", label = "Data")

median = np.median(pseudoexperiment_likelihoods)
one_sigma = np.percentile(pseudoexperiment_likelihoods, 84)
two_sigma = np.percentile(pseudoexperiment_likelihoods, 97.5)
three_sigma = np.percentile(pseudoexperiment_likelihoods, 99.7)



plt.hist(pseudoexperiment_likelihoods, bins = 50, density = True, alpha = 0.5, color = "blue", label = "Pseudoexperiments")
plt.axvline(data_likelihood, color = "red", label = "Data")

# plt.axvline(median, color = "black")
# plt.axvline(one_sigma, color = "black", linestyle = "--")
# plt.axvline(two_sigma, color = "black", linestyle = "--")
# plt.axvline(three_sigma, color = "black", linestyle = "--")

# fit a lognormal distribution to the pseudoexperiments
mu = np.mean(pseudoexperiment_likelihoods)
sigma = np.std(pseudoexperiment_likelihoods)



mu = np.mean(pseudoexperiment_likelihoods)
sigma = np.std(pseudoexperiment_likelihoods)
x = np.linspace(10, mu * 5, 1000)
y = stats.chi2.pdf(x,  mu)
y_gauss = stats.norm.pdf(x, np.mean(pseudoexperiment_likelihoods), np.std(pseudoexperiment_likelihoods))

Z = (data_likelihood - np.mean(pseudoexperiment_likelihoods)) / np.std(pseudoexperiment_likelihoods)

plt.plot(x, y, color = "black", label = "Chi2 distribution")
plt.plot(x, y_gauss, color = "grey", label = "Gaussian distribution")

plt.xlabel("Log likelihood")
# plt.yscale("log")
legend_title = "Gaussian Z = {:.2f}".format(Z)
plt.legend(frameon = False, title = legend_title)


plt.show()


# In[33]:


pseudo = 0


data_dict_by_fold = all_test_data_splits[pseudo]
scores_dict_by_fold = all_scores_splits[pseudo]


all_data = np.vstack([data_dict_by_fold[i] for i in range(n_folds)])
all_scores = np.vstack([scores_dict_by_fold[i].reshape(-1,1) for i in range(n_folds)]) 
all_masses = scaler.inverse_transform(all_data[:,-1].reshape(-1,1))



in_SR = (all_masses >= SR_left ) & (all_masses <= SR_right)
in_SBL = (all_masses < SR_left )
in_SBH = (all_masses > SR_right )

mass_SBL = all_masses[in_SBL]
mass_SR = all_masses[in_SR]
mass_SBH = all_masses[in_SBH]

scores_SBL = all_scores[in_SBL]
scores_SR = all_scores[in_SR]
scores_SBH = all_scores[in_SBH]


filtered_masses = np.concatenate((mass_SBL, mass_SR, mass_SBH))

popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(filtered_masses, "quintic", SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)
s, b, q0 = calculate_test_statistic(filtered_masses, starting_guess = popt, verbose_plot = True)


# In[77]:


plt.hist(scores_SR, bins = 50, histtype = 'step', color = 'black', label = 'SR')
print("mean: ", np.mean(scores_SR))


# In[80]:


likelihood_ratios = (scores_SR) / (1 - scores_SR)
plt.hist(likelihood_ratios, bins = 50, histtype = 'step', color = 'blue', alpha = 0.5, label = 'q0', density = True)
log_likelihood_ratios = -2 * np.log(likelihood_ratios)
log_likelihood = np.sum(log_likelihood_ratios)
# print(log_likelihood_ratios)
print(log_likelihood_ratios.sum())
print(log_likelihood_ratios.mean())
# print("Number of events in SR: ", len(scores_SR))
# label = "-2 Likelihood ratio: " + str(log_likelihood)
# plt.hist(log_likelihood_ratios, bins = 50, histtype = 'step', color = 'blue', alpha = 0.5, label = 'q0', density = True)
# plt.legend(title = label)


# In[31]:


n_folds = 5

"""
CALCULATE THE ROC CURVES
"""

# determine fpr thresholds as before
# yes this is repeated code
fpr_thresholds_finegrained = np.logspace(0, -4, 25)
#fpr_thresholds = np.linspace(1, 0 , 50)

plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR = num_bins_SR)


# first determine score cutoffs
score_cutoffs_finegrained = {pseudo_e:{i:{threshold:0 for threshold in fpr_thresholds_finegrained} for i in range(n_folds)} for pseudo_e in range(num_pseudoexperiments)}

for pseudo_e in range(num_pseudoexperiments):
    for i_fold in range(n_folds):
        loc_scores_sorted = np.sort(1.0-all_alt_scores_splits[pseudo_e]["FPR_validation"][i_fold])
        for threshold in fpr_thresholds_finegrained:
            loc_score_cutoff = 1-loc_scores_sorted[min(int(threshold*len(loc_scores_sorted)),len(loc_scores_sorted)-1)]
            score_cutoffs_finegrained[pseudo_e][i_fold][threshold] = loc_score_cutoff



n_folds = 5

"""
CALCULATE THE ROC CURVES
"""

# determine fpr thresholds as before
# yes this is repeated code
fpr_thresholds_finegrained = np.logspace(0, -4, 25)
#fpr_thresholds = np.linspace(1, 0 , 50)

plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR)


# first determine score cutoffs
score_cutoffs_finegrained = {pseudo_e:{i:{threshold:0 for threshold in fpr_thresholds_finegrained} for i in range(n_folds)} for pseudo_e in range(num_pseudoexperiments)}

for pseudo_e in range(num_pseudoexperiments):
    for i_fold in range(n_folds):
        loc_scores_sorted = np.sort(1.0-all_alt_scores_splits[pseudo_e]["FPR_validation"][i_fold])
        for threshold in fpr_thresholds_finegrained:
            loc_score_cutoff = 1-loc_scores_sorted[min(int(threshold*len(loc_scores_sorted)),len(loc_scores_sorted)-1)]
            score_cutoffs_finegrained[pseudo_e][i_fold][threshold] = loc_score_cutoff


        
def classifier_metrics_high_stats(dataset_by_pseudo_e, scores_by_pseudo_e, mass_scalar):

    if fit_type == "cubic": degree = 3
    if fit_type == "quintic": degree = 5
    if fit_type == "septic": degree = 7
            

    num_experiments = num_pseudoexperiments
    S_yield, B_yield = np.empty((1, num_experiments)), np.empty((1, num_experiments))
    significances = np.empty((1, num_experiments))


    for pseudo_e in range(num_experiments):

        data_dict_by_fold = dataset_by_pseudo_e[pseudo_e]
        scores_dict_by_fold = scores_by_pseudo_e[pseudo_e]

        all_data = np.vstack([data_dict_by_fold[i] for i in range(n_folds)])
        all_scores = np.vstack([scores_dict_by_fold[i].reshape(-1,1) for i in range(n_folds)]) 
        all_masses = mass_scalar.inverse_transform(all_data[:,-1].reshape(-1,1))
        in_SR = (all_masses >= SR_left ) & (all_masses <= SR_right)
        in_SBL = (all_masses < SR_left )
        in_SBH = (all_masses > SR_right )

        mass_SBL = all_masses[in_SBL]
        mass_SR = all_masses[in_SR]
        mass_SBH = all_masses[in_SBH]

        feature_SBL = all_scores[in_SBL]
        feature_SR = all_scores[in_SR]
        feature_SBH = all_scores[in_SBH]
        
        # Get a list of all possible cuts for the feature
        feature_cut_points = np.linspace(np.min(all_scores), np.max(all_scores), 10000)

         # For each cut, calculate the number of signal and background events in the SR
        num_in_SBL = []
        num_in_SR = []
        num_in_SBH = []
        FPR = []
        for cut in feature_cut_points:
            num_in_SBL.append(np.sum(feature_SBL >= cut)/len(feature_SBL))
            num_in_SR.append(np.sum(feature_SR >= cut)/len(feature_SR))
            num_in_SBH.append(np.sum(feature_SBH >= cut)/len(feature_SBH))

            FPR.append((np.sum(feature_SBH >= cut)+np.sum(feature_SBL >= cut))/(len(feature_SBH)+len(feature_SBL)))


        fit_function = bkg_fit_quintic


        print(f"On pseudo experiment {pseudo_e+1}...")
        fpr_thresholds_test = [fpr_thresholds_finegrained[FPR_index],]
        for t, threshold in enumerate(fpr_thresholds_test ):


            # Use interpolation to find the cut point that gives the desired FPR
            best_feature_cut = feature_cut_points[np.argmin(np.abs(np.array(FPR)-threshold))]


            mass_SBL_cut = mass_SBL[feature_SBL >= best_feature_cut]
            mass_SR_cut = mass_SR[feature_SR >= best_feature_cut]
            mass_SBH_cut = mass_SBH[feature_SBH >= best_feature_cut]

            # Concatenate to get the full mass spectrum
            filtered_masses = np.concatenate((mass_SBL_cut, mass_SR_cut, mass_SBH_cut))
            

            # get the fit function to SB background

            # calculate significance of bump
            popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(filtered_masses, fit_type, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)
            S,B, q0 = calculate_test_statistic(filtered_masses, SR_left, SR_right, SB_left, SB_right, degree, starting_guess = popt, verbose_plot = False)


            significances[t, pseudo_e] = np.sqrt(q0)

        
    
    return significances


# In[32]:


# significancess = TEST_classifier_metrics_high_stats(all_test_data_splits, all_scores_splits, scaler)
# save_dir = f"plot_data/fpr_tests"
# with open(f"{save_dir}/significances_{FPR_index}", "wb") as ofile:
#     pickle.dump(significancess, ofile)


# In[ ]:





# In[154]:


significances = significancess[0][1:]

key = "CATHODE"
z_score = significances
z_score = np.sort(z_score)
for i in range(len(z_score)):
    print(i, z_score[i])

# Print the median, 1sigma, and 2sigma values
print("Median: ", np.median(z_score))
print("1sigma: ", np.percentile(z_score, 16), np.percentile(z_score, 84))
print("2sigma: ", np.percentile(z_score, 2.5), np.percentile(z_score, 97.5))

# Print the percentile of z = 0, 1, 2
index = np.searchsorted(z_score, [0, 1, 2])
print("Percentile of 0: ", index[0] / len(z_score))
print("Percentile of Z = 1: ", index[1] / len(z_score))
print("Percentile of Z = 2: ", index[2] / len(z_score))

counts, bins = np.histogram(significances**2, bins = 15, density = True, range = (0, 4))
centers = (bins[:-1] + bins[1:]) / 2

plt.hist(significances**2, bins = 15, density = True, alpha = 0.5, range = (0, 4))

def f(q):
    return 1/(2*np.sqrt(q * 2 * np.pi)) * np.exp(-q/2)

plt.plot(centers, f(centers), color = "red")
plt.yscale("log")
plt.xlabel("q0")

text = r"Asymptotic $f(q) = \frac{1}{2}\delta(q) + \frac{1}{2}\frac{1}{\sqrt{2 \pi q}}e^{-q/2}$"
plt.text(0.5, 0.1, text, fontsize = 14)


# In[155]:


# significances = significancess[1][1:]

# key = "CATHODE"
# print(significances)
# z_score = significances
# z_score = np.sort(z_score)

# # Print the median, 1sigma, and 2sigma values
# print("Median: ", np.median(z_score))
# print("1sigma: ", np.percentile(z_score, 16), np.percentile(z_score, 84))
# print("2sigma: ", np.percentile(z_score, 2.5), np.percentile(z_score, 97.5))

# # Print the percentile of z = 0, 1, 2
# index = np.searchsorted(z_score, [0, 1, 2])
# print("Percentile of 0: ", index[0] / len(z_score))
# print("Percentile of Z = 1: ", index[1] / len(z_score))
# print("Percentile of Z = 2: ", index[2] / len(z_score))

# # Percage of signifcances that are 0
# print("Percentile of 0: ", np.sum(z_score == 0) / len(z_score))

# counts, bins = np.histogram(significances**2, bins = 15, density = True, range = (0, 4))
# centers = (bins[:-1] + bins[1:]) / 2

# plt.hist(significances**2, bins = 15, density = True, alpha = 0.5, range = (0, 4))

# def f(q):
#     return 1/(2*np.sqrt(q * 2 * np.pi)) * np.exp(-q/2)

# plt.plot(centers, f(centers), color = "red")
# plt.yscale("log")
# plt.xlabel("q0")

# text = r"Asymptotic $f(q) = \frac{1}{2}\delta(q) + \frac{1}{2}\frac{1}{\sqrt{2 \pi q}}e^{-q/2}$"
# plt.text(0.5, 0.1, text, fontsize = 14)


# In[156]:


# significances = significancess[2][1:]

# key = "CATHODE"
# print(significances)
# z_score = significances
# z_score = np.sort(z_score)

# # Print the median, 1sigma, and 2sigma values
# print("Median: ", np.median(z_score))
# print("1sigma: ", np.percentile(z_score, 16), np.percentile(z_score, 84))
# print("2sigma: ", np.percentile(z_score, 2.5), np.percentile(z_score, 97.5))

# # Print the percentile of z = 0, 1, 2
# index = np.searchsorted(z_score, [0, 1, 2])
# print("Percentile of 0: ", index[0] / len(z_score))
# print("Percentile of Z = 1: ", index[1] / len(z_score))
# print("Percentile of Z = 2: ", index[2] / len(z_score))

# # Percage of signifcances that are 0
# print("Percentile of 0: ", np.sum(z_score == 0) / len(z_score))

# counts, bins = np.histogram(significances**2, bins = 15, density = True, range = (0, 4))
# centers = (bins[:-1] + bins[1:]) / 2

# plt.hist(significances**2, bins = 15, density = True, alpha = 0.5, range = (0, 4))

# def f(q):
#     return 1/(2*np.sqrt(q * 2 * np.pi)) * np.exp(-q/2)

# plt.plot(centers, f(centers), color = "red")
# plt.yscale("log")
# plt.xlabel("q0")

# text = r"Asymptotic $f(q) = \frac{1}{2}\delta(q) + \frac{1}{2}\frac{1}{\sqrt{2 \pi q}}e^{-q/2}$"
# plt.text(0.5, 0.1, text, fontsize = 14)


# In[6]:


import matplotlib.pyplot as plt
import numpy as np

med = [.026042696390578803, 0.0, 0.04685278510245357]
one_sigma = [1.0583839738933054,  0.8671168065765918, 1.1003377022891243]
two_sigma = [1.8873922801335499, 1.8317919319818516, 1.8755455724561099]

fprs  = [0.01, 0.1, 1]
# plot
plt.plot(fprs, med, label = "Median", marker = "o", color = "black")

# fill band
plt.fill_between(fprs, 0, one_sigma, alpha = 0.25, label = "1$\sigma$", color = "black")
plt.fill_between(fprs, 0, two_sigma, alpha = 0.25, label = "2$\sigma$", color = "black")

plt.xscale("log")


# In[14]:


import matplotlib.pyplot as plt
import numpy as np
import pickle

fpr_thresholds = np.logspace(0, -4, 25)
significances_all = []
index_list = []
for i in range(25):

    try:
        with open(f"plot_data/fpr_tests/significances_{i}", "rb") as ifile:
            # pickle.load(ifile)
            # print(f"Found {i}")
            significances = pickle.load(ifile)[0]
            index_list.append(i)
            significances_all.append(significances)
    except:
        print(f"Could not find {i}")


significances_all = np.nan_to_num(np.array(significances_all))
print(index_list)
significances_obs = significances_all[:,0]
significances_pseudo = significances_all[:,1:]

med = np.median(significances_pseudo, axis = 1)
one_sigma_upper = np.percentile(significances_pseudo, 16, axis = 1), np.percentile(significances_pseudo, 84, axis = 1)
one_sigma_lower = np.percentile(significances_pseudo, 84, axis = 1), np.percentile(significances_pseudo, 16, axis = 1)
two_sigma_upper = np.percentile(significances_pseudo, 2.5, axis = 1), np.percentile(significances_pseudo, 97.5, axis = 1)
two_sigma_lower = np.percentile(significances_pseudo, 97.5, axis = 1), np.percentile(significances_pseudo, 2.5, axis = 1)
three_sigma_upper = np.percentile(significances_pseudo, 0.15, axis = 1), np.percentile(significances_pseudo, 99.85, axis = 1)
three_sigma_lower = np.percentile(significances_pseudo, 99.85, axis = 1), np.percentile(significances_pseudo, 0.15, axis = 1)

print(significances_all[-2,:])

plt.plot(fpr_thresholds[index_list], med, label = "Null (CATHODE)", marker = "o", color = "black")
plt.fill_between(fpr_thresholds[index_list], one_sigma_lower[0], one_sigma_upper[0], alpha = 0.125, color = "black")
plt.fill_between(fpr_thresholds[index_list], two_sigma_lower[0], two_sigma_upper[0], alpha = 0.125, color = "black")
plt.fill_between(fpr_thresholds[index_list], three_sigma_lower[0], three_sigma_upper[0], alpha = 0.125, color = "black")

plt.plot(fpr_thresholds[index_list], significances_obs, label = "Observed (CATHODE)", marker = "o", color = "red")

plt.axhline(0, color = "black", linestyle = "--", alpha = 0.5)
plt.axhline(1, color = "black", linestyle = "--", alpha = 0.5)
plt.axhline(2, color = "black", linestyle = "--", alpha = 0.5)
plt.axhline(3, color = "black", linestyle = "--", alpha = 0.5)

plt.xlim(1e-4, 1)

plt.xscale("log")
plt.xlabel("FPR")
plt.ylabel("Significance")
plt.legend(frameon = False)



# In[ ]:




