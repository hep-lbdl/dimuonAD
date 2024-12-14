#!/usr/bin/env python
# coding: utf-8

# In[142]:


import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from scipy.special import erfcinv
from scipy.optimize import curve_fit, minimize
from scipy import stats
from helpers.physics_functions import bkg_fit_cubic, bkg_fit_septic, bkg_fit_quintic, get_bins, select_top_events_fold, curve_fit_m_inv, calc_significance, get_errors_bkg_fit_ratio, calculate_test_statistic

import sys


# In[143]:


SR_left = 9.0
SR_right = 10.6
SB_left = 5.0
SB_right = 16.0
num_bins_SR = 12

FPR_index = int(sys.argv[1])

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

def likelihood(data, s, *theta):

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
    
def cheat_likelihood(data, *theta):

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
    
def null_hypothesis(data, *theta):
    return likelihood(data, 0, *theta)

def calculate_test_statistic(data, degree = 5, starting_guess = None, verbose_plot = False, return_popt = False):

    # We want to determine the profiled log likelihood ratio: -2 * [L(s, theta_hat_hat) - L(s_hat, theta_hat)]
    # for s = 0

    # Set up 
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    average_bin_count = len(data) / len(plot_bins)
    if starting_guess is None:
        starting_guess = [average_bin_count, 0, 0, 0, 0, 0, 0, 0, 0, 0]




    # Fit the s = 0 hypothesis
    lambda_null = lambda theta: null_hypothesis(data, *theta)
    fit = minimize(lambda_null , x0 = starting_guess, method = 'Nelder-Mead', options = {'maxiter': 15000, "disp": verbose_plot})
    theta_hat_hat = fit.x
    null_fit_likelihood = null_hypothesis(data, *theta_hat_hat)


        # Fit the s = float hypothesis
    lambda_cheat = lambda theta: cheat_likelihood(data, *theta)
    fit = minimize(lambda_cheat , x0 = theta_hat_hat, method = 'Nelder-Mead', options = {'maxiter': 15000, "disp": verbose_plot})
    theta_hat = fit.x
    integrated_background = integral(SR_left, SR_right, bin_width, *theta_hat)
    num_SR = len(data[np.logical_and(data > SR_left, data < SR_right)])
    integrated_signal = num_SR - integrated_background
    best_fit_likelihood = likelihood(data, integrated_signal, *theta_hat)


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



# q0s = []

# for i in tqdm(range(100)):
#     data = np.random.uniform(5.0, 16.0, 1000)
#     verbose_plot = i == 0
#     s, b, q0 = calculate_test_statistic(data, degree = 5, verbose_plot = verbose_plot)
#     q0s.append(q0)
#     print(i, q0, popt)


# In[144]:


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


# In[145]:


mean_events = 10000

def pseudoexperiment(seed = 0):

    background = 10000
    observed =  np.random.poisson(mean_events, 1)[0]

    ########## Likelihoods ##########
    def likelihood_function(b, s, N):
        return -2 * (stats.poisson.logpmf(N, b + s) )


    # Null hypothesis: s = 0 with PROFILED b. We need to minimize the likelihood to get the best fit b given the total_B and total_B_error
    def null_likelihood_function(bprime):
        return likelihood_function(bprime, 0, observed)



    minimization = minimize(null_likelihood_function, 100, bounds = [(0, None)])
    post_fit_B = minimization.x[0]

    null_likelihood = null_likelihood_function(background)
    likelihood = likelihood_function(observed, 0, observed)


    test_statistic =  (null_likelihood - likelihood)
    
    if observed < background:
        test_statistic = 0

    return test_statistic

    print("Seed: ", seed)
    print("Observed: ", observed)
    print("Best fit B: ", post_fit_B)


test_statistics = [pseudoexperiment(seed) for seed in range(1000)]
print(test_statistics)
plt.hist(test_statistics, bins = 50)


# In[146]:


z_score = np.sqrt(test_statistics)
z_score = np.sort(z_score)
plt.hist(test_statistics, bins = 50, density = True, alpha = 0.5)

counts, bins = np.histogram(test_statistics, bins = 50, density = True)
centers = (bins[:-1] + bins[1:]) / 2

def f(q):
    return 1/(2*np.sqrt(q * 2 * np.pi)) * np.exp(-q/2)

plt.plot(centers, f(centers), label = "Asymptotic", color = "red")

plt.yscale("log")
plt.xlabel("Test statistic $q_0$")

# Print the median, 1sigma, and 2sigma values
print("Median: ", np.median(z_score))
print("1sigma: ", np.percentile(z_score, 16), np.percentile(z_score, 84))
print("2sigma: ", np.percentile(z_score, 2.5), np.percentile(z_score, 97.5))

# Print the percentile of z = 0, 1, 2
index = np.searchsorted(z_score, [0, 1, 2])
print("Percentile of 0: ", index[0] / len(z_score))
print("Percentile of Z = 1: ", index[1] / len(z_score))
print("Percentile of Z = 2: ", index[2] / len(z_score))


# In[ ]:





# In[147]:


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
num_pseudoexperiments = 501 #1001
n_folds = 5

num_bins_SR = 12 # 16, 12, 8

pseudo_e_to_plot = 0 # this plots the actual data (not a boostrapped version)
fit_type = "quintic" # "cubic", "quintic", septic
if fit_type == "cubic": fit_function = bkg_fit_cubic
if fit_type == "quintic": fit_function = bkg_fit_quintic
if fit_type == "septic": fit_function = bkg_fit_septic


# In[148]:


# if train_samesign = False, this loads in the OS test data
# test 


def load_in_pseudoexperiments(file_string, num_pseudoexps):

    master_dict = {}

    with open(f"{pickle_save_dir}/{file_string}_{fit_type}_{num_bins_SR}_0_1", "rb") as ifile:
        loc_dict = pickle.load(ifile)
    master_dict = {**loc_dict}
    # load in the bootstraps
    for i in range(1, num_pseudoexps):
        with open(f"{pickle_save_dir}/bkg_samples/bootstrap{i}/{file_string}_{fit_type}_{num_bins_SR}_0_1", "rb") as ifile:
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


# In[149]:


SB_left = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SB_left"])
SR_left = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SR_left"])
SR_right = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SR_right"])
SB_right = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SB_right"])

print(SB_left, SB_right)


data_prefix = f"upsilon_{train_data_id}"
print(data_prefix)


# In[150]:


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

feature_SBL = all_scores[in_SBL]
feature_SR = all_scores[in_SR]
feature_SBH = all_scores[in_SBH]


filtered_masses = np.concatenate((mass_SBL, mass_SR, mass_SBH))

popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(filtered_masses, "quintic", SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)
s, b, q0 = calculate_test_statistic(filtered_masses, starting_guess = popt, verbose_plot = True)


# In[151]:


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

plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR = num_bins_SR)


# first determine score cutoffs
score_cutoffs_finegrained = {pseudo_e:{i:{threshold:0 for threshold in fpr_thresholds_finegrained} for i in range(n_folds)} for pseudo_e in range(num_pseudoexperiments)}

for pseudo_e in range(num_pseudoexperiments):
    for i_fold in range(n_folds):
        loc_scores_sorted = np.sort(1.0-all_alt_scores_splits[pseudo_e]["FPR_validation"][i_fold])
        for threshold in fpr_thresholds_finegrained:
            loc_score_cutoff = 1-loc_scores_sorted[min(int(threshold*len(loc_scores_sorted)),len(loc_scores_sorted)-1)]
            score_cutoffs_finegrained[pseudo_e][i_fold][threshold] = loc_score_cutoff


        
def TEST_classifier_metrics_high_stats(dataset_by_pseudo_e, scores_by_pseudo_e, mass_scalar):
            

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
            
            S,B, q0 = calculate_test_statistic(filtered_masses, starting_guess = popt, verbose_plot = False)
            print("S: ", S, "B: ", B, "q0: ", q0)

        

            # q0 = S/np.sqrt(B + total_B_error**2)
            # if S < 0 or B < 0:
                # q0 = 0

            significances[t, pseudo_e] = np.sqrt(q0)

        
    
    return significances


# In[152]:


significancess = TEST_classifier_metrics_high_stats(all_test_data_splits, all_scores_splits, scaler)
save_dir = f"plot_data/fpr_tests"
with open(f"{save_dir}/significances_{FPR_index}", "wb") as ofile:
    pickle.dump(significancess, ofile)


# In[ ]:





# In[153]:


print(significancess.shape)


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







# In[ ]:




