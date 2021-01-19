#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:37:56 2020

@author: zachz
"""

#%% Imports

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('seaborn')

#%% Load in data

amy = spio.loadmat('/Users/zachz/Dropbox/Timescales across species/By trial/Minxha - Human MFC/amygdala.mat',simplify_cells=True)

#%% Extract spiking data from one brain area

spikes = amy['spikes']

minxha_amyg_all_means = []
minxha_amyg_taus = []
minxha_amyg_failed_fits = []


minxha_amyg_failed_autocorr = []
minxha_amyg_no_spikes_in_a_bin = []
minxha_amyg_low_fr = []

minxha_amyg_avg_fr = []
minxha_amyg_correlation_matrices = []

for unit in range(len(spikes)):

        this_unit = spikes[unit]

        # Bin spikes from first second in 50 ms bins

        bins = np.arange(0,1,step=0.05)

        binned_unit_spikes = []

        for trial in range(len(this_unit)):

            binned, bin_edges = np.histogram(this_unit[trial],bins=bins)

            binned_unit_spikes.append(binned)

        binned_unit_spikes = np.vstack(binned_unit_spikes)

        [trials,bins] = binned_unit_spikes.shape

        summed_spikes_per_bin = np.sum(binned_unit_spikes,axis=0)

        minxha_amyg_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

        #%% Do autocorrelation

        one_autocorrelation = []

        for i in range(bins):
            for j in range(bins):
                ref_window = binned_unit_spikes[:,i]
                test_window = binned_unit_spikes[:,j]

                correlation = np.corrcoef(ref_window,test_window)[0,1]

                one_autocorrelation.append(correlation)

        if np.isnan(one_autocorrelation).any() == True:

            minxha_amyg_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

        elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
            # If there is not a spike in every bin
            minxha_amyg_no_spikes_in_a_bin.append(unit)

        elif np.sum(summed_spikes_per_bin)/trials < 1:

            minxha_amyg_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

        else:

            #%% Reshape list of autocorrelations into 19x19 matrix, plot it

            correlation_matrix = np.reshape(one_autocorrelation,(-1,19))

            minxha_amyg_correlation_matrices.append(correlation_matrix)

            # plt.imshow(correlation_matrix)
            # plt.title('Human amygdala unit %i' %unit)
            # plt.xlabel('lag')
            # plt.ylabel('lag')
            # plt.xticks(range(0,19))
            # plt.yticks(range(0,19))
            # plt.show()

            #%% Fit exponential decay

            # shift correlation matrix over so that all 1's are at x=0
            new_corr = []

            for i in range(18):
                new_corr.append(correlation_matrix[i,i:])

            new_corr = np.array(new_corr)

            new_lag = []

            for i in range(19,0,-1):
                new_lag.append(np.array(range(i)))

            new_lag = np.array(new_lag)

            new_corr = np.hstack(new_corr)
            new_lag = np.hstack(new_lag)
            new_lag = new_lag[:-1]

            # plt.scatter(new_lag,new_corr)
            # plt.ylabel('autocorrelation')
            # plt.xlabel('lag (ms)')

            #%% Remove 0 lag time and sort values for curve fitting

            no_1_corr = np.delete(new_corr,np.where(new_corr>=0.9))
            no_1_corr = no_1_corr[~ np.isnan(no_1_corr)]
            no_0_lag = np.delete(new_lag,np.where(new_lag==0))

            no_0_lag = no_0_lag * 50

            x = no_0_lag
            y = no_1_corr

            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            sorted_pairs = sorted((i,j) for i,j in zip(x,y))

            x_s = []
            y_s = []

            for q in range(len(sorted_pairs)):
                x_q = sorted_pairs[q][0]
                y_q = sorted_pairs[q][1]

                x_s.append(x_q)
                y_s.append(y_q)

            x_s = np.array(x_s)
            y_s = np.array(y_s)

            # plt.plot(x_s,y_s,'ro')
            # plt.title('Human amygdala unit %i' %unit)
            # plt.xlabel('lag (ms)')
            # plt.ylabel('autocorrelation')
            # plt.show()

            #%% get means and std

            from statistics import mean, stdev
            from itertools import groupby

            grouper = groupby(sorted_pairs, key=lambda x: x[0])
            #The next line is again more elegant, but slower:
            mean_pairs = [[x, mean(yi[1] for yi in y)] for x,y in grouper]
            std_pairs = [[x,stdev(yi[1] for yi in y)] for x,y in grouper]

            x_m = []
            y_m = []

            for w in range(len(mean_pairs)):
                x_w = mean_pairs[w][0]
                y_w = mean_pairs[w][1]

                x_m.append(x_w)
                y_m.append(y_w)

            x_m = np.array(x_m)
            y_m = np.array(y_m)

            # Only start fitting when slope is decreasing

            diff = np.diff(x_m)

            neg_diffs = []

            for dif in range(len(diff)):

                if diff[dif] >= 0:

                    neg_diffs.append(dif)

            first_neg_diff = np.min(neg_diffs)

            first_neg_diff = int(first_neg_diff)

            def func(x,a,tau,b):
                return a*((np.exp(-x/tau))+b)

            try:
                pars,cov = curve_fit(func,x_m[first_neg_diff:],y_m[first_neg_diff:],p0=[1,100,1],bounds=((0,np.inf)),maxfev=5000)

            except RuntimeError:
                print("Error - curve_fit failed")
                minxha_amyg_failed_fits.append(unit)

            minxha_amyg_taus.append(pars[1])

            minxha_amyg_all_means.append(y_m)

            # plt.plot(x_m,y_m,'ro',label='original data')
            # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
            # plt.xlabel('lag (ms)')
            # plt.ylabel('mean autocorrelation')
            # plt.title('Human amygdala %i' %unit)
            # plt.legend()
            # plt.show()


#%% How many units got filtered?

bad_units = len(minxha_amyg_failed_autocorr) + len(minxha_amyg_no_spikes_in_a_bin) + len(minxha_amyg_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(spikes))

#%% Take mean of all units

minxha_amyg_all_means = np.vstack(minxha_amyg_all_means)

minxha_amyg_mean = np.mean(minxha_amyg_all_means,axis=0)
minxha_amyg_sd = np.std(minxha_amyg_all_means,axis=0)
minxha_amyg_se = minxha_amyg_sd/np.sqrt(len(minxha_amyg_mean))

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(minxha_amyg_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

minxha_amyg_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],minxha_amyg_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,minxha_amyg_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*minxha_amyg_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Faraut human amygdala units \n Minxha')
plt.text(710,0.04,'tau = %i ms \n fr = %.2f hz \n n = %i' % (minxha_amyg_pars[1],minxha_amyg_mean_fr,len(minxha_amyg_taus)))
plt.ylim((0,0.07))
plt.show()

#%% Histogram of taus

plt.hist(np.log(minxha_amyg_taus))
plt.xlabel('log(tau)')
plt.ylabel('count')
plt.title('%i human amygdala units \n Minxha' %len(minxha_amyg_taus))
plt.show()

#%% Correlation matrix

minxha_amyg_mean_matrix = np.mean(minxha_amyg_correlation_matrices,axis=0)

plt.imshow(minxha_amyg_mean_matrix)
plt.tight_layout()
plt.title('Minxha Amygdala')
plt.xlabel('lag (ms)')
plt.ylabel('lag (ms)')
plt.xticks(range(0,20,2), range(0,1000,100))
plt.yticks(range(0,20,2), range(0,1000,100))
plt.colorbar()
plt.show()
