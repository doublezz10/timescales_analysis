#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:36:11 2020

@author: zachz
"""

#%% Imports

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

#%% Load in data

ofc = spio.loadmat('/Users/zachz/Dropbox/Timescales across species/By trial/Hunt - monkey frontal cortex/hunt_ofc.mat',simplify_cells=True)

#%% Extract spiking data from one brain area

spikes = ofc['fixation']

try:
   cell_info = ofc['cell_info']
except NameError:
   pass

hunt_ofc_all_means = []
hunt_ofc_taus = []
hunt_ofc_failed_fits = []

hunt_ofc_no_autocorr = []
hunt_ofc_no_spikes_in_a_bin = []
hunt_ofc_low_fr = []

hunt_ofc_avg_fr = []
hunt_ofc_correlation_matrices = []

binsize = 50

#%%

for unit in range(len(spikes)):

        this_unit = np.vstack(spikes[unit])

        [rows,cols] = this_unit.shape

        binned_unit_spikes = []

        for trials in range(0,rows):
            trial_spikes = []
            for times in range(0,550,binsize):
                spikes_ = np.sum(this_unit[trials,times:times+binsize])
                trial_spikes.append(spikes_)
            binned_unit_spikes.append(trial_spikes)
        binned_unit_spikes = np.array(binned_unit_spikes)
        binned_unit_spikes = binned_unit_spikes[:,:-1]

        summed_spikes_per_bin = np.sum(binned_unit_spikes,axis=0)

        hunt_ofc_avg_fr.append(np.sum(summed_spikes_per_bin)/rows)

        #%% Do autocorrelation

        one_autocorrelation = []

        [trials,bins] = binned_unit_spikes.shape

        for i in range(bins):
            for j in range(bins):
                ref_window = binned_unit_spikes[:,i]
                test_window = binned_unit_spikes[:,j]

                correlation = np.corrcoef(ref_window,test_window)[0,1]

                one_autocorrelation.append(correlation)

        if np.isnan(one_autocorrelation).any() == True:

            hunt_ofc_no_autocorr.append(unit)

        elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
            # If there is not a spike in every bin
            hunt_ofc_no_spikes_in_a_bin.append(unit)

        elif np.sum(summed_spikes_per_bin) < 1:

            hunt_ofc_low_fr.append(unit)

        else:

            #%% Reshape list of autocorrelations into 19x19 matrix, plot it

            correlation_matrix = np.reshape(one_autocorrelation,(-1,10))

            hunt_ofc_correlation_matrices.append(correlation_matrix)

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

            for i in range(8):
                new_corr.append(correlation_matrix[i,i:])

            new_corr = np.array(new_corr)

            new_lag = []

            for i in range(9,0,-1):
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
            # plt.title('Human ofc unit %i' %unit)
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

            def func(x,a,tau,b):
                return a*((np.exp(-x/tau))+b)

            try:
                pars,cov = curve_fit(func,x_m,y_m,p0=[1,100,1],bounds=((0,np.inf)),maxfev=5000)

            except RuntimeError:
                print("Error - curve_fit failed")
                hunt_ofc_failed_fits.append(unit)

            hunt_ofc_taus.append(pars[1])

            hunt_ofc_all_means.append(y_m)

            # plt.plot(x_m,y_m,'ro',label='original data')
            # plt.plot(x_m,func(x_m,*pars),label='fit')
            # plt.xlabel('lag (ms)')
            # plt.ylabel('mean autocorrelation')
            # plt.title('Monkey ofc %i' %unit)
            # plt.legend()
            # plt.show()

#%% How many units got filtered?

bad_units = len(hunt_ofc_no_autocorr) + len(hunt_ofc_no_spikes_in_a_bin) + len(hunt_ofc_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(spikes))

#%% Take mean of all units

hunt_ofc_all_means = np.vstack(hunt_ofc_all_means)

hunt_ofc_mean = np.mean(hunt_ofc_all_means,axis=0)
hunt_ofc_sd = np.std(hunt_ofc_all_means,axis=0)
hunt_ofc_se = hunt_ofc_sd/np.sqrt(len(hunt_ofc_mean))

hunt_ofc_mean_fr = np.mean(hunt_ofc_avg_fr)

mean_diff = np.diff(hunt_ofc_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] >= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

hunt_ofc_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],hunt_ofc_mean[first_neg_mean_diff:],bounds=((0,np.inf)),maxfev=5000)

plt.plot(x_m,hunt_ofc_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*hunt_ofc_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Mean of all monkey ofc units')
plt.text(100,0.06,'tau = %i ms \n fr = %.2f hz \n n = %i' % (hunt_ofc_pars[1],hunt_ofc_mean_fr,len(hunt_ofc_taus)))
plt.show()

#%% Histogram of taus

plt.hist(np.log(hunt_ofc_taus))
plt.axvline(hunt_ofc_pars[1],color='r',linestyle='dashed',linewidth=1)
plt.xlabel('log(tau)')
plt.ylabel('count')
plt.title('%i monkey ofc units' %len(hunt_ofc_taus))
plt.show()

#%% Correlation matrix

hunt_ofc_mean_matrix = np.mean(hunt_ofc_correlation_matrices,axis=0)

plt.imshow(hunt_ofc_mean_matrix,cmap='inferno')
plt.tight_layout()
plt.title('Hunt OFC')
plt.xlabel('lag (ms)')
plt.ylabel('lag (ms)')
plt.xticks(range(10),range(0,500,50))
plt.yticks(range(10),range(0,500,50))
plt.colorbar()
plt.show()
