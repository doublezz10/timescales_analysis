#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:22:48 2020

@author: zachz
"""

#%% Imports

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import csv
warnings.filterwarnings("ignore")

#%% Load in data

bla = spio.loadmat('/Users/zachz/Dropbox/Timescales across species/By trial/Steinmetz - mouse/bla_task_trial.mat',simplify_cells=True)

#%% Extract spiking data from one brain area

spikes = bla['bla_task_by_trial']

all_data_steinmetz_bla = []

steinmetz_bla_all_means = []
steinmetz_bla_taus = []
steinmetz_bla_failed_fits = []

steinmetz_bla_failed_autocorr = []
steinmetz_bla_no_spikes_in_a_bin = []
steinmetz_bla_low_fr = []

steinmetz_bla_avg_fr = []
steinmetz_bla_correlation_matrices = []

for unit in range(len(spikes)):

        this_unit = spikes[unit]

        # Bin spikes from first second in 50 ms bins

        bins = np.arange(0,1,step=0.05)

        binned_unit_spikes = []

        for trial in range(len(this_unit)):

            binned, bin_edges = np.histogram(this_unit[trial],bins=bins)

            binned_unit_spikes.append(binned)

        try:

            binned_unit_spikes = np.vstack(binned_unit_spikes)

            [trials,bins] = binned_unit_spikes.shape

            summed_spikes_per_bin = np.sum(binned_unit_spikes,axis=0)

            #%% Do autocorrelation

            one_autocorrelation = []

            for i in range(bins):
                for j in range(bins):
                    ref_window = binned_unit_spikes[:,i]
                    test_window = binned_unit_spikes[:,j]

                    correlation = np.corrcoef(ref_window,test_window)[0,1]

                    one_autocorrelation.append(correlation)

            if np.isnan(one_autocorrelation).any() == True:

                steinmetz_bla_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

            elif [summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))] == True:

                steinmetz_bla_no_spikes_in_a_bin.append(unit) # skip this unit if any bin doesn't have spikes

            elif np.sum(summed_spikes_per_bin) < 1:

                steinmetz_bla_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

            else:

                #%% Reshape list of autocorrelations into 19x19 matrix, plot it

                correlation_matrix = np.reshape(one_autocorrelation,(-1,19))

                steinmetz_bla_correlation_matrices.append(correlation_matrix)

                # plt.imshow(correlation_matrix)
                # plt.title('Mouse BLA unit %i' %unit)
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
                # plt.title('Mouse BLA unit %i' %unit)
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
                    steinmetz_bla_failed_fits.append(unit)

                r2 = (np.corrcoef(y_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars)))[0,1]**2

                steinmetz_bla_taus.append(pars[1])
                steinmetz_bla_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

                steinmetz_bla_all_means.append(y_m)

                # plt.plot(x_m,y_m,'ro',label='original data')
                # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit (tau = %i)' %pars[1])
                # plt.xlabel('lag (ms)')
                # plt.ylabel('mean autocorrelation')
                # plt.title('Steinmetz BLA %i' %unit)
                # plt.legend()
                # plt.show()

                #%% Add data to 'all_data'

                all_data_steinmetz_bla.append(('steinmetz','mouse','bla',unit,pars[1],np.sum(summed_spikes_per_bin)/trials,r2,pars[0],pars[2]))

        except ValueError:

            pass

with open('/Users/zachz/Documents/timescales_analysis/results.csv','a') as out:
    csv_out=csv.writer(out)
    for row in all_data_steinmetz_bla:
        csv_out.writerow(row)
        
#%% How many units got filtered?

steinmetz_bla_bad_units = len(steinmetz_bla_failed_autocorr) + len(steinmetz_bla_no_spikes_in_a_bin) + len(steinmetz_bla_low_fr)

print('%i units were filtered out' %steinmetz_bla_bad_units)
print('out of %i total units' %len(spikes))

#%% Take mean of all units

steinmetz_bla_all_means = np.vstack(steinmetz_bla_all_means)

steinmetz_bla_mean = np.mean(steinmetz_bla_all_means,axis=0)
steinmetz_bla_sd = np.std(steinmetz_bla_all_means,axis=0)
steinmetz_bla_se = steinmetz_bla_sd/np.sqrt(len(steinmetz_bla_mean))

steinmetz_bla_mean_fr = np.mean(steinmetz_bla_avg_fr)

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(steinmetz_bla_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

steinmetz_bla_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],steinmetz_bla_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,steinmetz_bla_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*steinmetz_bla_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Mean of all mouse BLA units \n Steinmetz')
plt.text(710,0.12,'tau = %i ms \n fr = %.2f hz \n n = %i' % (steinmetz_bla_pars[1],steinmetz_bla_mean_fr,len(steinmetz_bla_taus)))
plt.show()

a_population_steinmetz_bla = (('steinmetz','bla',steinmetz_bla_pars[1],steinmetz_bla_mean_fr,len(steinmetz_bla_taus)))

#%% Add error bars

plt.errorbar(x_m, steinmetz_bla_mean, yerr=steinmetz_bla_se, label='data +/- se')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*steinmetz_bla_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('autocorrelation')
plt.title('Mean of all mouse BLA units \n Steinmetz')
plt.text(710,0.03,'tau = %i ms \n fr = %.2f hz \n n = %i' % (steinmetz_bla_pars[1],steinmetz_bla_mean_fr,len(steinmetz_bla_taus)))
plt.ylim((0,0.16))
plt.show()

#%% Histogram of taus

bins = 10**np.arange(0,4,0.1)

plt.hist(steinmetz_bla_taus,bins=bins, weights=np.zeros_like(steinmetz_bla_taus) + 1. / len(steinmetz_bla_taus))
plt.axvline(steinmetz_bla_pars[1],color='r',linestyle='dashed',linewidth=1)
plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('%i Mouse BLA units \n Steinmetz' %len(steinmetz_bla_taus))
plt.show()

#%% Correlation matrix

steinmetz_bla_mean_matrix = np.mean(steinmetz_bla_correlation_matrices,axis=0)

plt.imshow(steinmetz_bla_mean_matrix)
plt.tight_layout()
plt.title('Steinmetz BLA')
plt.xlabel('lag (ms)')
plt.ylabel('lag (ms)')
plt.xticks(range(0,20,2),range(0,1000,100))
plt.yticks(range(0,20,2),range(0,1000,100))
plt.colorbar()
plt.show()
