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

#%% Load in data

ofc = spio.loadmat('/Users/zachz/Dropbox/Timescales across species/By trial/Hunt - monkey frontal cortex/hunt_ofc.mat',simplify_cells=True)

#%% Extract spiking data from one brain area



spikes = ofc['fixation']
        
all_means_ofc_monkey = []
ofc_taus_monkey = []
ofc_failed_fits = []

ofc_no_autocorr_monkey = []
ofc_no_spikes_in_a_bin_monkey = []
ofc_low_fr_monkey = []
        
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
            
            ofc_no_autocorr_monkey.append(unit)
            
        # elif [summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))]:
            
        #     ofc_no_spikes_in_a_bin_monkey.append(unit)
            
        elif np.sum(summed_spikes_per_bin) < 1:
            
            ofc_low_fr_monkey.append(unit)
        
        else:
                
            #%% Reshape list of autocorrelations into 19x19 matrix, plot it
            
            correlation_matrix = np.reshape(one_autocorrelation,(-1,10))
           
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
                pars,cov = curve_fit(func,x_m[first_neg_diff:],y_m[first_neg_diff:],p0=[1,100,1],bounds=((0,np.inf)),maxfev=5000)
                
            except RuntimeError:
                print("Error - curve_fit failed")
                ofc_failed_fits.append(unit)
            
            ofc_taus_monkey.append(pars[1])
            
            all_means_ofc_monkey.append(y_m)
            
            plt.plot(x_m,y_m,'ro',label='original data')
            plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
            plt.xlabel('lag (ms)')
            plt.ylabel('mean autocorrelation')
            plt.title('Monkey ofc %i' %unit)
            plt.legend()
            plt.show()
        
#%% Take mean of all units

all_means_ofc_monkey = np.vstack(all_means_ofc_monkey)

mean_ofc = np.mean(all_means_ofc_monkey,axis=0)
sd_ofc = np.std(all_means_ofc_monkey,axis=0)
se_ofc = sd_ofc/np.sqrt(len(mean_ofc))

mean_diff = np.diff(mean_ofc)

neg_mean_diffs = []

for diff in range(len(mean_diff)):
    
    if mean_diff[diff] >= 0:
        
        neg_mean_diffs.append(diff)
        
first_neg_mean_diff = np.min(neg_mean_diffs)

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

pars_ofc,cov = curve_fit(func,x_m[first_neg_mean_diff:],mean_ofc[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,mean_ofc,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*pars_ofc),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Mean of all monkey ofc units')
plt.text(100,0.06,'tau = %i' %pars_ofc[1])
plt.show()

#%% Histogram of taus

plt.hist(ofc_taus_monkey)
plt.xlabel('tau')
plt.ylabel('count')
plt.title('%i monkey ofc units' %len(ofc_taus_monkey))
plt.show()

#%% How many units show initial incresae vs decrease

# first_diff_ofc = []
# second_diff_ofc = []

# units = np.size(all_means_ofc_monkey,0)

# for unit in range(units):
    
#     first_diff_ofc.append(all_means_ofc_monkey[unit,0]-all_means_ofc_monkey[unit,1])
#     second_diff_ofc.append(all_means_ofc_monkey[unit,1]-all_means_ofc_monkey[unit,2])
    
# plt.hist(first_diff_ofc,label='first diff')
# plt.hist(second_diff_ofc,label='second diff')
# plt.ylabel('count')
# plt.title('ofc')
# plt.legend()
# plt.show()

#%% Plot autocorrelation curves for units with a positive vs negative difference between first and second lags

# pos_first_diff = []
# neg_first_diff = []

# fig,axs = plt.subplots(1,2,sharey=True)

# for unit in range(units):
    
#     if first_diff_ofc[unit] <= 0:
        
#         neg_first_diff.append(unit)
        
#     else:
        
#         pos_first_diff.append(unit)
        
# for dec_unit in range(len(neg_first_diff)):
    
#     axs[0].plot(x_m,all_means_ofc_monkey[neg_first_diff[dec_unit]])
    
# axs[0].set_title('%i ofc units with \n initial increase' %len(neg_first_diff))
# axs[0].set_ylabel('autocorrelation')
# axs[0].set_xlabel('lag (ms)')

# for inc_unit in range(len(pos_first_diff)):
    
#    axs[1].plot(x_m,all_means_ofc_monkey[pos_first_diff[inc_unit]])
    
# axs[1].set_title('%i ofc units with \n initial decrease' % len(pos_first_diff))

# plt.xlabel('lag (ms)')
# plt.tight_layout()
# plt.show()