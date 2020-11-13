#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:37:56 2020

@author: zachz
"""

#%% Imports

import numpy as np
import scipy.io as spio
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%% Load in data

dacc = spio.loadmat('/Users/zachz/Dropbox/Timescales across species/By trial/Minxha - Human MFC/dACC.mat',simplify_cells=True)

#%% Extract spiking data from one brain area

spikes = dacc['spikes']

all_means_dacc = []
dacc_taus = []

for unit in range(len(spikes)):

        this_unit = spikes[unit]
        
        # Bin spikes from first second in 50 ms bins
        
        bins = np.arange(0,1,step=0.05)
        
        binned_unit_spikes = []
        
        for trial in range(len(this_unit)):
            
            binned, bin_edges = np.histogram(this_unit[trial],bins=bins)
            
            binned_unit_spikes.append(binned)
            
        binned_unit_spikes = np.vstack(binned_unit_spikes)
        
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
            
            pass
        
        else:
                
            #%% Reshape list of autocorrelations into 19x19 matrix, plot it
            
            correlation_matrix = np.reshape(one_autocorrelation,(-1,19))
            
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
            # plt.title('Human dACC unit %i' %unit)
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
            
            def func(x,a,tau,b):
                return a*((np.exp(-x/tau))+b)


            try:
                pars,cov = curve_fit(func,x_m,y_m,p0=[1,100,1],bounds=((0,np.inf)),maxfev=5000)
                
            except RuntimeError:
                print("Error - curve_fit failed")
            
            dacc_taus.append(pars[1])
            
            all_means_dacc.append(y_m)
            
            # plt.plot(x_m,y_m,'ro')
            # plt.xlabel('lag (ms)')
            # plt.ylabel('mean autocorrelation')
            # plt.title('Human dACC %i' %unit)
            # plt.show()
        
#%% Take mean of all units

all_means_dacc = np.vstack(all_means_dacc)

mean_dacc = np.mean(all_means_dacc,axis=0)
sd_dacc = np.std(all_means_dacc,axis=0)
se_dacc = sd_dacc/np.sqrt(len(mean_dacc))

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

pars_dacc,cov = curve_fit(func,x_m,mean_dacc,p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,mean_dacc,label='original data')
plt.plot(x_m,func(x_m,*pars_dacc),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Mean of all human dACC units')
plt.text(710,0.055,'tau = %i' %pars_dacc[1])
plt.show()

#%% Histogram of taus

plt.hist(dacc_taus)
plt.xlabel('tau')
plt.ylabel('count')
plt.title('%i human dACC units' %len(dacc_taus))
plt.show()