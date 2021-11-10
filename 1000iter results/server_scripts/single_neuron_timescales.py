#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:03:33 2021

@author: zachz
"""

#%% Imports

import numpy as np
import scipy.io as spio
from scipy.optimize import curve_fit
import warnings
import random
import pandas as pd
warnings.filterwarnings("ignore")

# Taus for each unit

def get_single_unit_timescales(path_to_data,dataset,species,brain_area):
    
    """
    
    Compute exponentially fit timescales for each single unit in a dataset 
    using spiketimes only data
    
    Spikes are binned at 50ms, then "fake" trials are computed with 2-5sec iti
    
    Autocorrelation, then curve fitting
    
    Arguments: path_to_data: string path to .mat file with variable 'spikes'
               dataset: string name of dataset
               species: string name of species
               brain_area: string name of brain area
    
    Output: df - pandas dataframe with columns:
                 ['dataset','species','brain_area','unit','tau','fr','r2','a','b']
                 
    """
    
    print('Now fitting:',dataset,species,brain_area)
    
    all_data = []

    failed_autocorr = []
    no_spikes_in_a_bin = []
    low_fr = []
    
    data = spio.loadmat(path_to_data,simplify_cells=True)
    
    spikes = data['spikes']
    
    for unit in range(len(spikes)):
        
        try:

            unit_spikes = spikes[unit]
        
            # 0-align first spike to make binning easier, bin @ 50ms
        
            unit_spikes = unit_spikes - np.min(unit_spikes)
        
            bins = np.arange(0,np.max(unit_spikes),step=0.05)
        
            binned_spikes, edges = np.histogram(unit_spikes,bins=bins)
        
            # Random 2-5 second fake iti
        
            binned_unit_spikes = []
            
            max_trials = int(np.max(unit_spikes)/5)
            
            start_times = np.empty(max_trials)
            start_times[0] = 0
            
            for i in range(1, max_trials):
                start_times[i] = start_times[i-1] + random.randint(2,5)
                
            start_times = start_times * 20
            start_times = start_times.astype(int)
        
            for start_t in start_times:
        
                trial_spikes = binned_spikes[start_t:start_t+19]
        
                binned_unit_spikes.append(trial_spikes)
        
            binned_unit_spikes = binned_unit_spikes[:-1]
        
            binned_unit_spikes = np.vstack(binned_unit_spikes)
        
            [trials,bins] = binned_unit_spikes.shape
        
            summed_spikes_per_bin = np.sum(binned_unit_spikes,axis=0)
        
            # Do autocorrelation
            
            try:
        
                one_autocorrelation = []
            
                for i in range(bins):
                    for j in range(bins):
                        ref_window = binned_unit_spikes[:,i]
                        test_window = binned_unit_spikes[:,j]
            
                        correlation = np.corrcoef(ref_window,test_window)[0,1]
            
                        one_autocorrelation.append(correlation)
            
                if np.isnan(one_autocorrelation).any() == True:
            
                    failed_autocorr.append(unit) # skip this unit if any autocorrelation fails
            
                elif [summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))] == True:
            
                    no_spikes_in_a_bin.append(unit) # skip this unit if any bin doesn't have spikes
            
                elif np.sum(summed_spikes_per_bin) < 1:
            
                    low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1
            
                else:
            
                    # Reshape list of autocorrelations into 19x19 matrix, plot it
            
                    correlation_matrix = np.reshape(one_autocorrelation,(-1,19))
            
                    # Fit exponential decay
            
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
                    
                    # Remove 0 lag time and sort values for curve fitting
            
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
            
                    from statistics import mean
                    from itertools import groupby
            
                    grouper = groupby(sorted_pairs, key=lambda x: x[0])
                    #The next line is again more elegant, but slower:
                    mean_pairs = [[x, mean(yi[1] for yi in y)] for x,y in grouper]
            
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
            
                    except:
                        print("Error - curve_fit failed")
                        
                    r2 = (np.corrcoef(y_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars)))[0,1]**2
                    
                    # Add data to 'all_data'
                    
                    all_data.append((dataset,species,brain_area,unit,pars[1],np.sqrt(cov[1,1]),np.sum(summed_spikes_per_bin)/trials,r2,pars[0],pars[2]))
                
            except:
                
                pass
            
        except:
        
            pass
                
    df = pd.DataFrame(all_data,columns=['dataset','species','brain_area','unit','tau','sd_tau','fr','r2','a','b'])
    
    return df
            