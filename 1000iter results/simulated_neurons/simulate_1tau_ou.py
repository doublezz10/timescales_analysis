#%%

import sys
sys.path.append('/Users/zachz/Documents/abcTau/abcTau')
from scipy import stats
import numpy as np
import warnings
import random

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

plt.style.use('seaborn')

from abcTau.generative_models import oneTauOU

def compare_fits_oneTauOU(tau_in,mean_in,var_in):
    
    """
    Generate a simulated timeseries using oneTauOU
    Fit it once using direct fitting - output tau and R2
    Then fit it 1000x (with a new timeseries each iter) - output all taus and R2s
    
    Inputs:
        tau_in (list length 1): set tau to simulate timeseries
        mean_in (float): average of firing rate
        var_in (float): average variance of firing rate
        
    Outputs:
        one_fit_tau (float): tau after fitting only one time (the first time)
        one_fit_r2 (float): r2 after fitting only one time (the first time)
        iter_taus (np.array): all taus from 1000 iterative fitting
        iter_r2s (np.array): all r2s from 1000 iterative fitting
    """
    
    # do this 1000x
    
    iter_taus = []
    iter_r2s = []
    
    for iter in range(1000):
        
        ts = oneTauOU(tau_in,1,50,1000,500,mean_in,var_in)[0]
    
        [trials,bins] = np.array(ts.shape)
        
        summed_spikes_per_bin = np.sum(ts,axis=0)

            
        one_autocorrelation = []

        for i in range(bins):
            for j in range(bins):
                ref_window = ts[:,i]
                test_window = ts[:,j]

                correlation = np.corrcoef(ref_window,test_window)[0,1]

                one_autocorrelation.append(correlation)
                
            # check for exclusion criteria

        if np.isnan(one_autocorrelation).any() == True:

            print('Autocorrelation calculation failed') # skip this unit if any autocorrelation fails

        elif [summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))] == True:

            print("One bin didn't have any spikes") # skip this unit if any bin doesn't have spikes

        elif np.sum(summed_spikes_per_bin) < 1:

            print("Firing rate less than 1hz") # skip this unit if avg firing rate across all trials is < 1

        else:

            correlation_matrix = np.reshape(one_autocorrelation,(-1,20))
                
            # Fit exponential decay

            # shift correlation matrix over so that all 1's are at x=0
            new_corr = []

            for i in range(19):
                new_corr.append(correlation_matrix[i,i:])

            new_corr = np.array(new_corr)

            new_lag = []

            for i in range(20,0,-1):
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

            def func(x,a,tau_,b):
                return a*((np.exp(-x/tau_))+b)

            try:
                pars,cov = curve_fit(func,x_m[first_neg_diff:],y_m[first_neg_diff:],p0=[1,100,1],bounds=((0,np.inf)),maxfev=5000)

            except:
                print("Error - curve_fit failed")
                
            r2 = (np.corrcoef(y_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars)))[0,1]**2

            iter_taus.append(pars[1])
            iter_r2s.append(r2)
            
    one_fit_tau = iter_taus[0]
    one_fit_r2 = iter_r2s[0]
    
    iter_taus = np.array(iter_taus)
    iter_r2s = np.array(iter_r2s)
    
    return one_fit_tau, one_fit_r2, iter_taus, iter_r2s
        
#%% Try and plot true tau, one_fit, and distribution all at once

tau = 50

one_fit_tau, one_fit_r2, iter_taus, iter_r2s = compare_fits_oneTauOU([tau],1,1)   

plt.hist(iter_taus,density=True)

plt.axvline(np.mean(iter_taus),label="mean of 1000iter (R$^2$ = %1.2f)" %np.mean(iter_r2s),color='red',linestyle='--',alpha=0.7)
plt.axvline(one_fit_tau,label="one fit (R$^2$ = %1.2f)" %one_fit_r2,color='green',linestyle='--',alpha=0.7)
plt.axvline(tau,label='true tau',color='yellow',linestyle='--',alpha=0.7)

plt.xlabel('timescale (ms)')
plt.ylabel('probability')

plt.title('1000 iterative fitting')

plt.legend()

plt.show()

# %% Do it again

for tau in range(10,1010,50):

    one_fit_tau, one_fit_r2, iter_taus, iter_r2s = compare_fits_oneTauOU([tau],1,1)   

    plt.hist(iter_taus,density=True)

    plt.axvline(np.mean(iter_taus),label="mean of 1000iter (R$^2$ = %1.2f)" %np.mean(iter_r2s),color='red',linestyle='--',alpha=0.7)
    plt.axvline(one_fit_tau,label="one fit (R$^2$ = %1.2f)" %one_fit_r2,color='green',linestyle='--',alpha=0.7)
    plt.axvline(tau,label='true tau',color='yellow',linestyle='--',alpha=0.7)

    plt.xlabel('timescale (ms)')
    plt.ylabel('probability')

    plt.title('1000 iterative fitting')

    plt.legend()

    plt.show()

# %%
