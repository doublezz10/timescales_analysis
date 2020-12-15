#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:42:11 2020

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

#%% Load data

froot_pre = pd.read_csv('/Users/zachz/Dropbox/Timescales across species/By trial/FROOT/pre_foreperiod.txt',delimiter='\s+',header=None,names=['pre/post','unit_name','date','?','??','???','????','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
froot_post = pd.read_csv('/Users/zachz/Dropbox/Timescales across species/By trial/FROOT/post_foreperiod.txt',delimiter='\s+',header=None,names=['pre/post','unit_name','date','?','??','???','????','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])

brain_areas = pd.read_csv('/Users/zachz/Dropbox/Timescales across species/By trial/FROOT/brain_regions.csv')

# Now each row is a trial (all units mixed), and each column is num of spikes in 50 ms bin

#%% Group by unit name and brain area

# Pre surgery

froot_pre_units = froot_pre.unit_name.unique()

pre_brain_area_1 = []
pre_brain_area_2 = []

for unique_unit in range(len(froot_pre_units)):

    one_unit = froot_pre.loc[froot_pre['unit_name'] == froot_pre_units[unique_unit]]

    one_unit_spikes = one_unit[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']].to_numpy() / 20.41

    this_brain_area = brain_areas.loc[brain_areas['unit_name'] == froot_pre_units[unique_unit]]

    this_brain_area = this_brain_area.to_numpy()

    this_brain_area = this_brain_area[0,2]

    if this_brain_area == 1:

        pre_brain_area_1.append(one_unit_spikes)

    elif this_brain_area == 2:

        pre_brain_area_2.append(one_unit_spikes)

# Post surgery

froot_post_units = froot_post.unit_name.unique()

post_brain_area_1 = []
post_brain_area_2 = []

for unique_unit in range(len(froot_post_units)):

    one_unit = froot_post.loc[froot_post['unit_name'] == froot_post_units[unique_unit]]

    one_unit_spikes = one_unit[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']].to_numpy() / 20.41

    this_brain_area = brain_areas.loc[brain_areas['unit_name'] == froot_post_units[unique_unit]]

    this_brain_area = this_brain_area.to_numpy()

    this_brain_area = this_brain_area[0,2]

    if this_brain_area == 1:

        post_brain_area_1.append(one_unit_spikes)

    elif this_brain_area == 2:

        post_brain_area_2.append(one_unit_spikes)

#%% Start computing autocorrelation

# Pre Brain Area 1

pre_1_all_means = []
pre_1_taus = []
pre_1_failed_fits = []

pre_1_failed_autocorr = []
pre_1_no_spikes_in_a_bin = []
pre_1_low_fr = []

pre_1_avg_fr = []
pre_1_correlation_matrices = []

for unit in range(len(pre_brain_area_1)):

    binned_spikes = pre_brain_area_1[unit]

    [trials,bins] = binned_spikes.shape

    summed_spikes_per_bin = np.sum(binned_spikes,axis=0)

    pre_1_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

    #%% Do autocorrelation

    one_autocorrelation = []

    for i in range(bins):
        for j in range(bins):
            ref_window = binned_spikes[:,i]
            test_window = binned_spikes[:,j]

            correlation = np.corrcoef(ref_window,test_window)[0,1]

            one_autocorrelation.append(correlation)

    if np.isnan(one_autocorrelation).any() == True:

        pre_1_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

    elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
        # If there is not a spike in every bin
        pre_1_no_spikes_in_a_bin.append(unit)

    elif np.sum(summed_spikes_per_bin)/trials < 1:

        pre_1_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

    else:

        #%% Reshape list of autocorrelations into 19x19 matrix, plot it

        correlation_matrix = np.reshape(one_autocorrelation,(-1,20))

        pre_1_correlation_matrices.append(correlation_matrix)

        # plt.imshow(correlation_matrix)
        # plt.title('Froot pre 1 unit %i' %unit)
        # plt.xlabel('lag')
        # plt.ylabel('lag')
        # plt.xticks(range(0,19))
        # plt.yticks(range(0,19))
        # plt.show()

        #%% Fit exponential decay

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
        # plt.title('Froot pre 1 unit %i' %unit)
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
            print("Error - curve_fit failed; unit %i" %unit)
            pre_1_failed_fits.append(unit)

        pre_1_taus.append(pars[1])

        pre_1_all_means.append(y_m)

        # plt.plot(x_m,y_m,'ro')
        # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
        # plt.xlabel('lag (ms)')
        # plt.ylabel('mean autocorrelation')
        # plt.title('Froot 1 pre unit %i' %unit)
        # plt.show()


#%% How many units got filtered?

bad_units = len(pre_1_failed_autocorr) + len(pre_1_no_spikes_in_a_bin) + len(pre_1_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(pre_brain_area_1))

#%% Take mean of all units

pre_1_all_means = np.vstack(pre_1_all_means)

pre_1_mean = np.mean(pre_1_all_means,axis=0)
pre_1_sd = np.std(pre_1_all_means,axis=0)
pre_1_se = pre_1_sd/np.sqrt(len(pre_1_mean))

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(minxha_dacc_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

pre_1_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],pre_1_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,pre_1_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*pre_1_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Froot 1 pre')
plt.text(710,0.1,'tau = %i' %pre_1_pars[1])
plt.show()

#%% Histogram of taus

plt.hist(pre_1_taus)
plt.xlabel('tau')
plt.ylabel('count')
plt.title('%i pre 1 Froot units' %len(pre_1_taus))
plt.show()

#%% Compute autocorrelation

# Pre Brain Area 1

pre_2_all_means = []
pre_2_taus = []
pre_2_failed_fits = []

pre_2_failed_autocorr = []
pre_2_no_spikes_in_a_bin = []
pre_2_low_fr = []

pre_2_avg_fr = []
pre_2_correlation_matrices = []

for unit in range(len(pre_brain_area_2)):

    binned_spikes = pre_brain_area_2[unit]

    [trials,bins] = binned_spikes.shape

    summed_spikes_per_bin = np.sum(binned_spikes,axis=0)

    pre_2_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

    #%% Do autocorrelation

    one_autocorrelation = []

    for i in range(bins):
        for j in range(bins):
            ref_window = binned_spikes[:,i]
            test_window = binned_spikes[:,j]

            correlation = np.corrcoef(ref_window,test_window)[0,1]

            one_autocorrelation.append(correlation)

    if np.isnan(one_autocorrelation).any() == True:

        pre_2_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

    elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
        # If there is not a spike in every bin
        pre_2_no_spikes_in_a_bin.append(unit)

    elif np.sum(summed_spikes_per_bin)/trials < 1:

        pre_2_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

    else:

        #%% Reshape list of autocorrelations into 19x19 matrix, plot it

        correlation_matrix = np.reshape(one_autocorrelation,(-1,20))

        pre_2_correlation_matrices.append(correlation_matrix)

        # plt.imshow(correlation_matrix)
        # plt.title('Froot pre 2 unit %i' %unit)
        # plt.xlabel('lag')
        # plt.ylabel('lag')
        # plt.xticks(range(0,19))
        # plt.yticks(range(0,19))
        # plt.show()

        #%% Fit exponential decay

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
        # plt.title('Froot pre 2 unit %i' %unit)
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
            print("Error - curve_fit failed; unit %i" %unit)
            pre_2_failed_fits.append(unit)

        pre_2_taus.append(pars[1])

        pre_2_all_means.append(y_m)

        # plt.plot(x_m,y_m,'ro')
        # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
        # plt.xlabel('lag (ms)')
        # plt.ylabel('mean autocorrelation')
        # plt.title('Froot 2 pre unit %i' %unit)
        # plt.show()


#%% How many units got filtered?

bad_units = len(pre_2_failed_autocorr) + len(pre_2_no_spikes_in_a_bin) + len(pre_2_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(pre_brain_area_1))

#%% Take mean of all units

pre_2_all_means = np.vstack(pre_2_all_means)

pre_2_mean = np.mean(pre_2_all_means,axis=0)
pre_2_sd = np.std(pre_2_all_means,axis=0)
pre_2_se = pre_2_sd/np.sqrt(len(pre_2_mean))

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(minxha_dacc_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

pre_2_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],pre_2_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,pre_2_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*pre_2_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Froot 2 pre')
plt.text(710,0.1,'tau = %i' %pre_2_pars[1])
plt.show()

#%% Histogram of taus

plt.hist(pre_2_taus)
plt.xlabel('tau')
plt.ylabel('count')
plt.title('%i pre 2 Froot units' %len(pre_2_taus))
plt.show()

#%% Start computing autocorrelation

# post Brain Area 1

post_1_all_means = []
post_1_taus = []
post_1_failed_fits = []

post_1_failed_autocorr = []
post_1_no_spikes_in_a_bin = []
post_1_low_fr = []

post_1_avg_fr = []
post_1_correlation_matrices = []

for unit in range(len(post_brain_area_1)):

    binned_spikes = post_brain_area_1[unit]

    [trials,bins] = binned_spikes.shape

    summed_spikes_per_bin = np.sum(binned_spikes,axis=0)

    post_1_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

    #%% Do autocorrelation

    one_autocorrelation = []

    for i in range(bins):
        for j in range(bins):
            ref_window = binned_spikes[:,i]
            test_window = binned_spikes[:,j]

            correlation = np.corrcoef(ref_window,test_window)[0,1]

            one_autocorrelation.append(correlation)

    if np.isnan(one_autocorrelation).any() == True:

        post_1_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

    elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
        # If there is not a spike in every bin
        post_1_no_spikes_in_a_bin.append(unit)

    elif np.sum(summed_spikes_per_bin)/trials < 1:

        post_1_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

    else:

        #%% Reshape list of autocorrelations into 19x19 matrix, plot it

        correlation_matrix = np.reshape(one_autocorrelation,(-1,20))

        post_1_correlation_matrices.append(correlation_matrix)

        # plt.imshow(correlation_matrix)
        # plt.title('Froot post 1 unit %i' %unit)
        # plt.xlabel('lag')
        # plt.ylabel('lag')
        # plt.xticks(range(0,19))
        # plt.yticks(range(0,19))
        # plt.show()

        #%% Fit exponential decay

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
        # plt.title('Froot post 1 unit %i' %unit)
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
            print("Error - curve_fit failed; unit %i" %unit)
            post_1_failed_fits.append(unit)

        post_1_taus.append(pars[1])

        post_1_all_means.append(y_m)

        # plt.plot(x_m,y_m,'ro')
        # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
        # plt.xlabel('lag (ms)')
        # plt.ylabel('mean autocorrelation')
        # plt.title('Froot 1 post unit %i' %unit)
        # plt.show()


#%% How many units got filtered?

bad_units = len(post_1_failed_autocorr) + len(post_1_no_spikes_in_a_bin) + len(post_1_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(post_brain_area_1))

#%% Take mean of all units

post_1_all_means = np.vstack(post_1_all_means)

post_1_mean = np.mean(post_1_all_means,axis=0)
post_1_sd = np.std(post_1_all_means,axis=0)
post_1_se = post_1_sd/np.sqrt(len(post_1_mean))

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(minxha_dacc_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

post_1_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],post_1_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,post_1_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*post_1_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Froot 1 post')
plt.text(710,0.1,'tau = %i' %post_1_pars[1])
plt.show()

#%% Histogram of taus

plt.hist(post_1_taus)
plt.xlabel('tau')
plt.ylabel('count')
plt.title('%i post 1 Froot units' %len(post_1_taus))
plt.show()

#%% Compute autocorrelation

# post Brain Area 2

post_2_all_means = []
post_2_taus = []
post_2_failed_fits = []

post_2_failed_autocorr = []
post_2_no_spikes_in_a_bin = []
post_2_low_fr = []

post_2_avg_fr = []
post_2_correlation_matrices = []

for unit in range(len(post_brain_area_2)):

    binned_spikes = post_brain_area_2[unit]

    [trials,bins] = binned_spikes.shape

    summed_spikes_per_bin = np.sum(binned_spikes,axis=0)

    post_2_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

    #%% Do autocorrelation

    one_autocorrelation = []

    for i in range(bins):
        for j in range(bins):
            ref_window = binned_spikes[:,i]
            test_window = binned_spikes[:,j]

            correlation = np.corrcoef(ref_window,test_window)[0,1]

            one_autocorrelation.append(correlation)

    if np.isnan(one_autocorrelation).any() == True:

        post_2_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

    elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
        # If there is not a spike in every bin
        post_2_no_spikes_in_a_bin.append(unit)

    elif np.sum(summed_spikes_per_bin)/trials < 1:

        post_2_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

    else:

        #%% Reshape list of autocorrelations into 19x19 matrix, plot it

        correlation_matrix = np.reshape(one_autocorrelation,(-1,20))

        post_2_correlation_matrices.append(correlation_matrix)

        # plt.imshow(correlation_matrix)
        # plt.title('Froot post 2 unit %i' %unit)
        # plt.xlabel('lag')
        # plt.ylabel('lag')
        # plt.xticks(range(0,19))
        # plt.yticks(range(0,19))
        # plt.show()

        #%% Fit exponential decay

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
        # plt.title('Froot post 2 unit %i' %unit)
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
            print("Error - curve_fit failed; unit %i" %unit)
            post_2_failed_fits.append(unit)

        post_2_taus.append(pars[1])

        post_2_all_means.append(y_m)

        # plt.plot(x_m,y_m,'ro')
        # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
        # plt.xlabel('lag (ms)')
        # plt.ylabel('mean autocorrelation')
        # plt.title('Froot 2 post unit %i' %unit)
        # plt.show()


#%% How many units got filtered?

bad_units = len(post_2_failed_autocorr) + len(post_2_no_spikes_in_a_bin) + len(post_2_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(post_brain_area_1))

#%% Take mean of all units

post_2_all_means = np.vstack(post_2_all_means)

post_2_mean = np.mean(post_2_all_means,axis=0)
post_2_sd = np.std(post_2_all_means,axis=0)
post_2_se = post_2_sd/np.sqrt(len(post_2_mean))

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(minxha_dacc_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

post_2_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],post_2_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,post_2_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*post_2_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Froot 2 post')
plt.text(710,0.1,'tau = %i' %post_2_pars[1])
plt.show()

#%% Histogram of taus

plt.hist(post_2_taus)
plt.xlabel('tau')
plt.ylabel('count')
plt.title('%i post 2 Froot units' %len(post_2_taus))
plt.show()

#%% Correlation matrices

pre_1_mean_matrix = np.mean(pre_1_correlation_matrices,axis=0)
pre_2_mean_matrix = np.mean(pre_2_correlation_matrices,axis=0)
post_1_mean_matrix = np.mean(post_1_correlation_matrices,axis=0)
post_2_mean_matrix = np.mean(post_2_correlation_matrices,axis=0)

fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')

plt.title('FROOT')

axs[0,0].imshow(pre_1_mean_matrix,cmap='inferno')
axs[0,0].set_title('pre 1')

axs[0,1].imshow(pre_2_mean_matrix,cmap='inferno')
axs[0,1].set_title('pre 2')

axs[1,0].imshow(post_1_mean_matrix,cmap='inferno')
axs[1,0].set_title('post 1')

axs[1,1].imshow(post_2_mean_matrix,cmap='inferno')
axs[1,1].set_title('post 2')

plt.tight_layout()
plt.show()
