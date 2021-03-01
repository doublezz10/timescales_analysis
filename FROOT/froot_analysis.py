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
import warnings
import csv
warnings.filterwarnings("ignore")

#%% Labels

"""
1 = OFC
2 = ACC
"""

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

all_data_ofc_pre = []

ofc_pre_all_means = []
ofc_pre_taus = []
ofc_pre_failed_fits = []

ofc_pre_failed_autocorr = []
ofc_pre_no_spikes_in_a_bin = []
ofc_pre_low_fr = []

ofc_pre_avg_fr = []
ofc_pre_correlation_matrices = []

for unit in range(len(pre_brain_area_1)):

    binned_spikes = pre_brain_area_1[unit]

    [trials,bins] = binned_spikes.shape

    summed_spikes_per_bin = np.sum(binned_spikes,axis=0)

    #%% Do autocorrelation

    one_autocorrelation = []

    for i in range(bins):
        for j in range(bins):
            ref_window = binned_spikes[:,i]
            test_window = binned_spikes[:,j]

            correlation = np.corrcoef(ref_window,test_window)[0,1]

            one_autocorrelation.append(correlation)

    if np.isnan(one_autocorrelation).any() == True:

        ofc_pre_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

    elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
        # If there is not a spike in every bin
        ofc_pre_no_spikes_in_a_bin.append(unit)

    elif np.sum(summed_spikes_per_bin)/trials < 1:

        ofc_pre_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

    else:

        #%% Reshape list of autocorrelations into 19x19 matrix, plot it

        correlation_matrix = np.reshape(one_autocorrelation,(-1,20))

        ofc_pre_correlation_matrices.append(correlation_matrix)

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
        # plt.title('Froot OFC pre unit %i' %unit)
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
            ofc_pre_failed_fits.append(unit)

        r2 = (np.corrcoef(y_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars)))[0,1]**2

        ofc_pre_taus.append(pars[1])
        ofc_pre_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

        ofc_pre_all_means.append(y_m)

        # plt.plot(x_m,y_m,'ro')
        # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
        # plt.xlabel('lag (ms)')
        # plt.ylabel('mean autocorrelation')
        # plt.title('Froot OFC pre unit %i' %unit)
        # plt.show()

        #%% Add data to 'all_data'

        all_data_ofc_pre.append(('froot','monkey','ofc',unit,pars[1],np.sum(summed_spikes_per_bin)/trials,r2,pars[0],pars[2]))

with open('/Users/zachz/Documents/timescales_analysis/results.csv','a') as out:
    csv_out=csv.writer(out)
    for row in all_data_ofc_pre:
        csv_out.writerow(row)

#%% How many units got filtered?

bad_units = len(ofc_pre_failed_autocorr) + len(ofc_pre_no_spikes_in_a_bin) + len(ofc_pre_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(pre_brain_area_1))

#%% Take mean of all units

ofc_pre_all_means = np.vstack(ofc_pre_all_means)

ofc_pre_mean = np.mean(ofc_pre_all_means,axis=0)
ofc_pre_sd = np.std(ofc_pre_all_means,axis=0)
ofc_pre_se = ofc_pre_sd/np.sqrt(len(ofc_pre_mean))

ofc_pre_mean_fr = np.mean(ofc_pre_avg_fr)

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(ofc_pre_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

ofc_pre_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],ofc_pre_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,ofc_pre_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*ofc_pre_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Froot OFC pre')
plt.text(710,0.1,'tau = %i \n fr = %.2f \n n = %i' % (ofc_pre_pars[1], ofc_pre_mean_fr, len(ofc_pre_taus)))
plt.show()

a_population_ofc_pre = (('froot','ofc',ofc_pre_pars[1],ofc_pre_mean_fr,len(ofc_pre_taus)))

#%% Add error bars

plt.errorbar(x_m, ofc_pre_mean, yerr=ofc_pre_se, label='data +/- se')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*ofc_pre_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('autocorrelation')
plt.title('Mean of all monkey OFC pre units \n FROOT')
plt.text(710,0.1,'tau = %i ms \n fr = %.2f hz \n n = %i' % (ofc_pre_pars[1],ofc_pre_mean_fr,len(ofc_pre_taus)))
plt.ylim((0,0.2))
plt.show()

#%% Histogram of taus

bins = 10**np.arange(0,4,0.1)

plt.hist(ofc_pre_taus,bins=bins, weights=np.zeros_like(ofc_pre_taus) + 1. / len(ofc_pre_taus))
plt.axvline(ofc_pre_pars[1],color='r',linestyle='dashed',linewidth=1)
plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('%i Froot OFC pre units' %len(ofc_pre_taus))
plt.show()

#%% Compute autocorrelation

# Pre Brain Area 1

all_data_acc_pre = []

acc_pre_all_means = []
acc_pre_taus = []
acc_pre_failed_fits = []

acc_pre_failed_autocorr = []
acc_pre_no_spikes_in_a_bin = []
acc_pre_low_fr = []

acc_pre_avg_fr = []
acc_pre_correlation_matrices = []

for unit in range(len(pre_brain_area_2)):

    binned_spikes = pre_brain_area_2[unit]

    [trials,bins] = binned_spikes.shape

    summed_spikes_per_bin = np.sum(binned_spikes,axis=0)

    #%% Do autocorrelation

    one_autocorrelation = []

    for i in range(bins):
        for j in range(bins):
            ref_window = binned_spikes[:,i]
            test_window = binned_spikes[:,j]

            correlation = np.corrcoef(ref_window,test_window)[0,1]

            one_autocorrelation.append(correlation)

    if np.isnan(one_autocorrelation).any() == True:

        acc_pre_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

    elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
        # If there is not a spike in every bin
        acc_pre_no_spikes_in_a_bin.append(unit)

    elif np.sum(summed_spikes_per_bin)/trials < 1:

        acc_pre_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

    else:

        #%% Reshape list of autocorrelations into 19x19 matrix, plot it

        correlation_matrix = np.reshape(one_autocorrelation,(-1,20))

        acc_pre_correlation_matrices.append(correlation_matrix)

        # plt.imshow(correlation_matrix)
        # plt.title('Froot ACC pre unit %i' %unit)
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
        # plt.title('Froot ACC pre unit %i' %unit)
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
            acc_pre_failed_fits.append(unit)

        r2 = (np.corrcoef(y_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars)))[0,1]**2

        acc_pre_taus.append(pars[1])
        acc_pre_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

        acc_pre_all_means.append(y_m)

        # plt.plot(x_m,y_m,'ro')
        # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
        # plt.xlabel('lag (ms)')
        # plt.ylabel('mean autocorrelation')
        # plt.title('Froot ACC pre unit %i' %unit)
        # plt.show()

        #%% Add data to 'all_data'

        all_data_acc_pre.append(('froot','monkey','acc',unit,pars[1],np.sum(summed_spikes_per_bin)/trials,r2,pars[0],pars[2]))

with open('/Users/zachz/Documents/timescales_analysis/results.csv','a') as out:
    csv_out=csv.writer(out)
    for row in all_data_acc_pre:
        csv_out.writerow(row)

#%% How many units got filtered?

bad_units = len(acc_pre_failed_autocorr) + len(acc_pre_no_spikes_in_a_bin) + len(acc_pre_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(pre_brain_area_1))

#%% Take mean of all units

acc_pre_all_means = np.vstack(acc_pre_all_means)

acc_pre_mean = np.mean(acc_pre_all_means,axis=0)
acc_pre_sd = np.std(acc_pre_all_means,axis=0)
acc_pre_se = acc_pre_sd/np.sqrt(len(acc_pre_mean))

acc_pre_mean_fr = np.mean(acc_pre_avg_fr)

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(acc_pre_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

acc_pre_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],acc_pre_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,acc_pre_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*acc_pre_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Froot ACC pre')
plt.text(710,0.1,'tau = %i \n fr = %.2f \n n = %i' % (acc_pre_pars[1], acc_pre_mean_fr, len(acc_pre_taus)))
plt.show()

a_population_acc_pre = (('froot','acc',acc_pre_pars[1],acc_pre_mean_fr,len(acc_pre_taus)))

#%% Add error bars

plt.errorbar(x_m, acc_pre_mean, yerr=acc_pre_se, label='data +/- se')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*acc_pre_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('autocorrelation')
plt.title('Mean of all monkey ACC pre units \n FROOT')
plt.text(710,0.1,'tau = %i ms \n fr = %.2f hz \n n = %i' % (acc_pre_pars[1],acc_pre_mean_fr,len(acc_pre_taus)))
plt.ylim((0,0.2))
plt.show()

#%% Histogram of taus

bins = 10**np.arange(0,4,0.1)

plt.hist(acc_pre_taus,bins=bins, weights=np.zeros_like(acc_pre_taus) + 1. / len(acc_pre_taus))
plt.axvline(acc_pre_pars[1],color='r',linestyle='dashed',linewidth=1)
plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('%i Froot ACC pre' %len(acc_pre_taus))
plt.show()

#%% Start computing autocorrelation

# post Brain Area 1

ofc_post_all_means = []
ofc_post_taus = []
ofc_post_failed_fits = []

ofc_post_failed_autocorr = []
ofc_post_no_spikes_in_a_bin = []
ofc_post_low_fr = []

ofc_post_avg_fr = []
ofc_post_correlation_matrices = []

for unit in range(len(post_brain_area_1)):

    binned_spikes = post_brain_area_1[unit]

    [trials,bins] = binned_spikes.shape

    summed_spikes_per_bin = np.sum(binned_spikes,axis=0)

    #%% Do autocorrelation

    one_autocorrelation = []

    for i in range(bins):
        for j in range(bins):
            ref_window = binned_spikes[:,i]
            test_window = binned_spikes[:,j]

            correlation = np.corrcoef(ref_window,test_window)[0,1]

            one_autocorrelation.append(correlation)

    if np.isnan(one_autocorrelation).any() == True:

        ofc_post_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

    elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
        # If there is not a spike in every bin
        ofc_post_no_spikes_in_a_bin.append(unit)

    elif np.sum(summed_spikes_per_bin)/trials < 1:

        ofc_post_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

    else:

        #%% Reshape list of autocorrelations into 19x19 matrix, plot it

        correlation_matrix = np.reshape(one_autocorrelation,(-1,20))

        ofc_post_correlation_matrices.append(correlation_matrix)

        # plt.imshow(correlation_matrix)
        # plt.title('Froot OFC post unit %i' %unit)
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
        # plt.title('Froot OFC post unit %i' %unit)
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
            ofc_post_failed_fits.append(unit)

        ofc_post_taus.append(pars[1])
        ofc_post_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

        ofc_post_all_means.append(y_m)

        # plt.plot(x_m,y_m,'ro')
        # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
        # plt.xlabel('lag (ms)')
        # plt.ylabel('mean autocorrelation')
        # plt.title('Froot OFC post unit %i' %unit)
        # plt.show()


#%% How many units got filtered?

bad_units = len(ofc_post_failed_autocorr) + len(ofc_post_no_spikes_in_a_bin) + len(ofc_post_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(post_brain_area_1))

#%% Take mean of all units

ofc_post_all_means = np.vstack(ofc_post_all_means)

ofc_post_mean = np.mean(ofc_post_all_means,axis=0)
ofc_post_sd = np.std(ofc_post_all_means,axis=0)
ofc_post_se = ofc_post_sd/np.sqrt(len(ofc_post_mean))

ofc_post_mean_fr = np.mean(ofc_post_avg_fr)

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(ofc_post_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

ofc_post_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],ofc_post_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,ofc_post_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*ofc_post_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Froot OFC post')
plt.text(710,0.1,'tau = %i \n fr = %.2f \n n = %i' % (ofc_post_pars[1], ofc_post_mean_fr, len(ofc_post_taus)))
plt.show()

#%% Add error bars

plt.errorbar(x_m, ofc_post_mean, yerr=ofc_post_se, label='data +/- se')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*ofc_post_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('autocorrelation')
plt.title('Mean of all monkey OFC post units \n FROOT')
plt.text(710,0.1,'tau = %i ms \n fr = %.2f hz \n n = %i' % (ofc_post_pars[1],ofc_post_mean_fr,len(ofc_post_taus)))
plt.ylim((0,0.2))
plt.show()

#%% Histogram of taus

bins = 10**np.arange(0,4,0.1)

plt.hist(ofc_post_taus,bins=bins, weights=np.zeros_like(ofc_post_taus) + 1 / len(ofc_post_taus))
plt.axvline(ofc_post_pars[1],color='r',linestyle='dashed',linewidth=1)
plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('%i Froot OFC post units' %len(ofc_post_taus))
plt.show()

#%% Compute autocorrelation

# post Brain Area 2

acc_post_all_means = []
acc_post_taus = []
acc_post_failed_fits = []

acc_post_failed_autocorr = []
acc_post_no_spikes_in_a_bin = []
acc_post_low_fr = []

acc_post_avg_fr = []
acc_post_correlation_matrices = []

for unit in range(len(post_brain_area_2)):

    binned_spikes = post_brain_area_2[unit]

    [trials,bins] = binned_spikes.shape

    summed_spikes_per_bin = np.sum(binned_spikes,axis=0)

    #%% Do autocorrelation

    one_autocorrelation = []

    for i in range(bins):
        for j in range(bins):
            ref_window = binned_spikes[:,i]
            test_window = binned_spikes[:,j]

            correlation = np.corrcoef(ref_window,test_window)[0,1]

            one_autocorrelation.append(correlation)

    if np.isnan(one_autocorrelation).any() == True:

        acc_post_failed_autocorr.append(unit) # skip this unit if any autocorrelation fails

    elif np.any(summed_spikes_per_bin[bin] == 0 for bin in range(len(summed_spikes_per_bin))) == True:
        # If there is not a spike in every bin
        acc_post_no_spikes_in_a_bin.append(unit)

    elif np.sum(summed_spikes_per_bin)/trials < 1:

        acc_post_low_fr.append(unit) # skip this unit if avg firing rate across all trials is < 1

    else:

        #%% Reshape list of autocorrelations into 19x19 matrix, plot it

        correlation_matrix = np.reshape(one_autocorrelation,(-1,20))

        acc_post_correlation_matrices.append(correlation_matrix)

        # plt.imshow(correlation_matrix)
        # plt.title('Froot ACC post unit %i' %unit)
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
        # plt.title('Froot ACC post unit %i' %unit)
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
            acc_post_failed_fits.append(unit)

        acc_post_taus.append(pars[1])
        acc_post_avg_fr.append(np.sum(summed_spikes_per_bin)/trials)

        acc_post_all_means.append(y_m)

        # plt.plot(x_m,y_m,'ro')
        # plt.plot(x_m[first_neg_diff:],func(x_m[first_neg_diff:],*pars),label='fit')
        # plt.xlabel('lag (ms)')
        # plt.ylabel('mean autocorrelation')
        # plt.title('Froot ACC post unit %i' %unit)
        # plt.show()


#%% How many units got filtered?

bad_units = len(acc_post_failed_autocorr) + len(acc_post_no_spikes_in_a_bin) + len(acc_post_low_fr)

print('%i units were filtered out' %bad_units)
print('out of %i total units' %len(post_brain_area_1))

#%% Take mean of all units

acc_post_all_means = np.vstack(acc_post_all_means)

acc_post_mean = np.mean(acc_post_all_means,axis=0)
acc_post_sd = np.std(acc_post_all_means,axis=0)
acc_post_se = acc_post_sd/np.sqrt(len(acc_post_mean))

acc_post_mean_fr = np.mean(acc_post_avg_fr)

def func(x,a,tau,b):
    return a*((np.exp(-x/tau))+b)

mean_diff = np.diff(acc_post_mean)

neg_mean_diffs = []

for diff in range(len(mean_diff)):

    if mean_diff[diff] <= 0:

        neg_mean_diffs.append(diff)

first_neg_mean_diff = np.min(neg_mean_diffs)

acc_post_pars,cov = curve_fit(func,x_m[first_neg_mean_diff:],acc_post_mean[first_neg_mean_diff:],p0=[1,100,1],bounds=((0,np.inf)))

plt.plot(x_m,acc_post_mean,label='original data')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*acc_post_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('mean autocorrelation')
plt.title('Froot ACC post')
plt.text(710,0.1,'tau = %i \n fr = %.2f \n n = %i' % (acc_post_pars[1], acc_post_mean_fr, len(acc_post_taus)))
plt.show()

#%% Add error bars

plt.errorbar(x_m, acc_post_mean, yerr=acc_post_se, label='data +/- se')
plt.plot(x_m[first_neg_mean_diff:],func(x_m[first_neg_mean_diff:],*acc_post_pars),label='fit curve')
plt.legend(loc='upper right')
plt.xlabel('lag (ms)')
plt.ylabel('autocorrelation')
plt.title('Mean of all monkey ACC post units \n FROOT')
plt.text(710,0.1,'tau = %i ms \n fr = %.2f hz \n n = %i' % (acc_post_pars[1],acc_post_mean_fr,len(acc_post_taus)))
plt.ylim((0,0.2))
plt.show()

#%% Histogram of taus

bins = 10**np.arange(0,4,0.1)

plt.hist(acc_post_taus,bins=bins, weights=np.zeros_like(acc_post_taus) + 1. / len(acc_post_taus))
plt.axvline(acc_post_pars[1],color='r',linestyle='dashed',linewidth=1)
plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('%i Froot ACC post units' %len(acc_post_taus))
plt.show()

#%% Correlation matrices

# ofc_pre_mean_matrix = np.mean(ofc_pre_correlation_matrices,axis=0)
# acc_pre_mean_matrix = np.mean(acc_pre_correlation_matrices,axis=0)
# ofc_post_mean_matrix = np.mean(ofc_post_correlation_matrices,axis=0)
# acc_post_mean_matrix = np.mean(acc_post_correlation_matrices,axis=0)

# fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')

# plt.title('FROOT')

# axs[0,0].imshow(ofc_pre_mean_matrix,cmap='inferno')
# axs[0,0].set_title('pre 1')

# axs[0,1].imshow(acc_pre_mean_matrix,cmap='inferno')
# axs[0,1].set_title('pre 2')

# axs[1,0].imshow(ofc_post_mean_matrix,cmap='inferno')
# axs[1,0].set_title('post 1')

# axs[1,1].imshow(acc_post_mean_matrix,cmap='inferno')
# axs[1,1].set_title('post 2')

# plt.tight_layout()
# plt.show()
