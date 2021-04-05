#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:44:44 2021

@author: zachz
"""

real_data = []
fake_data = []

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels as sm

plt.style.use('ggplot')

#%%

filtered_real = real_data[real_data.r2 >= 0.5]
filtered_real = real_data[(real_data.tau >= 10) & (real_data.tau <= 1000)]

filtered_fake = fake_data[(fake_data.tau >= 10) & (fake_data.tau <= 1000)]

#%% count units with >50% of fake simulations above r2 = 0.5

percent_good_fits = []

unique_units = fake_data['unit'].unique()

for unit in range(len(unique_units)):
    
    this_unit = fake_data[fake_data['unit'] == unit]
    
    n_good_fits = len(this_unit[this_unit['r2'] >= 0.5])
    
    percent_good_fits.append(n_good_fits/1000)
    
plt.hist(percent_good_fits,color='#00b2ee')
plt.xlabel('Percent of fits with R$^2$ > 0.5')
plt.ylabel('Count')
plt.show()

#%% "Box plot"

for unit in range(len(filtered_real)):
    
    this_unit = filtered_real.iloc[unit]
    
    this_id = this_unit['unit_id']
    
    fake_units = filtered_fake[filtered_fake['unit'] == this_id]
    
    taus = fake_units['tau']
    
    mean_tau = np.mean(taus)
    
    sd_tau = np.std(taus)
    se_tau = sd_tau/np.sqrt(len(taus))
    
    plt.scatter(unit,mean_tau)
    
    plt.errorbar(unit,mean_tau,yerr=se_tau)
    
plt.title('Individual unit variation (mean +/- se) \n Meg amygdala, 1000 iterations')
plt.xlabel('unit')
plt.ylabel('tau (ms)')
plt.show()

#%% "Box plot"

for unit in range(len(filtered_real)):
    
    this_unit = filtered_real.iloc[unit]
    
    this_id = this_unit['unit_id']
    
    fake_units = filtered_fake[filtered_fake['unit'] == this_id]
    
    taus = fake_units['tau']
    
    mean_tau = np.mean(taus)
    
    sd_tau = np.std(taus)
    se_tau = sd_tau/np.sqrt(len(taus))
    
    plt.scatter(unit,mean_tau)
    
    plt.errorbar(unit,mean_tau,yerr=sd_tau)
    
plt.title('Individual unit variation (mean +/- sd) \n Meg amygdala, 1000 iterations')
plt.xlabel('unit')
plt.ylabel('tau (ms)')
plt.show()

#%% Correlation

mean_fake = []
mean_real = []

for unit in range(len(filtered_real)):
    
    this_unit = filtered_real.iloc[unit]
    
    this_id = this_unit['unit_id']
    
    mean_real.append(this_unit['tau'])

    fake_units = filtered_fake[filtered_fake['unit'] == this_id]
    
    taus = fake_units['tau']
    
    mean_tau = np.mean(taus)
    
    mean_fake.append(mean_tau)
    
    plt.scatter(this_unit['tau'],mean_tau)
    
mean_fake = np.array(mean_fake).reshape(-1,1)
mean_real = np.array(mean_real).reshape(-1,1)
    
regr = linear_model.LinearRegression()
regr.fit(mean_real, mean_fake)
fake_tau_pred = regr.predict(mean_real)    

r2 = r2_score(mean_fake, fake_tau_pred)

import statsmodels.api as sm
mod = sm.OLS(mean_fake,mean_real)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']

plt.plot(mean_real,fake_tau_pred,color='#00b2ee',label='regression')

plt.plot(range(0,1000),range(0,1000),'-',color='#f00c93',label='identity')
plt.title('Monkey amygdala \n 1000 iterations')
plt.xlabel('"real" tau (ms)')
plt.ylabel('"fake" tau (ms)')
plt.text(0,800,'$R^2 = %.2f $ \n $p < 10^{-30}$' %(r2))
plt.legend()
plt.show()

#%% num good trials

for unit in range(len(filtered_real)):
    
    this_unit = filtered_real.iloc[unit]
    
    this_id = this_unit['unit_id']

    fake_units = filtered_fake[filtered_fake['unit'] == this_id]
    
    taus = fake_units['tau']
    
    mean_tau = np.mean(taus)
    
    tau_diff = mean_tau - this_unit['tau']
    
    n_trials = len(taus)
    
    plt.scatter(n_trials,tau_diff)
    
plt.ylabel('mean(fake taus) - real tau')
plt.xlabel('n fake trials')
plt.title('Amygdala \n 1000 iterations')
plt.show()

#%% r2

for unit in range(len(filtered_real)):
    
    this_unit = filtered_real.iloc[unit]
    
    this_id = this_unit['unit_id']

    fake_units = filtered_fake[filtered_fake['unit'] == this_id]
    
    taus = fake_units['tau']
    
    mean_tau = np.mean(taus)
    
    tau_diff = mean_tau - this_unit['tau']
    
    r2s = fake_units['r2']
    
    mean_r2 = np.mean(r2s)
    
    plt.scatter(mean_r2,tau_diff)
    
plt.ylabel('mean(fake taus) - real tau')
plt.xlabel('mean r$^2$')
plt.title('Amygdala \n 1000 iterations')
plt.show()

#%% Individual violin plots

plt.figure(figsize=(4,4))

for unit in range(len(filtered_real)):
    
    this_unit = filtered_real.iloc[unit]
    
    this_id = this_unit['unit_id']

    fake_units = filtered_fake[filtered_fake['unit'] == this_id]
    
    taus = fake_units['tau']
    
    bins = 10**np.arange(0,4,0.1)
    histogram, bins = np.histogram(taus, bins=bins)
    
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    
    # Compute the PDF on the bin centers from scipy distribution object
    from scipy import stats
    pdf = stats.norm.pdf(bin_centers)
    
    plt.figure(figsize=(4,4))
    
    plt.plot(bin_centers, histogram, color='#400080')
    
    plt.xscale('log')
    
    plt.title('unit #%i' %(unit+1))
    
    plt.xlabel('tau (ms)')
    
    plt.ylabel('count')
    
    plt.show()

#%% Histogram

bins = 10**np.arange(0,4,0.1)

plt.hist(mean_fake,bins=bins, weights=np.zeros_like(mean_fake) + 1. / len(mean_fake),color='#00b2ee',label='fake',alpha=0.6,zorder=2)
plt.hist(mean_real,bins=bins, weights=np.zeros_like(mean_real) + 1. / len(mean_real),color='#f00c93',label='real',alpha=0.6,zorder=1)

plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.legend()
plt.grid(False)
plt.show()