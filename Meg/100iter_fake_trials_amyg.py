#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:43:01 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import ptitprince as pt

#%%

filtered_real = real_data[real_data.r2 >= 0.5]
filtered_real = real_data[(real_data.tau >= 10) & (real_data.tau <= 1000)]

filtered_fake = fake_data[fake_data.r2 >= 0.5]
filtered_fake = fake_data[(fake_data.tau >= 10) & (fake_data.tau <= 1000)]

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
    
plt.title('Individual unit variation (mean +/- se) \n Meg amygdala, 100 iterations')
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
    
plt.title('Individual unit variation (mean +/- sd) \n Meg amygdala, 100 iterations')
plt.xlabel('unit')
plt.ylabel('tau (ms)')
plt.show()

#%% Correlation

for unit in range(len(filtered_real)):
    
    this_unit = filtered_real.iloc[unit]
    
    this_id = this_unit['unit_id']

    fake_units = filtered_fake[filtered_fake['unit'] == this_id]
    
    taus = fake_units['tau']
    
    mean_tau = np.mean(taus)
    
    plt.scatter(this_unit['tau'],mean_tau)
    
plt.plot(range(0,1000),range(0,1000),'-',color='red',label='identity')
plt.title('Real vs mean fake \n 100 iterations')
plt.xlabel('"real" tau (ms)')
plt.ylabel('"fake" tau (ms)')
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
plt.show()

#%% Individual violin plots

for unit in range(len(filtered_real)):
    
    this_unit = filtered_real.iloc[unit]
    
    this_id = this_unit['unit_id']

    fake_units = filtered_fake[filtered_fake['unit'] == this_id]
    
    taus = fake_units['tau']
    
    try:
    
        plt.violinplot(taus,showmeans=True)
        
        plt.title('%i' %this_id)
        
        plt.show()
        
    except ValueError:
        
        pass
    
