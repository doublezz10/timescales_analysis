#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:15:30 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#%%

real_trials = pd.DataFrame(all_data_meg_amyg,columns=['dataset','species','brain_area','unit_id','tau','fr','r2','a','b'])
filtered_real = real_trials[real_trials['r2'] >= 0.5]
filtered_real = real_trials[real_trials['tau'] <= 1000]

fake_trials = pd.DataFrame(all_data_meg_amyg_faketrials,columns=['dataset','species','brain_area','unit_id','tau','fr','r2','a','b'])
filtered_fake = fake_trials[fake_trials['r2'] >= 0.5]
filtered_fake = fake_trials[fake_trials['tau'] <= 1000]

#%% Unfiltered

for unit in range(len(fake_trials)):
    
    match = real_trials.loc[real_trials['unit_id'] == fake_trials.iloc[unit]['unit_id']]
    
    real_tau = match['tau']
    fake_tau = fake_trials.iloc[unit]['tau']
    
    if len(match) == 0:
        
        pass
    
    else:
        plt.scatter(real_tau,fake_tau)
    
        
plt.plot(range(0,1000),range(0,1000))
plt.xlabel('real trials')
plt.ylabel('fake trials')
plt.title('Amyg Unfiltered')
plt.show()
        
#%% Filtered

real_taus = []
fake_taus = []

for unit in range(len(filtered_fake)):
    
    match = filtered_real.loc[filtered_real['unit_id'] == filtered_fake.iloc[unit]['unit_id']]
    
    real_tau = match['tau']
    fake_tau = filtered_fake.iloc[unit]['tau']
    
    if len(match) == 0:
        
        pass
    
    else:
        real_taus.append(real_tau)
        fake_taus.append(fake_tau)
        plt.scatter(real_tau,fake_tau)
        
regr = linear_model.LinearRegression()
regr.fit(real_taus, fake_taus)
fake_tau_pred = regr.predict(real_taus)

r2 = r2_score(fake_taus, fake_tau_pred)
        
plt.plot(range(0,1000),range(0,1000),'-',color='red',label='identity')
plt.plot(real_taus,fake_tau_pred,color='blue',label='regression')
plt.xlabel('real trials')
plt.ylabel('fake trials')
plt.title('Amyg filtered for R$^2$ > 0.5')
plt.text(0,860,'R$^2$ = %.2f' %r2)
plt.legend()
plt.show()