#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:40:49 2021

@author: zachz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

#%% Load in traditional data

old_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/GLM - first attempts/zach_grouped.csv',index_col=0)

meg = old_data[old_data.dataset == 'meg']

meg_matches = []

for unit in range(len(meg)):
    
    brain_area = meg.iloc[unit].brain_area
    
    if brain_area == 'amyg':
        
        brain_area = 'amygdala'
        
    elif brain_area == 'scacc':
        
        brain_area = 'scACC'
        
    elif brain_area == 'vs':
        
        brain_area = 'vStriatum'
    
    unit_id = meg.iloc[unit].unit_id
    
    matches = data[(data.dataset == 'meg') & (data.brain_area == brain_area) & (data.unit == unit_id)]
    
    mean_tau = np.mean(matches['tau'])
    
    trad_tau = meg.iloc[unit]['tau']
    
    meg_matches.append((unit_id,brain_area,mean_tau,trad_tau))
    
meg_match = pd.DataFrame(meg_matches,columns=['unit','brain_area','iter_tau','trad_tau'])

#%%

sns.lmplot(data=meg_match,x='iter_tau',y='trad_tau',hue='brain_area',legend=True,ci=False,scatter_kws={'s': 15, 'alpha': 0.5})

plt.plot(range(1000),range(1000),linestyle='--',label='identity',color='black')

plt.xlabel('trial-agnostic tau')
plt.ylabel('fixation only tau')

plt.title('Megs data - monkey')

plt.show()

#%%

stein = old_data[old_data.dataset == 'steinmetz']

stein_matches = []

for unit in range(len(stein)):
    
    brain_area = stein.iloc[unit].brain_area
    
    unit_id = stein.iloc[unit].unit_id
    
    matches = data[(data.dataset == 'steinmetz') & (data.brain_area == brain_area) & (data.unit == unit_id)]
    
    mean_tau = np.mean(matches['tau'])
    
    trad_tau = stein.iloc[unit]['tau']
    
    stein_matches.append((unit_id,brain_area,mean_tau,trad_tau))
    
stein_match = pd.DataFrame(stein_matches,columns=['unit','brain_area','iter_tau','trad_tau'])

#%%

sns.lmplot(data=stein_match,x='iter_tau',y='trad_tau',hue='brain_area',legend=True,ci=False,scatter_kws={'s': 15, 'alpha': 0.5})

plt.plot(range(1000),range(1000),linestyle='--',label='identity',color='black')

plt.xlabel('trial-agnostic tau')
plt.ylabel('fixation only tau')

plt.title('Steinmetz data - mouse')

plt.show()