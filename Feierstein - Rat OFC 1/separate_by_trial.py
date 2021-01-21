#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:06:17 2021

@author: zachz
"""

# What unit are we working in?
# This data is so hard to work with for some reason
# And there's very little info to describe what's happening

#%% Imports

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

#%% Load data

ofc = spio.loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Feierstein - rat OFC 1/ofc_1.mat',simplify_cells=True)

spikes = ofc['spikes']
trial_start_times = ofc['trialstart'] * 10
for unit in range(len(trial_start_times)):
    trial_start_times[unit] = trial_start_times[unit][~np.isnan(trial_start_times[unit])]

#%% Put first second of each trial on a new line

unit_by_trial = []

for unit in range(len(spikes)):
    
    unit_spikes = spikes[unit]
    trial_start = trial_start_times[unit]
    trial_end = trial_start + 1
    
    all_trial_spikes = []
    
    for trial in range(len(trial_start)):
        
        start_t = trial_start[trial]
        end_t = trial_end[trial]
        
        trial_spikes = []
        
        idx = (unit_spikes>start_t) * (unit_spikes<=end_t)
        
        trial_spikes = np.where(idx)
                
        all_trial_spikes.append(trial_spikes)
        
    unit_by_trial.append(all_trial_spikes)
    
#%%

plt.eventplot(spikes[0],lineoffsets=-1)
plt.eventplot(trial_start_times[0])
plt.show()

