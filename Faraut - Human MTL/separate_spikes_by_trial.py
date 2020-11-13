#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:28:03 2020

@author: zachz
"""

#%% Imports

import numpy as np
import scipy.io as spio

#%% Load in data

hc = spio.loadmat('/Users/zachz/Downloads/faraut_hippocampus.mat',simplify_cells=True)
amyg = spio.loadmat('/Users/zachz/Downloads/faraut_amygdala.mat',simplify_cells=True)

hc_cell_info = hc['cell_info']
hc_spikes = hc['spikes']

amyg_cell_info = amyg['cell_info']
amyg_spikes = amyg['spikes']

hc_trial_start = []

for unit in range(len(hc_cell_info)):
    
    this_unit = hc_cell_info[unit]
    
    trial_start_times = this_unit['TrialBreaks']
    
    hc_trial_start.append(trial_start_times)
    
amyg_trial_start = []

for unit in range(len(amyg_cell_info)):
    
    this_unit = amyg_cell_info[unit]
    
    trial_start_times = this_unit['TrialBreaks']
    
    amyg_trial_start.append(trial_start_times)
    
#%% For each unit, split spikes into trials

hc_trial_spikes = []

for hc_unit in range(len(hc_spikes)):
    
    unit_spikes = []
    
    spikes = hc_spikes[hc_unit]
    
    trials = hc_trial_start[hc_unit]
    
    for trial in range(len(trials)-1):
        
        trial_spikes = []
        
        for spike in range(len(spikes)):
            
            if trials[trial] <= spikes[spike] <= trials[trial+1]:
                
                # Align spike times to beginning of trial, convert from us to sec
                
                trial_spikes.append((spikes[spike] - trials[trial])/10**6)
                
        unit_spikes.append(trial_spikes)
        
    hc_trial_spikes.append(unit_spikes)
    
amyg_trial_spikes = []

for amyg_unit in range(len(amyg_spikes)):
    
    unit_spikes = []
    
    spikes = amyg_spikes[amyg_unit]
    
    trials = amyg_trial_start[amyg_unit]
    
    for trial in range(len(trials)-1):
        
        trial_spikes = []
        
        for spike in range(len(spikes)):
            
            if trials[trial] <= spikes[spike] <= trials[trial+1]:
                
                trial_spikes.append((spikes[spike] - trials[trial])/10**6)
                
        unit_spikes.append(trial_spikes)
        
    amyg_trial_spikes.append(unit_spikes)
    
#%% Re save as matlab

spikes = amyg_trial_spikes
cell_info = amyg_cell_info

spio.savemat('faraut_amygdala.mat',dict(spikes = spikes,cell_info = cell_info))

spikes = hc_trial_spikes
cell_info = hc_cell_info

spio.savemat('faraut_hc.mat',dict(spikes=spikes,cell_info=cell_info))