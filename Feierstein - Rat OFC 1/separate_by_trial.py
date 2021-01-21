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

#%% Repeat "make fake trials" procedure from Buzsaki
