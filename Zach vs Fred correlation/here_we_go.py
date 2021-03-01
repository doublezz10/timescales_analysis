#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:35:25 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import scipy.io as spio
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import dfply

#%% Load data

fred = pd.read_csv('/Users/zachz/Documents/timescales_analysis/GLM/fred_grouped.csv')

zach = pd.read_csv('/Users/zachz/Documents/timescales_analysis/GLM/zach_grouped.csv')

#%% Find matching units, build dataframe of them

matching_units = []

for unit in range(len(zach)):
    
    this_dataset = zach.iloc[unit]['dataset']
    
    if this_dataset == 'steinmetz':
        
        this_dataset = 'stein'

    this_brain_area = zach.iloc[unit]['brain_region']
        
    this_unit = zach.iloc[unit]['unit_id']

    matching_unit = fred[fred['dataset'].str.contains(this_dataset) & fred['brain_region'].str.contains(this_brain_area) & fred['unit_id'] == this_unit]
    
    if len(matching_unit) == 0:
        
        pass
    
    else:
        
        matching_units.append((this_dataset,this_brain_area,this_unit,zach.iloc[unit]['tau'],zach.iloc[unit]['r2'],matching_unit.iloc[0]['tau'],matching_unit.iloc[0]['r2']))

matching_units = pd.DataFrame(matching_units,columns=['dataset','brain_region','unit_id','zach_tau','zach_r2','fred_tau','fred_r2'])

#%%

faraut = matching_units[matching_units['dataset'].str.contains('faraut')]
minxha = matching_units[matching_units['dataset'].str.contains('minxha')]
meg = matching_units[matching_units['dataset'].str.contains('meg')]
stein = matching_units[matching_units['dataset'].str.contains('stein')]

for unit in stein:
    
    