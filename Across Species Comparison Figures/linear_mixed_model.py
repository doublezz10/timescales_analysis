#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:29:35 2021

@author: zachz
"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.io as spio
import pandas as pd

#%% Load in data

# use .spydata file

#%% make into one big dataframe

big_model = smf.mixedlm("tau ~ fr + brain_area + species",df,groups=df['dataset'])

fit = big_model.fit()

print(big_model.summary())