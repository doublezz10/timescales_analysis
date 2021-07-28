#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:21:21 2021

@author: zachz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import ptitprince as pt

import seaborn as sns

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

raw_acc = raw_data[(raw_data.brain_area == 'acc') | (raw_data.brain_area == 'dACC') | (raw_data.brain_area == 'aca') | (raw_data.brain_area == 'mcc')]

raw_amyg = raw_data[(raw_data.brain_area == 'amygdala') | (raw_data.brain_area == 'central') | (raw_data.brain_area == 'bla')]

raw_hc = raw_data[(raw_data.brain_area == 'hc') | (raw_data.brain_area == 'hc2') | (raw_data.brain_area == 'ca1') | (raw_data.brain_area == 'ca2') | (raw_data.brain_area == 'ca3') | (raw_data.brain_area == 'dg')]

raw_mpfc = raw_data[(raw_data.brain_area == 'mpfc') | (raw_data.brain_area == 'pl') | (raw_data.brain_area == 'ila') | (raw_data.brain_area == 'scACC')]

raw_ofc = raw_data[(raw_data.brain_area == 'ofc') | (raw_data.brain_area == 'orb')]
           
raw_striatum = raw_data[(raw_data.brain_area == 'vStriatum') | (raw_data.brain_area == 'putamen') | (raw_data.brain_area == 'caudate')]


#%%

survives = []

for r_2 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    
    filt_data = data[(data.r2 >= r_2)]
    
    acc = filt_data[(filt_data.brain_area == 'acc') | (filt_data.brain_area == 'dACC') | (filt_data.brain_area == 'aca') | (filt_data.brain_area == 'mcc')]

    amyg = filt_data[(filt_data.brain_area == 'amygdala') | (filt_data.brain_area == 'central') | (filt_data.brain_area == 'bla')]
    
    hc = filt_data[(filt_data.brain_area == 'hc') | (filt_data.brain_area == 'ca1') | (filt_data.brain_area == 'ca2') | (filt_data.brain_area == 'ca3') | (filt_data.brain_area == 'dg')]
    
    mpfc = filt_data[(filt_data.brain_area == 'mpfc') | (filt_data.brain_area == 'pl') | (filt_data.brain_area == 'ila') | (filt_data.brain_area == 'scACC')]
    
    ofc = filt_data[(filt_data.brain_area == 'ofc') | (filt_data.brain_area == 'orb')]
               
    striatum = filt_data[(filt_data.brain_area == 'vStriatum') | (filt_data.brain_area == 'putamen') | (filt_data.brain_area == 'caudate')]
    
    survives.append([r_2,'all_data',len(filt_data)/len(raw_data)])
    survives.append([r_2,'acc',len(acc)/len(raw_acc)])
    survives.append([r_2,'amyg',len(amyg)/len(raw_amyg)])
    survives.append([r_2,'hc',len(hc)/len(raw_hc)])
    survives.append([r_2,'mpfc',len(mpfc)/len(raw_mpfc)])
    survives.append([r_2,'ofc',len(ofc)/len(raw_ofc)])
    survives.append([r_2,'str',len(striatum)/len(raw_striatum)])

surv = pd.DataFrame(survives,columns=['r2','brain_area','prop'])

#%%

sns.catplot(data=surv,x='r2',y='prop',hue='brain_area')

plt.xlabel('R$^2$ threshold')
plt.ylabel('Prop surviving')

plt.show()