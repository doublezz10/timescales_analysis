#%%

"""
Created on Tue Apr 13 15:00:07 2021

@author: zachz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import ptitprince as pt
import math

import seaborn as sns

plt.style.use('default')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

# raw data

raw_acc = raw_data[(raw_data.brain_area == 'acc') | (raw_data.brain_area == 'dACC') | (raw_data.brain_area == 'aca') | (raw_data.brain_area == 'mcc')]

raw_amyg = raw_data[(raw_data.brain_area == 'amygdala') | (raw_data.brain_area == 'central') | (raw_data.brain_area == 'bla')]

raw_hc = raw_data[(raw_data.brain_area == 'hc') | (raw_data.brain_area == 'hc2') | (raw_data.brain_area == 'ca1') | (raw_data.brain_area == 'ca2') | (raw_data.brain_area == 'ca3') | (raw_data.brain_area == 'dg')]

raw_mpfc = raw_data[(raw_data.brain_area == 'mpfc') | (raw_data.brain_area == 'pl') | (raw_data.brain_area == 'ila') | (raw_data.brain_area == 'scACC')]

raw_ofc = raw_data[(raw_data.brain_area == 'ofc') | (raw_data.brain_area == 'orb')]
           
raw_striatum = raw_data[(raw_data.brain_area == 'vStriatum') | (raw_data.brain_area == 'putamen') | (raw_data.brain_area == 'caudate')]

# filtered data

acc = data[(data.brain_area == 'acc') | (data.brain_area == 'dACC') | (data.brain_area == 'aca') | (data.brain_area == 'mcc')]

amyg = data[(data.brain_area == 'amygdala') | (data.brain_area == 'central') | (data.brain_area == 'bla')]

hc = data[(data.brain_area == 'hc') | (data.brain_area == 'ca1') | (data.brain_area == 'ca2') | (data.brain_area == 'ca3') | (data.brain_area == 'dg')]

mpfc = data[(data.brain_area == 'mpfc') | (data.brain_area == 'pl') | (data.brain_area == 'ila') | (data.brain_area == 'scACC')]

ofc = data[(data.brain_area == 'ofc') | (data.brain_area == 'orb')]
           
striatum = data[(data.brain_area == 'vStriatum') | (data.brain_area == 'putamen') | (data.brain_area == 'caudate')]

acc2 = acc.assign(brain_region = 'acc')
amyg2 = amyg.assign(brain_region = 'amygdala')
hc2 = hc.assign(brain_region = 'hippocampus')
mpfc2 = mpfc.assign(brain_region = 'mpfc')
ofc2 = ofc.assign(brain_region = 'ofc')
str2 = striatum.assign(brain_region = 'striatum')

brain_region_data = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2,str2))

#%%

# sns.catplot(data=amyg,x='unit',y='tau',col='species',col_wrap=2,s=0.5)
# plt.xticks([])

# plt.show()

#%% this takes forever to run lol

for dataset in brain_region_data.dataset.unique():
    
    this_data = brain_region_data[brain_region_data.dataset == dataset]

    for brain_region in this_data.brain_region.unique():
        
        this_region = this_data[this_data.brain_region == brain_region]
        
        units = this_region.unit.unique()
        
        n = 50
        
        unit_slices = [units[i:i+n] for i in range(0,len(units),n)]
        
        for fig in unit_slices:
            
            these_units = this_region.loc[this_region['unit'].isin(fig)]
            
            plt.figure(figsize=(25,5))
            
            f = sns.violinplot(data=these_units,x='unit',y='tau',aspect='auto')
            
            sep = ', '
                            
            plt.title(sep.join((dataset,brain_region)))
            
            plt.ylim((0,1000))
            
            plt.xlabel('unit')
        
            plt.show()
                
           