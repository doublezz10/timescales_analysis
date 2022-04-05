#%% Imports

import numpy as np
from numpy.lib.function_base import median
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from scipy.io import loadmat

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

plt.style.use('seaborn')

#%% Load in data

ofc = pd.read_csv('fred_ofc_isi.csv')
lai = pd.read_csv('fred_lai_isi.csv')
vl = pd.read_csv('fred_vl_isi.csv')

ofc_lai = pd.concat((ofc,lai,vl),ignore_index=True)

ofc_lai['brain_area'] = ofc_lai['brain_area'].str.replace('LAI','AI')
ofc_lai['specific_area'] = ofc_lai['specific_area'].str.replace('LAI','AI')

listofspecies = ['mouse','monkey','human']
ofc_lai['species'] = pd.Categorical(ofc_lai['species'], categories=listofspecies, ordered=True)

ofc_lai = ofc_lai[ofc_lai.specific_area!='13b']
ofc_lai= ofc_lai[ofc_lai.specific_area != '45']

counts = ofc_lai.groupby('specific_area',as_index=False).size()

counts.loc[counts['specific_area'] == 'human\nOFC' , 'granularity'] = 'granular'
counts.loc[counts['specific_area'] == '11m', 'granularity'] = 'granular'
counts.loc[counts['specific_area']== '11l', 'granularity'] = 'granular'
counts.loc[counts['specific_area']== '12m', 'granularity'] = 'granular'
counts.loc[counts['specific_area']== '12l', 'granularity'] = 'granular'
counts.loc[counts['specific_area']== '12o', 'granularity'] = 'granular'
counts.loc[counts['specific_area']== '45', 'granularity'] = 'granular'
counts.loc[counts['specific_area']== '12r', 'granularity'] = 'dysgranular'
counts.loc[counts['specific_area'] =='13m', 'granularity'] = 'dysgranular'
counts.loc[counts['specific_area'] == '13l', 'granularity'] = 'dysgranular'
counts.loc[counts['specific_area'] =='mouse\nORB', 'granularity'] = 'agranular'
counts.loc[counts['specific_area'] =='AI', 'granularity'] = 'agranular'

counts['granularity'] = pd.Categorical(counts['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)

counts['specific_area'] = pd.Categorical(counts['specific_area'],categories=['11m','11l','12m','12l','12o','12r','13m','13l','AI'],ordered=True)

#%%

fig,axs = plt.subplots(1,1,figsize=(3.4,2))

sns.scatterplot(ax=axs,data=counts,x='specific_area',y='size',hue='granularity',palette='Set2')

axs.set_xlabel(None)
axs.tick_params(axis='x',rotation=0, labelsize=7)
axs.tick_params(axis='y',labelsize=7)
axs.set_ylabel('number of neurons',fontsize=7)
axs.legend(title='',prop={'size':7})

plt.show()
# %%
