#%%

import numpy as np
from numpy.lib.function_base import median
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from scipy.io import loadmat

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

plt.rcParams['font.size'] = '7'

plt.style.use('seaborn')

#%%

ofc = pd.read_csv('fred_ofc.csv')
lai = pd.read_csv('fred_lai.csv')

ofc_lai = pd.concat((ofc,lai),ignore_index=True)

order=['11m','13l','13m','AI','11l']

ofc_lai['specific_area'] = pd.Categorical(ofc_lai['specific_area'],categories=order,ordered=True)

#%%

fig= plt.figure(figsize=(3.4,2.6))

sns.countplot(data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area')

plt.tick_params(axis='x', labelsize=7)
plt.tick_params(axis='y',labelsize=7)

plt.xlabel('cytoarchitectonic area',fontdict={'fontsize':7})
plt.ylabel('number of neurons',fontdict={'fontsize':7})

plt.show()

#%%

fig= plt.figure(figsize=(3.4,2.6))

sns.lineplot(data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='fred_tau')

plt.tick_params(axis='x', labelsize=7)
plt.tick_params(axis='y',labelsize=7)

plt.xlabel('')
plt.ylabel('timescale (ms)',fontdict={'fontsize':7})

plt.show()

#%%

fig= plt.figure(figsize=(3.4,2.6))

sns.lineplot(data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='fred_lat')

plt.tick_params(axis='x', labelsize=7)
plt.tick_params(axis='y',labelsize=7)

plt.xlabel('')
plt.ylabel('latency (ms)',fontdict={'fontsize':7})

plt.show()
# %%
