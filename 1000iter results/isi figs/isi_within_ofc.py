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

#%% plot tau and lat by area

order = ['11m','11l','12m','12l','12o','45','12r','13m','13l','AI']

ofc_lai['specific_area'] = pd.Categorical(ofc_lai['specific_area'],categories=order,ordered=True)

fig, axs = plt.subplots(1,2,figsize=(6,5))

sns.lineplot(ax=axs[0],data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='tau',ci=95,estimator=np.mean)

axs[0].set_xlabel(None)
axs[0].tick_params(axis='x', labelsize=7)
axs[0].tick_params(axis='y',labelsize=7)
axs[0].set_ylabel('timescale (ms)',fontsize=7)


sns.lineplot(ax=axs[1],data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='lat',ci=95,estimator=np.mean)

axs[1].set_xlabel(None)
axs[1].tick_params(axis='x',labelsize=7)
axs[1].tick_params(axis='y',labelsize=7)
axs[1].set_ylabel('latency (ms)',fontsize=7)

axs[1].set_ylim(20,120)

plt.tight_layout()

plt.show()

#%%

data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data.csv')

mouse_orb = data[data.species=='mouse']
mouse_orb = mouse_orb[mouse_orb.brain_region=='OFC']
mouse_orb['specific_area'] = 'mouse\nORB'

human_ofc = data[data.species=='human']
human_ofc = human_ofc[human_ofc.brain_region=='OFC']
human_ofc['specific_area'] = 'human\nOFC'

human_mouse = pd.concat((mouse_orb,human_ofc),ignore_index=True)
    
all_3_species = pd.concat((ofc_lai,human_mouse),ignore_index=True)

all_3_species = all_3_species[all_3_species.specific_area != '13b']
all_3_species = all_3_species[all_3_species.specific_area != '45']

#order = ['human\nofc','11m','11l','13m','13l','AI','mouse\norb']

#all_3_species['specific_area'] = pd.Categorical(all_3_species['specific_area'], categories = order , ordered = True)

#all_3_species['species'] = pd.Categorical(all_3_species['species'], categories=listofspecies, ordered=True)

#%%

all_3_species.loc[all_3_species['specific_area'] == 'human\nOFC' , 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area'] == '11m', 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area']== '11l', 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area']== '12m', 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area']== '12l', 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area']== '12o', 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area']== '45', 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area']== '12r', 'granularity'] = 'dysgranular'
all_3_species.loc[all_3_species['specific_area'] =='13m', 'granularity'] = 'dysgranular'
all_3_species.loc[all_3_species['specific_area'] == '13l', 'granularity'] = 'dysgranular'
all_3_species.loc[all_3_species['specific_area'] =='mouse\nORB', 'granularity'] = 'agranular'
all_3_species.loc[all_3_species['specific_area'] =='AI', 'granularity'] = 'agranular'

all_3_species['granularity'] = pd.Categorical(all_3_species['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)


#%%

fig, axs = plt.subplots(1,4,figsize=(6.7,4), gridspec_kw={'width_ratios': [2, 1, 2, 1]})

sns.pointplot(ax=axs[0],data=all_3_species[all_3_species.species == 'monkey'],x='specific_area',y='tau',hue='granularity',order=['11m','11l','12m','12l','12o','12r','13m','13l','AI'],palette="Set2",ci=95,estimator=np.mean)

axs[0].set_xlabel(None)
axs[0].tick_params(axis='x', labelsize=7)
axs[0].tick_params(axis='y',labelsize=7)
axs[0].set_ylabel('timescale (ms)',fontsize=7)
axs[0].legend(title='',prop={'size':7})
axs[0].set_ylim(0,300)


sns.pointplot(ax=axs[1],data=all_3_species[all_3_species.species != 'monkey'],x='specific_area',y='tau',hue='granularity',palette="Set2",order=['human\nOFC','mouse\nORB'],ci=95,estimator=np.mean)
axs[1].tick_params(axis='x',labelsize=7)
axs[1].tick_params(axis='y',labelsize=7)
axs[1].set_xlabel(None)
axs[1].set_ylabel(None)
axs[1].get_legend().remove()
axs[1].set_ylim(0,300)
axs[1].set_yticklabels([])


sns.pointplot(ax=axs[2],data=all_3_species[all_3_species.species == 'monkey'],x='specific_area',y='lat',hue='granularity',order=['11m','11l','12m','12l','12o','12r','13m','13l','AI'],palette="Set2",ci=95,estimator=np.mean)

axs[2].set_xlabel(None)
axs[2].tick_params(axis='x',labelsize=7)
axs[2].tick_params(axis='y',labelsize=7)
axs[2].set_ylabel('latency (ms)',fontsize=7)
axs[2].get_legend().remove()
axs[2].set_ylim(0,70)


sns.pointplot(ax=axs[3],data=all_3_species[all_3_species.species != 'monkey'],x='specific_area',y='lat',hue='granularity',palette="Set2",order=['human\nOFC','mouse\nORB'],ci=95,estimator=np.mean)

axs[3].tick_params(axis='x',labelsize=7)
axs[3].tick_params(axis='y',labelsize=7)
axs[3].set_xlabel(None)
axs[3].set_ylabel(None)
axs[3].get_legend().remove()
axs[3].set_ylim(0,70)
axs[3].set_yticklabels([])

plt.tight_layout(pad=0.2)

plt.show()
# %%

fig, axs = plt.subplots(1,2,figsize=(6.75,3))

sns.pointplot(ax=axs[0],data=all_3_species,x='specific_area',y='tau',hue='granularity',order=['11m','11l','12m','12l','12o','12r','13m','13l','AI'],palette="Set2",ci=95,estimator=np.mean,scale=0.5,errwidth=1)

axs[0].set_xlabel(None)
axs[0].tick_params(axis='x', labelsize=7,)
axs[0].tick_params(axis='y',labelsize=7)
axs[0].set_ylabel('timescale (ms)',fontsize=7)
axs[0].set_ylim(0,300)
axs[0].legend(title='',prop={'size':7})


sns.pointplot(ax=axs[1],data=all_3_species,x='specific_area',y='lat',hue='granularity',order=['11m','11l','12m','12l','12o','12r','13m','13l','AI'],palette="Set2",ci=95,estimator=np.mean,scale=0.5,errwidth=1)

axs[1].set_xlabel(None)
axs[1].tick_params(axis='x',labelsize=7)
axs[1].tick_params(axis='y',labelsize=7)
axs[1].set_ylabel('latency (ms)',fontsize=7)
axs[1].get_legend().remove()
axs[1].set_ylim(0,70)

plt.tight_layout(pad=0.2)

plt.show()

#%%
