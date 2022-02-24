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

ofc = pd.read_csv('fred_ofc.csv')
lai = pd.read_csv('fred_lai.csv')

ofc_lai = pd.concat((ofc,lai),ignore_index=True)

ofc_lai['brain_area'] = ofc_lai['brain_area'].str.replace('LAI','AI')
ofc_lai['specific_area'] = ofc_lai['specific_area'].str.replace('LAI','AI')

listofspecies = ['mouse','monkey','human']
ofc_lai['species'] = pd.Categorical(ofc_lai['species'], categories=listofspecies, ordered=True)

#%% plot tau and lat by area

order = ['11m','11l','13m','13l','AI']

ofc_lai['specific_area'] = pd.Categorical(ofc_lai['specific_area'],categories=order,ordered=True)

fig, axs = plt.subplots(1,2,figsize=(4,2.5))

sns.lineplot(ax=axs[0],data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='fred_tau')

axs[0].set_xlabel(None)
axs[0].tick_params(axis='x', labelsize=7)
axs[0].tick_params(axis='y',labelsize=7)
axs[0].set_ylabel('timescale (ms)',fontsize=7)


sns.lineplot(ax=axs[1],data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='fred_lat')

axs[1].set_xlabel(None)
axs[1].tick_params(axis='x',labelsize=7)
axs[1].tick_params(axis='y',labelsize=7)
axs[1].set_ylabel('latency (ms)',fontsize=7)

axs[1].set_ylim(20,120)

plt.tight_layout()

plt.show()

#%%

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')
fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})

listofspecies=['mouse','monkey','human']

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

# rename columns to match

fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus','hc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('mPFC','mpfc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('ventralStriatum','vStriatum')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('AMG','amygdala')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('Cd','caudate')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('OFC','ofc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('PUT','putamen')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus2','hc2')

fred_data['dataset'] = fred_data['dataset'].str.replace('stein','steinmetz')

raw_data = fred_data[fred_data.keep==1]

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

data = data[data.species != 'rat']

data['species'] = pd.Categorical(data['species'], categories = listofspecies , ordered = True)

mouse_orb = data[data.dataset=='steinmetz']
mouse_orb = mouse_orb[mouse_orb.brain_area=='orb']

human_ofc = data[data.dataset=='minxha']
human_ofc = human_ofc[human_ofc.brain_area=='ofc']

to_append = []

for unit in range(len(mouse_orb.unit.unique())):
    
    dataset = 'steinmetz'
    
    species = 'mouse'
    
    brain_area = 'orb'
    
    this_unit = mouse_orb[mouse_orb.unit == unit]
    
    try:
    
        fred_tau = np.array(this_unit.tau)[0]
        
        lat = np.array(this_unit.lat)[0]
        
        to_append.append((dataset,species,brain_area,unit + 1,np.nan,np.nan,np.nan,np.nan,np.nan,fred_tau,lat,np.nan,np.nan,np.nan,'mouse\nORB'))
        
        
    except:
        
        pass
    
for unit in range(len(human_ofc.unit.unique())):
    
    dataset = 'minxha'
    
    species = 'human'
    
    brain_area = 'ofc'
    
    this_unit = human_ofc[human_ofc.unit == unit]
    
    try:
    
        fred_tau = np.array(this_unit.tau)[0]
        
        lat = np.array(this_unit.lat)[0]
        
        to_append.append((dataset,species,brain_area,unit + 1,np.nan,np.nan,np.nan,np.nan,np.nan,fred_tau,lat,np.nan,np.nan,np.nan,'human\nOFC'))
        
        
    except:
        
        pass
    
human_mouse = pd.DataFrame(to_append,columns=['dataset','species','brain_area','unit','zach_tau','zach_tau_sd','zach_fr','zach_n','zach_r2','fred_tau','fred_lat','fred_fr','fred_r2','tau_diff','specific_area'])

all_3_species = pd.concat((ofc_lai,human_mouse),ignore_index=True)

all_3_species = all_3_species[all_3_species.specific_area != '13b']

#order = ['human\nofc','11m','11l','13m','13l','AI','mouse\norb']

#all_3_species['specific_area'] = pd.Categorical(all_3_species['specific_area'], categories = order , ordered = True)

#all_3_species['species'] = pd.Categorical(all_3_species['species'], categories=listofspecies, ordered=True)

#%%

all_3_species.loc[all_3_species['specific_area'] == 'human\nOFC' , 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area'] == '11m', 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area']== '11l', 'granularity'] = 'granular'
all_3_species.loc[all_3_species['specific_area'] =='13m', 'granularity'] = 'dysgranular'
all_3_species.loc[all_3_species['specific_area'] == '13l', 'granularity'] = 'dysgranular'
all_3_species.loc[all_3_species['specific_area'] =='mouse\nORB', 'granularity'] = 'agranular'
all_3_species.loc[all_3_species['specific_area'] =='AI', 'granularity'] = 'agranular'

all_3_species['granularity'] = pd.Categorical(all_3_species['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)


#%%

fig, axs = plt.subplots(1,4,figsize=(6.85,3), gridspec_kw={'width_ratios': [2, 1, 2, 1]},sharey=True)

sns.pointplot(ax=axs[0],data=all_3_species[all_3_species.species == 'monkey'],x='specific_area',y='fred_tau',hue='granularity',order=['11m','11l','13m','13l','AI'],palette="Set2")

axs[0].set_xlabel(None)
axs[0].tick_params(axis='x', labelsize=7)
axs[0].tick_params(axis='y',labelsize=7)
axs[0].set_ylabel('timescale (ms)',fontsize=7)
axs[0].legend(title='',prop={'size':7})


sns.pointplot(ax=axs[1],data=all_3_species[all_3_species.species != 'monkey'],x='specific_area',y='fred_tau',hue='granularity',palette="Set2",order=['human\nOFC','mouse\nORB'])
axs[1].tick_params(axis='x',labelsize=7)
axs[1].tick_params(axis='y',labelsize=7)
axs[1].set_xlabel(None)
axs[1].set_ylabel(None)
axs[1].get_legend().remove()


sns.pointplot(ax=axs[2],data=all_3_species[all_3_species.species == 'monkey'],x='specific_area',y='fred_lat',hue='granularity',order=['11m','11l','13m','13l','AI'],palette="Set2")

axs[2].set_xlabel(None)
axs[2].tick_params(axis='x',labelsize=7)
axs[2].tick_params(axis='y',labelsize=7)
axs[2].set_ylabel('latency (ms)',fontsize=7)
axs[2].get_legend().remove()


sns.pointplot(ax=axs[3],data=all_3_species[all_3_species.species != 'monkey'],x='specific_area',y='fred_lat',hue='granularity',palette="Set2",order=['human\nOFC','mouse\nORB'])

axs[3].tick_params(axis='x',labelsize=7)
axs[3].tick_params(axis='y',labelsize=7)
axs[3].set_xlabel(None)
axs[3].set_ylabel(None)
axs[3].get_legend().remove()

plt.tight_layout()

plt.show()
# %%
