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

#%% plot tau and lat by area

order = ['11m','13l','13m','AI','11l']

ofc_lai['specific_area'] = pd.Categorical(ofc_lai['specific_area'],categories=order,ordered=True)

plt.figure(figsize=(11,8.5))

sns.pointplot(data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='fred_tau')

plt.xlabel('Cytoarchitectonic area')
plt.ylabel('Timescale (ms)')

plt.show()

plt.figure(figsize=(11,8.5))

sns.pointplot(data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='fred_lat')

plt.xlabel('Cytoarchitectonic area')
plt.ylabel('Latency (ms)')

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
        
        to_append.append((dataset,species,brain_area,unit + 1,np.nan,np.nan,np.nan,np.nan,np.nan,fred_tau,lat,np.nan,np.nan,np.nan,'mouse orb'))
        
        
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
        
        to_append.append((dataset,species,brain_area,unit + 1,np.nan,np.nan,np.nan,np.nan,np.nan,fred_tau,lat,np.nan,np.nan,np.nan,'human ofc'))
        
        
    except:
        
        pass
    
human_mouse = pd.DataFrame(to_append,columns=['dataset','species','brain_area','unit','zach_tau','zach_tau_sd','zach_fr','zach_n','zach_r2','fred_tau','fred_lat','fred_fr','fred_r2','tau_diff','specific_area'])

all_3_species = pd.concat((ofc_lai,human_mouse),ignore_index=True)

all_3_species = all_3_species[all_3_species.specific_area != '13b']

order = ['11m','13l','13m','AI','mouse orb','human ofc','11l']

all_3_species['specific_area'] = pd.Categorical(all_3_species['specific_area'], categories = order , ordered = True)

#%%

sns.pointplot(data=all_3_species[all_3_species.specific_area != '13b'],x='specific_area',y='fred_tau')

plt.show()

sns.pointplot(data=all_3_species[all_3_species.specific_area != '13b'],x='specific_area',y='fred_lat')

plt.show()

# %%
