#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:16:34 2021

@author: zachz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

print('Proportion of units surviving filtering:', len(data)/len(raw_data))

#%% Get mean values over all iterations

all_means = []

for dataset in data.dataset.unique():
    
    this_dataset = data[data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            # fixed python numbering here - first unit is index 1 not 0
            
            all_means.append((dataset,species,brain_area,unit_n + 1,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
    
all_means = pd.DataFrame(all_means,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

all_means['species'] = pd.Categorical(all_means['species'], categories=listofspecies, ordered=True)

del dataset, brain_area, this_dataset, these_data, this_unit, species, unit_n
del mean_tau, sd_tau, mean_r2, sd_r2, mean_fr, sd_fr, n

#%% Load in Fred's data

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')
fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
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

fred_data = fred_data[fred_data.r2 >= 0.5]

fred_data = fred_data[(fred_data.tau >=10) & (fred_data.tau <= 1000)]

#%% Loop through and pull out matching taus

matching_units = []

for dataset in all_means.dataset.unique():
    
    this_dataset = all_means[all_means.dataset == dataset]
    
    fred_dataset = fred_data[fred_data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]
        
        fred_these_data = fred_dataset[fred_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            fred_unit = fred_these_data[fred_these_data.unit == unit_n]
            
            if len(fred_unit) == 0:
                
                pass
            
            elif fred_unit['keep'].values[0] == 0:
                
                pass
            
            elif fred_unit['r2'].values[0] <= 0.5:
                
                pass
            
            else:
                
                zach_tau = this_unit['tau'].values[0]
                zach_tau_sd = this_unit['sd_tau'].values[0]
                zach_fr = this_unit['mean_fr'].values[0]
                zach_n = this_unit['n'].values[0]
                zach_r2 = this_unit['mean_r2'].values[0]
                
                fred_tau = fred_unit['tau'].values[0]
                fred_fr = fred_unit['FR'].values[0]
                fred_r2 = fred_unit['r2'].values[0]
                
                tau_diff = zach_tau - fred_tau
                
                matching_units.append((dataset,brain_area,unit_n,zach_tau,zach_tau_sd,zach_fr,zach_n,zach_r2,fred_tau,fred_fr,fred_r2,tau_diff))
            
matching_units = pd.DataFrame(matching_units,columns=['dataset','brain_area','unit','zach_tau','zach_tau_sd','zach_fr','zach_n','zach_r2','fred_tau','fred_fr','fred_r2','tau_diff'])

#%% Separate by brain area

acc = matching_units[(matching_units.brain_area == 'acc') | (matching_units.brain_area == 'dACC') | (matching_units.brain_area == 'aca')]

amyg = matching_units[(matching_units.brain_area == 'amygdala') | (matching_units.brain_area == 'central') | (matching_units.brain_area == 'bla')]

hc = matching_units[(matching_units.brain_area == 'hc') | (matching_units.brain_area == 'ca1') | (matching_units.brain_area == 'ca2') | (matching_units.brain_area == 'ca3') | (matching_units.brain_area == 'dg')]

mpfc = matching_units[(matching_units.brain_area == 'mpfc') | (matching_units.brain_area == 'pl') | (matching_units.brain_area == 'ila') | (matching_units.brain_area == 'scACC')]

ofc = matching_units[(matching_units.brain_area == 'ofc') | (matching_units.brain_area == 'orb')]
           
striatum = matching_units[(matching_units.brain_area == 'vStriatum') | (matching_units.brain_area == 'putamen') | (matching_units.brain_area == 'caudate')]

#%% New dataframe with all brain regions stacked

acc2 = acc.assign(brain_region='ACC')
amyg2 = amyg.assign(brain_region='Amygdala')
hc2 = hc.assign(brain_region='Hippocampus')
mpfc2 = mpfc.assign(brain_region='mPFC')
ofc2 = ofc.assign(brain_region='OFC')
striatum2 = striatum.assign(brain_region='Striatum')

brain_region_data = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2,striatum2))

#%% Do correlation plots

sns.lmplot(data=acc,x='zach_tau',y='fred_tau',hue='dataset',scatter=False,legend=False)
sns.scatterplot(data=acc,x='zach_tau',y='fred_tau',hue='dataset',alpha=0.4,legend=False)
plt.title('ACC')
plt.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')
plt.legend(loc='upper left')
plt.xlabel('Zach tau (ms)')
plt.ylabel('Fred tau (ms)')
plt.show()

sns.lmplot(data=amyg,x='zach_tau',y='fred_tau',hue='dataset',scatter=False,legend=False)
sns.scatterplot(data=amyg,x='zach_tau',y='fred_tau',hue='dataset',alpha=0.4,legend=False)
plt.title('Amygdala')
plt.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')
plt.legend(loc='upper left')
plt.show()

sns.lmplot(data=hc,x='zach_tau',y='fred_tau',hue='dataset',scatter=False,legend=False)
sns.scatterplot(data=hc,x='zach_tau',y='fred_tau',hue='dataset',alpha=0.4,legend=False)
plt.title('Hippocampus')
plt.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')
plt.legend(loc='upper left')
plt.show()

sns.lmplot(data=mpfc,x='zach_tau',y='fred_tau',hue='dataset',scatter=False,legend=False)
sns.scatterplot(data=mpfc,x='zach_tau',y='fred_tau',hue='dataset',alpha=0.4,legend=False)
plt.title('mPFC')
plt.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')
plt.legend(loc='upper left')
plt.show()

sns.lmplot(data=ofc,x='zach_tau',y='fred_tau',hue='dataset',scatter=False,legend=False)
sns.scatterplot(data=ofc,x='zach_tau',y='fred_tau',hue='dataset',alpha=0.4,legend=False)
plt.title('OFC')
plt.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')
plt.legend(loc='upper left')
plt.show()

sns.lmplot(data=striatum,x='zach_tau',y='fred_tau',hue='dataset',scatter=False,legend=False)
sns.scatterplot(data=striatum,x='zach_tau',y='fred_tau',hue='dataset',alpha=0.4,legend=False)
plt.title('Striatum')
plt.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')
plt.legend(loc='upper left')
plt.show()

#%% one big subplot

g = sns.FacetGrid(data=brain_region_data,hue='dataset',col='brain_region',col_wrap=3,legend_out=True)
g.map_dataframe(sns.regplot,x='zach_tau',y='fred_tau',scatter_kws={'s':5, 'alpha': 0.5},ci=None)

for a in g.axes:
    a.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')
    
plt.tight_layout()
plt.show()

#%% Prettier way to do it

sns.lmplot(data=brain_region_data,x='zach_tau',y='fred_tau',hue='dataset',col='brain_region',col_wrap=3,ci=None,scatter_kws={'s':5, 'alpha': 0.5})

plt.show()

#%% Is tau diff related to anything

sns.lmplot(data=brain_region_data,x='zach_r2',y='tau_diff',hue='dataset',col='brain_region',col_wrap=3,ci=None,scatter_kws={'s':5, 'alpha': 0.5})

plt.show()

sns.lmplot(data=brain_region_data,x='fred_r2',y='tau_diff',hue='dataset',col='brain_region',col_wrap=3,ci=None,scatter_kws={'s':5, 'alpha': 0.5})

plt.show()

sns.lmplot(data=brain_region_data,x='zach_r2',y='fred_r2',hue='dataset',col='brain_region',col_wrap=3,ci=None,scatter_kws={'s':5, 'alpha': 0.5})

plt.show()

sns.lmplot(data=brain_region_data,x='zach_tau',y='tau_diff',hue='dataset',col='brain_region',col_wrap=3,ci=None,scatter_kws={'s':5, 'alpha': 0.5})

plt.show()

sns.lmplot(data=brain_region_data,x='fred_tau',y='tau_diff',hue='dataset',col='brain_region',col_wrap=3,ci=None,scatter_kws={'s':5, 'alpha': 0.5})

plt.show()