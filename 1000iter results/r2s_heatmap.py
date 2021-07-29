#%%

"""
Created on Mon Jun 14 16:45:25 2021

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

raw_acc['brain_region'] = 'acc'
raw_amyg['brain_region'] = 'amygdala'
raw_hc['brain_region'] = 'hippocampus'
raw_mpfc['brain_region'] = 'mpfc'
raw_ofc['brain_region'] = 'ofc'
raw_striatum['brain_region'] = 'striatum'

brain_region_raw = pd.concat((raw_acc,raw_amyg,raw_hc,raw_mpfc,raw_ofc,raw_striatum))

#%% r2 distributions by brain region

# ACC

bins = np.linspace(0,1,21)

r2s = []

for dataset in raw_acc.dataset.unique():
    
    these_data = raw_acc[raw_acc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('ACC')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# Amygdala

bins = np.linspace(0,1,21)

r2s = []

for dataset in raw_amyg.dataset.unique():
    
    these_data = raw_amyg[raw_amyg.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('Amygdala')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# Hippocampus

bins = np.linspace(0,1,21)

r2s = []

for dataset in raw_hc.dataset.unique():
    
    these_data = raw_hc[raw_hc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('Hippocampus')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# mPFC

bins = np.linspace(0,1,21)

r2s = []

for dataset in raw_mpfc.dataset.unique():
    
    these_data = raw_mpfc[raw_mpfc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('mPFC')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# OFC

bins = np.linspace(0,1,21)

r2s = []

for dataset in raw_ofc.dataset.unique():
    
    these_data = raw_ofc[raw_ofc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('OFC')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# Striatum

bins = np.linspace(0,1,21)

r2s = []

for dataset in raw_striatum.dataset.unique():
    
    these_data = raw_striatum[raw_striatum.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('Striatum')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

#%% Do again on filtered data

# ACC

bins = np.linspace(0,1,21)

r2s = []

for dataset in acc.dataset.unique():
    
    these_data = acc[acc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('ACC')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# Amygdala

bins = np.linspace(0,1,21)

r2s = []

for dataset in amyg.dataset.unique():
    
    these_data = amyg[amyg.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('Amygdala')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# Hippocampus

bins = np.linspace(0,1,21)

r2s = []

for dataset in hc.dataset.unique():
    
    these_data = hc[hc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('Hippocampus')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# mPFC

bins = np.linspace(0,1,21)

r2s = []

for dataset in mpfc.dataset.unique():
    
    these_data = mpfc[mpfc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('mPFC')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# OFC

bins = np.linspace(0,1,21)

r2s = []

for dataset in ofc.dataset.unique():
    
    these_data = ofc[ofc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('OFC')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

# Striatum

bins = np.linspace(0,1,21)

r2s = []

for dataset in striatum.dataset.unique():
    
    these_data = striatum[striatum.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            r2s.append(r2_dist)
        
r2s = np.vstack(r2s)

bins = np.around(bins[:-1],decimals=2)

sns.heatmap(r2s,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'})
plt.title('Striatum')
plt.xlabel('R$^2$')
plt.ylabel('unit')

plt.show()

#%% Is r2 peak at top (how many iters above 0.8) related to tau spread (sd)

# ACC unfiltered

acc_corrs = []

bins = np.linspace(0,1,21)

for dataset in raw_acc.dataset.unique():
    
    these_data = raw_acc[raw_acc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            sd_tau = np.std(this_unit['tau'])
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            prop_good_r2 = (r2_dist < 0.8).sum()/len(r2_dist)
            
            acc_corrs.append((sd_tau,prop_good_r2))

acc_corrs = np.vstack(acc_corrs)

for unit in range(len(acc_corrs)):
    
    plt.scatter(acc_corrs[unit,0],acc_corrs[unit,1])
    
plt.xlabel('standard deviation of tau')
plt.ylabel('prop R$^2$ > 0.8')
plt.title('ACC unfiltered')
plt.show()

# ACC filtered

acc_corrs = []

bins = np.linspace(0,1,21)

for dataset in acc.dataset.unique():
    
    these_data = acc[acc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            sd_tau = np.std(this_unit['tau'])
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            prop_good_r2 = (r2_dist < 0.8).sum()/len(r2_dist)
            
            acc_corrs.append((sd_tau,prop_good_r2))

acc_corrs = np.vstack(acc_corrs)

for unit in range(len(acc_corrs)):
    
    plt.scatter(acc_corrs[unit,0],acc_corrs[unit,1])
    
plt.xlabel('standard deviation of tau')
plt.ylabel('prop R$^2$ > 0.8')
plt.title('ACC filtered')
plt.show()

#%% Amygdala

# Amygdala unfiltered

amyg_corrs = []

bins = np.linspace(0,1,21)

for dataset in raw_amyg.dataset.unique():
    
    these_data = raw_amyg[raw_amyg.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            sd_tau = np.std(this_unit['tau'])
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            prop_good_r2 = (r2_dist < 0.8).sum()/len(r2_dist)
            
            amyg_corrs.append((sd_tau,prop_good_r2))

amyg_corrs = np.vstack(amyg_corrs)

for unit in range(len(amyg_corrs)):
    
    plt.scatter(amyg_corrs[unit,0],amyg_corrs[unit,1])
    
plt.xlabel('standard deviation of tau')
plt.ylabel('prop R$^2$ > 0.8')
plt.title('Amygdala unfiltered')
plt.show()

# Amygdala filtered

amyg_corrs = []

bins = np.linspace(0,1,21)

for dataset in amyg.dataset.unique():
    
    these_data = amyg[amyg.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            sd_tau = np.std(this_unit['tau'])
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            prop_good_r2 = (r2_dist < 0.8).sum()/len(r2_dist)
            
            amyg_corrs.append((sd_tau,prop_good_r2))

amyg_corrs = np.vstack(amyg_corrs)

for unit in range(len(amyg_corrs)):
    
    plt.scatter(amyg_corrs[unit,0],amyg_corrs[unit,1])
    
plt.xlabel('standard deviation of tau')
plt.ylabel('prop R$^2$ > 0.8')
plt.title('Amygdala filtered')
plt.show()

#%% Hippocampus

# hc unfiltered

hc_corrs = []

bins = np.linspace(0,1,21)

for dataset in raw_hc.dataset.unique():
    
    these_data = raw_hc[raw_hc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            sd_tau = np.std(this_unit['tau'])
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            prop_good_r2 = (r2_dist < 0.8).sum()/len(r2_dist)
            
            hc_corrs.append((sd_tau,prop_good_r2))

hc_corrs = np.vstack(hc_corrs)

for unit in range(len(hc_corrs)):
    
    plt.scatter(hc_corrs[unit,0],hc_corrs[unit,1])
    
plt.xlabel('standard deviation of tau')
plt.ylabel('prop R$^2$ > 0.8')
plt.title('Hippocampus unfiltered')
plt.show()

# hc filtered

hc_corrs = []

bins = np.linspace(0,1,21)

for dataset in hc.dataset.unique():
    
    these_data = hc[hc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            sd_tau = np.std(this_unit['tau'])
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            prop_good_r2 = (r2_dist < 0.8).sum()/len(r2_dist)
            
            hc_corrs.append((sd_tau,prop_good_r2))

hc_corrs = np.vstack(hc_corrs)

for unit in range(len(hc_corrs)):
    
    plt.scatter(hc_corrs[unit,0],hc_corrs[unit,1])
    
plt.xlabel('standard deviation of tau')
plt.ylabel('prop R$^2$ > 0.8')
plt.title('Hippocampus filtered')
plt.show()

#%% OFC

# OFC unfiltered

ofc_corrs = []

bins = np.linspace(0,1,21)

for dataset in raw_ofc.dataset.unique():
    
    these_data = raw_ofc[raw_ofc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            sd_tau = np.std(this_unit['tau'])
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            prop_good_r2 = (r2_dist < 0.8).sum()/len(r2_dist)
            
            ofc_corrs.append((sd_tau,prop_good_r2))

ofc_corrs = np.vstack(ofc_corrs)

for unit in range(len(ofc_corrs)):
    
    plt.scatter(ofc_corrs[unit,0],ofc_corrs[unit,1])
    
plt.xlabel('standard deviation of tau')
plt.ylabel('prop R$^2$ > 0.8')
plt.title('OFC unfiltered')
plt.show()

# OFC filtered

ofc_corrs = []

bins = np.linspace(0,1,21)

for dataset in ofc.dataset.unique():
    
    these_data = ofc[ofc.dataset == dataset]
    
    for brain_area in these_data.brain_area.unique():
        
        this_area = these_data[these_data.brain_area == brain_area]
    
        for unit_n in this_area.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            sd_tau = np.std(this_unit['tau'])
            
            these_r2s = this_unit['r2']
            
            species = these_data.iloc[0]['species']
            
            r2_dist = np.histogram(these_r2s,bins=bins)[0]
            
            prop_good_r2 = (r2_dist < 0.8).sum()/len(r2_dist)
            
            ofc_corrs.append((sd_tau,prop_good_r2))

ofc_corrs = np.vstack(ofc_corrs)

for unit in range(len(ofc_corrs)):
    
    plt.scatter(ofc_corrs[unit,0],ofc_corrs[unit,1])
    
plt.xlabel('standard deviation of tau')
plt.ylabel('prop R$^2$ > 0.8')
plt.title('OFC filtered')
plt.show()