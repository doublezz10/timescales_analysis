#%%

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

listofspecies = ['mouse','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

data = data[data.species != 'rat']

data['species'] = pd.Categorical(data['species'], categories = listofspecies , ordered = True)

print('Proportion of units surviving filtering:', len(data)/len(raw_data))

#%% Separate by brain_region

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

#%% ACC

acc_means = []

for dataset in acc.dataset.unique():
    
    these_data = acc[acc.dataset == dataset]

    for unit_n in these_data.unit.unique():

        this_unit = these_data[these_data.unit == unit_n]
        
        if len(this_unit) < 100:
                
                pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                acc_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass
                
acc_means = pd.DataFrame(acc_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

acc_means['species'] = pd.Categorical(acc_means['species'], categories=listofspecies, ordered=True)

# Amygdala

amyg_means = []

for dataset in amyg.dataset.unique():
    
    these_data = amyg[amyg.dataset == dataset]

    for unit_n in these_data.unit.unique():

        this_unit = these_data[these_data.unit == unit_n]
        
        if len(this_unit) < 100:
                
            pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            amyg_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))

amyg_means = pd.DataFrame(amyg_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

amyg_means['species'] = pd.Categorical(amyg_means['species'], categories=listofspecies, ordered=True)

# Hippocampus

hc_means = []

for dataset in hc.dataset.unique():
    
    these_data = hc[hc.dataset == dataset]

    for unit_n in these_data.unit.unique():
        
        this_unit = these_data[these_data.unit == unit_n]

        if len(this_unit) < 100:
                
                pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                hc_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass

hc_means = pd.DataFrame(hc_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

hc_means['species'] = pd.Categorical(hc_means['species'], categories=listofspecies, ordered=True)

# mPFC

mpfc_means = []

for dataset in mpfc.dataset.unique():
    
    these_data = mpfc[mpfc.dataset == dataset]

    for unit_n in these_data.unit.unique():
        
        this_unit = these_data[these_data.unit == unit_n]

        if len(this_unit) < 100:
                
            pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                mpfc_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass

mpfc_means = pd.DataFrame(mpfc_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

mpfc_means['species'] = pd.Categorical(mpfc_means['species'], categories=listofspecies, ordered=True)

# OFC

ofc_means = []

for dataset in ofc.dataset.unique():
    
    these_data = ofc[ofc.dataset == dataset]

    for unit_n in these_data.unit.unique():
        
        this_unit = these_data[these_data.unit == unit_n]
        
        if len(this_unit) < 100:
                
            pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                ofc_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass

ofc_means = pd.DataFrame(ofc_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

ofc_means['species'] = pd.Categorical(ofc_means['species'], categories=listofspecies, ordered=True)

#%%

acc_means['brain_region'] = 'acc'
amyg_means['brain_region'] = 'amygdala'
hc_means['brain_region'] = 'hippocampus'
mpfc_means['brain_region'] = 'mpfc'
ofc_means['brain_region'] = 'ofc'

#%% Raincloud plots by brain area - mean tau

# ACC

dx = "species"; dy = "log_tau"; ort = "h"; sigma = .2

fig, ax = plt.subplots(figsize=(5,5))

pt.RainCloud(x = dx, y = dy, data = acc_means, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort,point_size=1)

# Calculate number of obs per group & median to position labels
medians = acc_means.groupby(['species'])['log_tau'].median().values
nobs = acc_means['species'].value_counts().values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

myorder = [0,1,2]
nobs = [nobs[i] for i in myorder]
 
# Add text to the figure
pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
   ax.text(medians[tick], pos[tick] - 0.22, nobs[tick],
            horizontalalignment='center',
            size='small',
            color='black',
            weight='semibold')

plt.xlim((1,3))

plt.title('ACC')
plt.show() 

#%% Amygdala

dx = "species"; dy = "log_tau"; ort = "h"; sigma = .2

fig, ax = plt.subplots(figsize=(5,5))

pt.RainCloud(x = dx, y = dy, data = amyg_means, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort,point_size=1)

# Calculate number of obs per group & median to position labels
medians = amyg_means.groupby(['species'])['log_tau'].median().values
nobs = amyg_means['species'].value_counts().values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

myorder = [2,0,1]
nobs = [nobs[i] for i in myorder]
 
# Add text to the figure
pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
    ax.text(medians[tick], pos[tick] - 0.22, nobs[tick],
            horizontalalignment='center',
            size='small',
            color='black',
            weight='semibold')

plt.xlim((1,3))

plt.title('Amygdala')
plt.show() 

#%% Hippocampus

dx = "species"; dy = "log_tau"; ort = "h"; sigma = .2

fig, ax = plt.subplots(figsize=(5,5))

pt.RainCloud(x = dx, y = dy, data = hc_means, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort,point_size=1)

# Calculate number of obs per group & median to position labels

myorder = [0,1,2]
nobs = [nobs[i] for i in myorder]

medians = hc_means.groupby(['species'])['log_tau'].median().values
nobs = hc_means['species'].value_counts().values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]
 
# Add text to the figure
pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
   ax.text(medians[tick], pos[tick] - 0.22, nobs[tick],
            horizontalalignment='center',
            size='small',
            color='black',
            weight='semibold')

plt.xlim((1,3))

plt.title('Hippocampus')
plt.show() 

#%% mPFC

dx = "species"; dy = "log_tau"; ort = "h"; sigma = .2

fig, ax = plt.subplots(figsize=(5,5))

pt.RainCloud(x = dx, y = dy, data = mpfc_means, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort,point_size=1)

# Calculate number of obs per group & median to position labels

myorder = [0,1]
nobs = [nobs[i] for i in myorder]

medians = mpfc_means.groupby(['species'])['log_tau'].median().values
nobs = mpfc_means['species'].value_counts().values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

myorder = [1,0,2]
nobs = [nobs[i] for i in myorder]
 
# Add text to the figure
pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
   ax.text(medians[tick], pos[tick] - 0.22, nobs[tick],
            horizontalalignment='center',
            size='small',
            color='black',
            weight='semibold')

plt.xlim((1,3))

plt.title('mPFC')
plt.show() 

#%% OFC

dx = "species"; dy = "log_tau"; ort = "h"; sigma = .2

fig, ax = plt.subplots(figsize=(5,5))

pt.RainCloud(x = dx, y = dy, data = ofc_means, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort,point_size=1)

# Calculate number of obs per group & median to position labels
medians = ofc_means.groupby(['species'])['log_tau'].median().values
nobs = ofc_means['species'].value_counts().values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

myorder = [1,0,2]
nobs = [nobs[i] for i in myorder]
 
# Add text to the figure
pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
   ax.text(medians[tick], pos[tick] - 0.22, nobs[tick],
            horizontalalignment='center',
            size='small',
            color='black',
            weight='semibold')

plt.xlim((1,3))

plt.title('OFC')
plt.show() 

# %%
