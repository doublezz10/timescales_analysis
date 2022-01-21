#%%
"""
Created on Thu Apr  8 11:40:49 2021

@author: zachz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress

import seaborn as sns

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

data = raw_data

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
            
            all_means.append((dataset,species,brain_area,unit_n ,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
    
all_means = pd.DataFrame(all_means,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

all_means['species'] = pd.Categorical(all_means['species'], categories=listofspecies, ordered=True)

del dataset, brain_area, this_dataset, these_data, this_unit, species, unit_n
del mean_tau, sd_tau, mean_r2, sd_r2, mean_fr, sd_fr, n


#%% Load in traditional data

old_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/results.csv')

print('%i units fit using iterative method' %(len(all_means[all_means.species != 'rat'])))
print('%i units fit using traditional method' %(len(old_data[old_data.species != 'rat'])))
print('%.2f percent more units fit using iterative method' %((len(all_means[all_means.species != 'rat'])) / (len(old_data))  * 100) )

#%% Loop through and pull out matching taus

matching_units = []

for dataset in all_means.dataset.unique():
    
    this_dataset = all_means[all_means.dataset == dataset]
    
    one_fit_dataset = old_data[old_data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]
        
        one_fit_these_data = one_fit_dataset[one_fit_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            one_fit_unit = one_fit_these_data[one_fit_these_data.unit == unit_n]
            
            species = this_unit.iloc[0]['species']
            
            if len(one_fit_unit) == 0:
                
                pass
            
            elif one_fit_unit['r2'].values[0] <= 0.5:
                
                pass
            
            elif one_fit_unit['tau'].values[0] >= 1000:
            
                pass
            
            elif one_fit_unit['tau'].values[0] <= 10:
                
                pass
            
            else:
                
                zach_tau = this_unit['tau'].values[0]
                zach_tau_sd = this_unit['sd_tau'].values[0]
                zach_fr = this_unit['mean_fr'].values[0]
                zach_n = this_unit['n'].values[0]
                zach_r2 = this_unit['mean_r2'].values[0]
                
                one_fit_tau = one_fit_unit['tau'].values[0]
                one_fit_fr = one_fit_unit['fr'].values[0]
                one_fit_r2 = one_fit_unit['r2'].values[0]
                
                tau_diff = zach_tau - one_fit_tau
                
                if dataset == 'meg':
                    
                    dataset = 'young/mosher'
                
                matching_units.append((dataset,species,brain_area,unit_n,zach_tau,zach_tau_sd,zach_fr,zach_n,zach_r2,one_fit_tau,one_fit_fr,one_fit_r2,tau_diff))
            
matching_units = pd.DataFrame(matching_units,columns=['dataset','species','brain_area','unit','iter_tau','iter_tau_sd','iter_fr','iter_n','iter_r2','one_fit_tau','one_fit_fr','one_fit_r2','tau_diff'])

listofspecies = ['mouse','monkey','human']

matching_units['species'] = pd.Categorical(matching_units['species'], categories = listofspecies , ordered = True)

#%% Separate by brain area

acc = matching_units[(matching_units.brain_area == 'acc') | (matching_units.brain_area == 'dACC') | (matching_units.brain_area == 'aca') | (matching_units.brain_area == 'mcc')]

amyg = matching_units[(matching_units.brain_area == 'amygdala') | (matching_units.brain_area == 'central') | (matching_units.brain_area == 'bla')]

hc = matching_units[(matching_units.brain_area == 'hc') | (matching_units.brain_area == 'ca1') | (matching_units.brain_area == 'ca2') | (matching_units.brain_area == 'ca3') | (matching_units.brain_area == 'dg')]

mpfc = matching_units[(matching_units.brain_area == 'mpfc') | (matching_units.brain_area == 'pl') | (matching_units.brain_area == 'ila') | (matching_units.brain_area == 'scACC')]

ofc = matching_units[(matching_units.brain_area == 'ofc') | (matching_units.brain_area == 'orb')]

# New dataframe with all brain regions stacked

acc2 = acc.assign(brain_region='ACC')
amyg2 = amyg.assign(brain_region='Amygdala')
hc2 = hc.assign(brain_region='Hippocampus')
mpfc2 = mpfc.assign(brain_region='mPFC')
ofc2 = ofc.assign(brain_region='OFC')

brain_region_data = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2))

#%% one big subplot

plt.figure(figsize=(11,8.5))

g = sns.FacetGrid(data=brain_region_data,hue='dataset',col='brain_region',col_wrap=3,legend_out=True)
g.map_dataframe(sns.regplot,x='iter_tau',y='one_fit_tau',scatter_kws={'s':5, 'alpha': 0.5},ci=95)

for a in g.axes:
    a.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')
    a.set_xlabel('iter_tau')
    a.set_ylabel('fix_tau')
    a.legend()
plt.tight_layout()

plt.show()

#%% Prettier way to do it
plt.figure(figsize=(11,8.5))

sns.lmplot(data=brain_region_data,x='iter_tau',y='one_fit_tau',hue='dataset',ci=95,scatter_kws={'s':5, 'alpha': 0.5})

plt.plot(range(1000),range(1000),linestyle='--',color='black',label='identity')

plt.xlabel('Iteratively fit tau (ms)')
plt.ylabel('One fit tau (ms)')

plt.show()

#%%
_,_,r2,p,_ = linregress(brain_region_data.iter_tau,brain_region_data.one_fit_tau)

print('R^2 = %.2f, p = %.2f' %(r2,p))
#%%

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

model = smf.ols('iter_tau ~ one_fit_tau + dataset',data=brain_region_data)

model = model.fit()

print(model.summary())

model2 = smf.ols('iter_tau ~ one_fit_tau',data=brain_region_data)

model2 = model2.fit()

print(model2.summary())

#%%

print(anova_lm(model2,model))
# %%

model = smf.ols('iter_tau ~ one_fit_tau + dataset',data=brain_region_data)

model = model.fit()

print(model.summary())
# %%
