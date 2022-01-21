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

plt.style.use('seaborn')

#%%

ofc = pd.read_csv('fred_ofc.csv')
lai = pd.read_csv('fred_lai.csv')

ofc_lai = pd.concat((ofc,lai),ignore_index=True)

order=['11m','13m','13l','AI','11l','13b']

ofc_lai['specific_area'] = pd.Categorical(ofc_lai['specific_area'],categories=order,ordered=True)

#%%

sns.scatterplot(data=ofc_lai,x='zach_tau',y='depth',hue='specific_area',size=0.3,alpha=0.6)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel('iterative tau (ms)')
plt.ylabel('depth (mm)')

plt.show()

#%% Sorted by mean (median just switches 13b and 11l)

sns.violinplot(data=ofc_lai,x='specific_area',y='zach_tau',cut=0)

plt.xlabel('Cytoarchitectonic area')
plt.ylabel('Iteratively fit tau (ms)')

plt.show()

#%%

order=['11m','13m','13l','AI','11l']

ofc_lai['specific_area'] = pd.Categorical(ofc_lai['specific_area'],categories=order,ordered=True)

plt.figure(figsize=(11,8.5))

sns.pointplot(data=ofc_lai[ofc_lai.specific_area != '13b'],x='specific_area',y='zach_tau')

plt.xlabel('Cytoarchitectonic area')
plt.ylabel('Iteratively fit tau (ms)')

plt.show()

#%%

order=['11m','13m','13l','AI','11l','13b']

ofc_lai['specific_area'] = pd.Categorical(ofc_lai['specific_area'],categories=order,ordered=True)


sns.scatterplot(data=ofc_lai,x='fred_lat',y='depth',hue='specific_area',size=0.3,alpha=0.6)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel('ISI latency (ms)')
plt.ylabel('depth (mm)')

plt.show()

#%% Sorted by mean (median just switches 13b and 11l)

sns.violinplot(data=ofc_lai,x='specific_area',y='fred_lat',)

plt.show()

#%%

sns.pointplot(data=ofc_lai,x='specific_area',y='fred_lat')

plt.show()
#%%

model = smf.ols('zach_tau ~ specific_area',data=ofc_lai)

res = model.fit()

print(res.summary())

# %%

model2 = smf.ols('zach_tau ~ specific_area + depth',data=ofc_lai)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

# %% Now let's add in human 'OFC' and mouse ORB

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

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
    
    mean_tau = np.mean(this_unit.tau)
    
    tau_sd = np.std(this_unit.tau)
    
    fr = np.mean(this_unit.fr)
    
    n = len(this_unit)
    
    r2 = np.mean(this_unit.r2)
    
    to_append.append((dataset,species,brain_area,unit + 1,mean_tau,tau_sd,n,r2,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'mouse orb'))
    
for unit in range(len(human_ofc.unit.unique())):
    
    dataset = 'minxha'
    
    species = 'human'
    
    brain_area = 'ofc'
    
    this_unit = mouse_orb[mouse_orb.unit == unit]
    
    mean_tau = np.mean(this_unit.tau)
    
    tau_sd = np.std(this_unit.tau)
    
    fr = np.mean(this_unit.fr)
    
    n = len(this_unit)
    
    r2 = np.mean(this_unit.r2)
    
    to_append.append((dataset,species,brain_area,unit + 1,mean_tau,tau_sd,n,r2,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'human ofc'))
    
human_mouse = pd.DataFrame(to_append,columns=['dataset','species','brain_area','unit','zach_tau','zach_tau_sd','zach_fr','zach_n','zach_r2','fred_tau','fred_lat','fred_fr','fred_r2','tau_diff','specific_area'])

all_3_species = pd.concat((ofc_lai,human_mouse),ignore_index=True)

#%%

all_3_species = all_3_species[all_3_species.specific_area != '13b']

order = ['11m','13m','13l','human ofc','AI','mouse orb','11l']

all_3_species['specific_area'] = pd.Categorical(all_3_species['specific_area'], categories = order , ordered = True)

#%% Sorted by mean (median just switches 13b and 11l)

sns.violinplot(data=all_3_species,x='specific_area',y='zach_tau',cut=0)

plt.show()

#%%

sns.pointplot(data=all_3_species[all_3_species.specific_area != '13b'],x='specific_area',y='zach_tau')

plt.show()
# %%
