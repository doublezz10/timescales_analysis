#%% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import ptitprince as pt

import seaborn as sns

plt.style.use('seaborn')

#%% Filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixedpop.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

#%%

acc = data[(data.brain_area == 'acc') | (data.brain_area == 'dACC') | (data.brain_area == 'aca') | (data.brain_area == 'mcc')]

amyg = data[(data.brain_area == 'amygdala') | (data.brain_area == 'central') | (data.brain_area == 'bla')]

hc = data[(data.brain_area == 'hc') | (data.brain_area == 'ca1') | (data.brain_area == 'ca2') | (data.brain_area == 'ca3') | (data.brain_area == 'dg')]

mpfc = data[(data.brain_area == 'mpfc') | (data.brain_area == 'pl') | (data.brain_area == 'ila') | (data.brain_area == 'scACC')]

ofc = data[(data.brain_area == 'ofc') | (data.brain_area == 'orb')]
           
striatum = data[(data.brain_area == 'vStriatum') | (data.brain_area == 'putamen') | (data.brain_area == 'caudate')]

acc_g = acc.assign(brain_region = 'ACC')
amyg_g = amyg.assign(brain_region = 'Amygdala')
hc_g = hc.assign(brain_region = 'Hippocampus')
mpfc_g = mpfc.assign(brain_region = 'mPFC')
ofc_g = ofc.assign(brain_region = 'OFC')

grouped_data = pd.concat((acc_g,amyg_g,hc_g,mpfc_g,ofc_g))

brain_regions = ['Hippocampus','Amygdala','OFC','mPFC','ACC']

grouped_data['brain_region'] = pd.Categorical(grouped_data['brain_region'], categories = brain_regions , ordered = True)

#%%
import matplotlib

no_rats = grouped_data[grouped_data.species != 'rat']

norats = ['mouse','monkey','human']

no_rats['species'] = pd.Categorical(no_rats['species'], categories = norats , ordered = True)

mean_no_rats = []

for dataset in no_rats.dataset.unique():
    
    this_dataset = no_rats[no_rats.dataset == dataset]
    
    for brain_region in this_dataset.brain_region.unique():
        
        this_region = this_dataset[this_dataset.brain_region == brain_region]
        
        species = this_region.iloc[0].species
        
        mean_tau = np.mean(this_region['tau'])
        
        mean_fr = np.mean(this_region['fr'])
        
        mean_no_rats.append((species,dataset,brain_region,mean_tau,mean_fr))
        
mean_no_rats = pd.DataFrame(mean_no_rats,columns=['species','dataset','brain_region','mean_tau','mean_fr'])
mean_no_rats['species'] = pd.Categorical(mean_no_rats['species'], categories = norats , ordered = True)
mean_no_rats['brain_region'] = pd.Categorical(mean_no_rats['brain_region'], categories = brain_regions , ordered = True)

#%%

plt.figure(figsize=(11,8.5))

# matplotlib.rcParams.update({'font.size': 12})

sns.lineplot(data=no_rats,x='brain_region',y='tau',hue='species')

sns.scatterplot(data=mean_no_rats,x='brain_region',y='mean_tau',hue='species',legend=False)

plt.xlabel('brain region')
plt.ylabel('population timescale (ms)')

plt.ylim((100,550))

plt.show()

#%% GLM

model = smf.ols(formula='mean_tau ~ species + brain_region',data=mean_no_rats)

res = model.fit()

print(res.summary())

#%%

model2 = smf.ols(formula='mean_tau ~ species + brain_region + mean_fr', data=mean_no_rats)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

# %% Prove amygdala is different

model3 = smf.ols(formula='mean_tau ~ species',data=mean_no_rats[mean_no_rats.brain_region=='Amygdala'])
res3 = model3.fit()

print(res3.summary())
# %% Repeat with OFC

model4 = smf.ols(formula='mean_tau ~ species',data=mean_no_rats[mean_no_rats.brain_region=='OFC'])
res4 = model4.fit()

print(res4.summary())
# %%
