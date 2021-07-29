#%%

"""
Created on Tue Jul 27 10:33:29 2021

@author: zachz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn.categorical import boxenplot
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import ptitprince as pt

import seaborn as sns

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

# data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

# data = data[(data.r2 >= 0.5)]

data = raw_data

#%% Loop over units, compute get taus for first 500 and second 500 iters

# for column 'half' 1 = iters 0-499, 2 = iters 500-1000

all_means1 = []
all_means2 = []

for dataset in data.dataset.unique():
    
    this_dataset = data[data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            if len(this_unit) < 100:
                
                pass
            
            else:
                
                first_half = this_unit[this_unit.iter < len(this_unit)/2]
            
                species = first_half.iloc[0]['species']
                
                brain_area = first_half.iloc[0]['brain_area']
                
                mean_tau1 = np.mean(first_half['tau'])
                
                sd_tau1 = np.std(first_half['tau'])
                
                mean_r21 = np.mean(first_half['r2'])
                
                sd_r21 = np.std(first_half['r2'])
                
                mean_fr1 = np.mean(first_half['fr'])
                
                sd_fr1 = np.std(first_half['fr'])
                
                n1 = len(first_half)
                
                half1 = 1
                
                second_half = this_unit[this_unit.iter >= len(this_unit)/2]
                
                species = second_half.iloc[0]['species']
                
                brain_area = second_half.iloc[0]['brain_area']
                
                mean_tau = np.mean(second_half['tau'])
                
                sd_tau = np.std(second_half['tau'])
                
                mean_r2 = np.mean(second_half['r2'])
                
                sd_r2 = np.std(second_half['r2'])
                
                mean_fr = np.mean(second_half['fr'])
                
                sd_fr = np.std(second_half['fr'])
                
                n = len(second_half)
                half2 = 2
                
                try:
                
                    all_means1.append((dataset,species,brain_area,unit_n,mean_tau1,sd_tau1,np.log10(mean_tau1),mean_r21,sd_r21,mean_fr1,sd_fr1,n1,half1))
                    all_means2.append((dataset,species,brain_area,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n,half2))
                    
                except:
                    
                    pass
    
all_means1 = pd.DataFrame(all_means1,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n','half'])
all_means2 = pd.DataFrame(all_means2,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n','half'])

all_means1['species'] = pd.Categorical(all_means1['species'], categories=listofspecies, ordered=True)
all_means2['species'] = pd.Categorical(all_means2['species'], categories=listofspecies, ordered=True)

split_halves = pd.concat((all_means1,all_means2))

split_halves['species'] = pd.Categorical(split_halves['species'], categories=listofspecies, ordered=True)

acc = split_halves[(split_halves.brain_area == 'acc') | (split_halves.brain_area == 'dACC') | (split_halves.brain_area == 'aca') | (split_halves.brain_area == 'mcc')]

amyg = split_halves[(split_halves.brain_area == 'amygdala') | (split_halves.brain_area == 'central') | (split_halves.brain_area == 'bla')]

hc = split_halves[(split_halves.brain_area == 'hc') | (split_halves.brain_area == 'ca1') | (split_halves.brain_area == 'ca2') | (split_halves.brain_area == 'ca3') | (split_halves.brain_area == 'dg')]

mpfc = split_halves[(split_halves.brain_area == 'mpfc') | (split_halves.brain_area == 'pl') | (split_halves.brain_area == 'ila') | (split_halves.brain_area == 'scACC')]

ofc = split_halves[(split_halves.brain_area == 'ofc') | (split_halves.brain_area == 'orb')]
           
striatum = split_halves[(split_halves.brain_area == 'vStriatum') | (split_halves.brain_area == 'putamen') | (split_halves.brain_area == 'caudate')]

acc['brain_region'] = 'acc'
amyg['brain_region'] = 'amygdala'
hc['brain_region'] = 'hippocampus'
mpfc['brain_region'] = 'mpfc'
ofc['brain_region'] = 'ofc'
striatum['brain_region'] = 'striatum'

halves = pd.concat((acc,amyg,hc,mpfc,ofc,striatum))

#%%

for brain_region in halves.brain_region.unique():

    if brain_region == 'striatum':

        pass

    else:

        sns.catplot(data=halves.loc[(halves['brain_region']==brain_region)],x='half',y='tau',col='species',kind='boxen')

        plt.suptitle(brain_region)

        plt.show()

# %% t tests for each brain area, separated by species

from scipy.stats import ttest_ind

for brain_region in halves.brain_region.unique():

    if brain_region == 'striatum':

        pass

    else:

        for species in listofspecies:

            dat = halves[halves.species == species]

            cat1 = dat[dat['half']==1]
            cat2 = dat[dat['half']==2]

            print(species,brain_region)

            print('p = ',ttest_ind(cat1.tau.values, cat2.tau.values)[1])
            
# %% Linear modelling to be fancy

model = smf.ols(formula='tau ~ species + brain_region + half',data=halves)

res = model.fit()

print(res.summary())
