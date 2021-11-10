#%% Load data, calculate mean across iterations
 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

individual_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/by_individual.csv')
listofspecies = ['mouse','rat','monkey','human']

individual_data['species'] = pd.Categorical(individual_data['species'], categories = listofspecies , ordered = True)

#%%

all_means = []

for dataset in individual_data.dataset.unique():
    
    this_dataset = individual_data[individual_data.dataset == dataset]
    
    individuals = this_dataset.individual.unique()
    individual_key = range(len(individuals))
    
    zipped = zip(individuals,individual_key)
    
    individual_dict = dict(zipped)
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            if len(this_unit) < 100:
                
                pass
            
            else:
            
                species = this_unit.iloc[0]['species']
                
                brain_area = this_unit.iloc[0]['brain_area']

                mean_tau = np.mean(this_unit['tau'])
                
                sd_tau = np.std(this_unit['tau'])
                
                mean_r2 = np.mean(this_unit['r2'])
                
                sd_r2 = np.std(this_unit['r2'])
                
                mean_fr = np.mean(this_unit['fr'])
                
                sd_fr = np.std(this_unit['fr'])
                
                n = len(this_unit)
                
                individual = this_unit.iloc[0]['individual']
                
                individual = individual_dict.get(individual)
                
                try:
                
                    all_means.append((dataset,species,brain_area,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n,individual))
                    
                except:
                    
                    pass
    
all_means = pd.DataFrame(all_means,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n','individual'])

all_means['species'] = pd.Categorical(all_means['species'], categories=listofspecies, ordered=True)

acc = all_means[(all_means.brain_area == 'acc') | (all_means.brain_area == 'dACC') | (all_means.brain_area == 'aca') | (all_means.brain_area == 'mcc')]

amyg = all_means[(all_means.brain_area == 'amygdala') | (all_means.brain_area == 'central') | (all_means.brain_area == 'bla')]

hc = all_means[(all_means.brain_area == 'hc') | (all_means.brain_area == 'ca1') | (all_means.brain_area == 'ca2') | (all_means.brain_area == 'ca3') | (all_means.brain_area == 'dg')]

mpfc = all_means[(all_means.brain_area == 'mpfc') | (all_means.brain_area == 'pl') | (all_means.brain_area == 'ila') | (all_means.brain_area == 'scACC')]

ofc = all_means[(all_means.brain_area == 'ofc') | (all_means.brain_area == 'orb')]

brain_region = pd.concat((acc,amyg,hc,mpfc,ofc))
        

#%%

import statsmodels.formula.api as smf

plt.figure(figsize=(11,8.5))

sns.catplot(data=all_means[all_means.dataset=='steinmetz'],x='individual',y='tau',col='brain_area',kind='box',col_wrap=3)

plt.xticks([])

plt.show()

# model

stein_data = all_means[all_means.dataset=='steinmetz']

model = smf.ols(formula='tau ~ individual',data=stein_data)

res = model.fit()

print(res.summary())
# %%

plt.figure(figsize=(11,8.5))

sns.catplot(data=all_means[all_means.dataset=='chandravadia'],x='individual',y='tau',col='brain_area',kind='box')

plt.xticks([])

plt.show()

# model

chan_data = all_means[all_means.dataset=='chandravadia']

model = smf.ols(formula='tau ~ individual',data=chan_data)

res = model.fit()

print(res.summary())
# %%
