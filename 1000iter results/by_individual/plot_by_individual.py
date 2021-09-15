#%% Load data, calculate mean across iterations
 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

individual_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/by_individual.csv')
listofspecies = ['mouse','rat','monkey','human']

individual_data['species'] = pd.Categorical(individual_data['species'], categories = listofspecies , ordered = True)

#

all_means = []

for dataset in individual_data.dataset.unique():
    
    this_dataset = individual_data[individual_data.dataset == dataset]
    
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
                
                try:
                
                    all_means.append((dataset,species,brain_area,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n,individual))
                    
                except:
                    
                    pass
    
all_means = pd.DataFrame(all_means,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n','individual'])

all_means['species'] = pd.Categorical(all_means['species'], categories=listofspecies, ordered=True)

# get mean across iterations

# then plot separated by individual (color?)
# or each graph is a dataset, col is brain_area, and x = individual

#%%

sns.catplot(data=all_means[all_means.dataset=='steinmetz'],x='unit',y='tau',hue='individual',col='brain_area',kind='bar',col_wrap=5)

plt.show()
# %%

sns.catplot(data=individual_data[individual_data.dataset=='chandravadia'],x='individual',y='tau',col='brain_area',kind='bar')

plt.show()
# %%
