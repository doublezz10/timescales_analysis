#%%

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

new_data = pd.read_csv('processed_data.csv')
old_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/2023/old_data.csv')

#%%

# get matches

match_data = []

for dataset in new_data.dataset.unique():
    
    this_dataset = new_data[new_data.dataset==dataset]
    old_dataset = old_data[old_data.name==dataset]
    
    for brain_area in this_dataset.area.unique():
        
        this_area = this_dataset[this_dataset.area==brain_area]
        old_area = old_dataset[old_dataset.area==brain_area]
        
        for unit in this_area.unit.unique():
            
            this_unit = this_area[this_area.unit==unit]
            old_unit = old_area[old_area.unitID==unit]
            
            if (len(this_unit) > 0 and len(old_unit) > 0):
                
                this_unit = this_unit.iloc[0]
                old_unit = old_unit.iloc[0]
                
                new_tau = this_unit['tau']
                new_lat = this_unit['peak_lat']
                #new_r2 = this_unit['r2']
                new_spikes = this_unit['nbSpk']
                
                old_tau = old_unit['tau']
                old_lat = old_unit['lat']
                #old_r2 = old_unit['r2']
                old_spikes = old_unit['nbSpk']
                
                match_data.append((dataset,brain_area,unit,new_tau,new_lat,new_spikes,old_tau,old_lat,old_spikes))
                
match_data = pd.DataFrame(match_data,columns=['dataset','brain_area','unitID','new_tau','new_lat','new_spikes','old_tau','old_lat','old_spikes'])
            
#%%

# filter match data

good_data = []

for entry in range(len(match_data)):
    
    this_cell = match_data.iloc[entry]
    
    if np.logical_and(this_cell.new_tau > 10, this_cell.new_tau < 1000) and np.logical_and(this_cell.old_tau > 10, this_cell.old_tau < 1000):
        
        good_data.append(this_cell)
        
good_data = pd.DataFrame(good_data,columns=['dataset','brain_area','unitID','new_tau','new_lat','new_spikes','old_tau','old_lat','old_spikes'])

#%%

# do correlations

sns.lmplot(data=good_data[good_data.dataset=='fontanier'],x='old_tau',y='new_tau',hue='dataset')
plt.show()

# %%
