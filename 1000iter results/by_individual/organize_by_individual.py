#%%
 
import numpy as np
import pandas as pd

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

#%% Loop through units and assign individual/session id

all_individuals = pd.DataFrame()

for dataset in raw_data.dataset.unique():
    
    if dataset == 'steinmetz':
        
        map_df = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/steinmetz_map.csv',index_col=0)
        
        this_dataset = raw_data[raw_data.dataset == dataset]
        
        for brain_area in this_dataset.brain_area.unique():
            
            this_brain_area = this_dataset[this_dataset.brain_area == brain_area]
            map_brain_area = map_df[map_df.brain_area == brain_area]
            
            for unit in this_brain_area.unit.unique():
                
                this_unit = this_brain_area[this_brain_area.unit == unit]
                map_unit = map_brain_area[map_brain_area.unit_id == unit]
                
                this_individual = map_unit.session_idx
                
                this_unit = this_unit.assign(individual = this_individual)
                
                all_individuals = all_individuals.append(this_unit)
        
    elif dataset == 'chandravadia':
        
        this_dataset = raw_data[raw_data.dataset == dataset]
        
        for brain_area in this_dataset.brain_area.unique():
            
            if brain_area == 'amygdala':
                
                map_df = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/chadrandavia_amyg_map.csv',index_col=0)

                for unit in this_brain_area.unit.unique():
                
                    this_unit = this_brain_area[this_brain_area.unit == unit]
                    map_unit = map_df[map_df.unit_n == unit]
                    
                    this_individual = map_unit.session
                    
                    this_unit = this_unit.assign(individual = this_individual)
                    
                    all_individuals = all_individuals.append(this_unit)
                    
            elif brain_area == 'hippocampus':
            
                map_df = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/chadrandavia_hc_map.csv',index_col=0)

                for unit in this_brain_area.unit.unique():
                
                    this_unit = this_brain_area[this_brain_area.unit == unit]
                    map_unit = map_df[map_df.unit_n == unit]
                    
                    this_individual = map_unit.session
                    
                    this_unit = this_unit.assign(individual = this_individual)
                    
                    all_individuals = all_individuals.append(this_unit)
                    
# %% Output as csv

all_individuals.to_csv('by_individual.csv')

# %%
