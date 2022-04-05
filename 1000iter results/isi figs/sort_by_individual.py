#%%
 
import numpy as np
import pandas as pd

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

data = fred_data


#%%

all_individuals = pd.DataFrame()

for dataset in data.dataset.unique():
    
    if dataset == 'steinmetz':
        
        map_df = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/steinmetz_map.csv',index_col=0)
        
        this_dataset = data[data.dataset == dataset]
                
        individuals = this_dataset.individual.unique()
        individual_key = range(len(individuals))
        
        zipped = zip(individuals,individual_key)
        
        individual_dict = dict(zipped)
        
        for brain_area in this_dataset.brain_area.unique():
            
            this_brain_area = this_dataset[this_dataset.brain_area == brain_area]
            map_brain_area = map_df[map_df.brain_area == brain_area]
            
            for unit in this_brain_area.unit.unique():
                
                this_unit = this_brain_area[this_brain_area.unit == unit]
                map_unit = map_brain_area[map_brain_area.unit_id + 1 == unit]
                
                individual = this_unit.iloc[0]['individual']
                
                individual = individual_dict.get(individual)
                
                this_unit = this_unit.assign(individual = individual)
                
                all_individuals = all_individuals.append(this_unit)
        
    elif dataset == 'chandravadia':
        
        this_dataset = data[data.dataset == dataset]
        
        individuals = this_dataset.individual.unique()
        individual_key = range(len(individuals))
        
        zipped = zip(individuals,individual_key)
        
        for brain_area in this_dataset.brain_area.unique():
            
            if brain_area == 'amyg':
                
                this_brain_area = this_dataset[this_dataset.brain_area == brain_area]
                
                map_df = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/chadrandavia_amyg_map.csv',index_col=0)
                
                for unit in this_brain_area.unit.unique():
                
                    this_unit = this_brain_area[this_brain_area.unit == unit]
                    map_unit = map_df[map_df.unit_n +1 == unit]
                    
                    if len(map_unit) == 0:
                        
                        pass
                    
                    else:
                   
                        individual = this_unit.iloc[0]['individual']
                
                        individual = individual_dict.get(individual)
                        
                        this_unit = this_unit.assign(individual = individual)
                        
                        all_individuals = all_individuals.append(this_unit)
                    
            elif brain_area == 'hc':
                
                this_brain_area = this_dataset[this_dataset.brain_area == brain_area]
            
                map_df = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/chadrandavia_hc_map.csv',index_col=0)

                for unit in this_brain_area.unit.unique():
                
                    this_unit = this_brain_area[this_brain_area.unit == unit]
                    map_unit = map_df[map_df.unit_n +1 == unit]
                    
                    if len(map_unit) == 0:
                        
                        pass
                    
                    else:
                   
                        this_individual = map_unit.session.values[0]
                        
                        individual = this_unit.iloc[0]['individual']
                
                        individual = individual_dict.get(individual)
                
                        this_unit = this_unit.assign(individual = individual)
                        
                        all_individuals = all_individuals.append(this_unit)
                        
#%%

all_individuals.to_csv('all_individuals2.csv',index=False)
                    
# %%
