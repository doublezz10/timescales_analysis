#%%
 
import numpy as np
import pandas as pd

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')

fred_data = fred_data[fred_data.species != 'rat']

fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

# rename columns to match

fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus','hc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('mPFC','mpfc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('ventralStriatum','vStriatum')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('AMG','amygdala')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('Cd','caudate')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('OFC','ofc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('PUT','putamen')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus2','hc2')

fred_data['dataset'] = fred_data['dataset'].str.replace('stein','steinmetz')

fred_data = fred_data[fred_data.r2 >= 0.5]

fred_data = fred_data[(fred_data.tau >=10) & (fred_data.tau <= 1000)]

data = fred_data[fred_data.keep == 1]

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
