#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from scipy.io import loadmat

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

data = data[data.species != 'rat']

listofspecies = ['mouse','monkey','human']

data['species'] = pd.Categorical(data['species'], categories = listofspecies , ordered = True)

#%% Get mean values over all iterations

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
            
            # fixed python numbering here - first unit is index 1 not 0
            
            all_means.append((dataset,species,brain_area,unit_n + 1,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
    
all_means = pd.DataFrame(all_means,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

all_means['species'] = pd.Categorical(all_means['species'], categories=listofspecies, ordered=True)

del dataset, brain_area, this_dataset, these_data, this_unit, species, unit_n
del mean_tau, sd_tau, mean_r2, sd_r2, mean_fr, sd_fr, n

#%% Load in Fred's data

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')
fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

# rename columns to match

fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus','hc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('mPFC','mpfc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('ventralStriatum','vStriatum')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('Cd','caudate')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('OFC','ofc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('PUT','putamen')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus2','hc2')

fred_data = fred_data.replace(['amyg','AMG'],'amygdala')

fred_data['dataset'] = fred_data['dataset'].str.replace('stein','steinmetz')

fred_data = fred_data[fred_data.r2 >= 0.5]

fred_data = fred_data[(fred_data.tau >=10) & (fred_data.tau <= 1000)]

all_means['brain_area'] = all_means['brain_area'].str.replace('hippocampus','hc')

#%% Loop through and pull out matching taus

matching_units = []

for dataset in all_means.dataset.unique():
    
    this_dataset = all_means[all_means.dataset == dataset]
    
    fred_dataset = fred_data[fred_data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]
        
        fred_these_data = fred_dataset[fred_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            fred_unit = fred_these_data[fred_these_data.unit == unit_n]
            
            species = this_unit.iloc[0]['species']
            
            if len(fred_unit) == 0:
                
                pass
            
            elif fred_unit['keep'].values[0] == 0:
                
                pass
            
            elif fred_unit['r2'].values[0] <= 0.5:
                
                pass
            
            else:
                
                zach_tau = this_unit['tau'].values[0]
                zach_tau_sd = this_unit['sd_tau'].values[0]
                zach_fr = this_unit['mean_fr'].values[0]
                zach_n = this_unit['n'].values[0]
                zach_r2 = this_unit['mean_r2'].values[0]
                
                fred_tau = fred_unit['tau'].values[0]
                fred_lat = fred_unit['lat'].values[0]
                fred_fr = fred_unit['FR'].values[0]
                fred_r2 = fred_unit['r2'].values[0]
                
                tau_diff = zach_tau - fred_tau
                
                if dataset == 'meg':
                    
                    dataset = 'young/mosher'
                
                matching_units.append((dataset,species,brain_area,unit_n,zach_tau,zach_tau_sd,zach_fr,zach_n,zach_r2,fred_tau,fred_lat,fred_fr,fred_r2,tau_diff))
            
matching_units = pd.DataFrame(matching_units,columns=['dataset','species','brain_area','unit','zach_tau','zach_tau_sd','zach_fr','zach_n','zach_r2','fred_tau','fred_lat','fred_fr','fred_r2','tau_diff'])

listofspecies = ['mouse','rat','monkey','human']

matching_units['species'] = pd.Categorical(matching_units['species'], categories = listofspecies , ordered = True)

stoll_amyg = matching_units[matching_units.brain_area == 'amygdala']

stoll_amyg = stoll_amyg[stoll_amyg.dataset=='stoll']

stoll_ofc = matching_units[(matching_units.brain_area == 'ofc') | (matching_units.brain_area == 'orb')]

stoll_ofc = stoll_ofc[stoll_ofc.dataset=='stoll']

stoll_lai = matching_units[matching_units.brain_area == 'LAI']

#%% Import cell_info to get which Brodmann's area
#   Put together matching units along with which area

ofc_details = []
amyg_details = []
lai_details = []

mat = loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Stoll - monkey1/stoll_OFC.mat',simplify_cells=True)
ofc_cell_info = mat['cell_info']

for unit in range(len(ofc_cell_info)):
    
    unit_id = unit + 1
    
    depth = ofc_cell_info[unit]['depth']
    
    area = ofc_cell_info[unit]['area']
    
    ofc_details.append((unit_id,'ofc',area,depth))

mat = loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Stoll - monkey1/stoll_AMG.mat',simplify_cells=True)
amyg_cell_info = mat['cell_info']

for unit in range(len(amyg_cell_info)):
    
    unit_id = unit + 1
    
    depth = amyg_cell_info[unit]['depth']
    
    area = amyg_cell_info[unit]['area']
    
    amyg_details.append((unit_id,'amygdala',area,depth))
    
mat = loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Stoll - monkey1/stoll_LAI.mat',simplify_cells=True)
lai_cell_info = mat['cell_info']

for unit in range(len(lai_cell_info)):
    
    unit_id = unit + 1
    
    depth = lai_cell_info[unit]['depth']
    
    area = lai_cell_info[unit]['area']
    
    lai_details.append((unit_id,'LAI',area,depth))
    
ofc_details = pd.DataFrame(ofc_details,columns=['unit','brain_region','specific_area','depth'])
amyg_details = pd.DataFrame(amyg_details,columns=['unit','brain_region','specific_area','depth'])
lai_details = pd.DataFrame(lai_details,columns=['unit','brain_region','specific_area','depth'])

# %% add depth and specific area columns to dataframes

all_depths = []
all_areas = []

for unit in range(len(stoll_ofc)):
    
    this_unit = stoll_ofc.iloc[unit]
    
    unit_id = this_unit.unit
    
    these_details = ofc_details[ofc_details.unit==unit_id]
    
    this_depth = these_details.iloc[0]['depth']
    this_area = these_details.iloc[0]['specific_area']
    
    all_depths.append(this_depth)
    all_areas.append(this_area)
    
stoll_ofc['depth'] = all_depths
stoll_ofc['specific_area'] = all_areas

all_depths = []
all_areas = []

for unit in range(len(stoll_amyg)):
    
    this_unit = stoll_amyg.iloc[unit]
    
    unit_id = this_unit.unit
    
    these_details = amyg_details[amyg_details.unit==unit_id]
    
    this_depth = these_details.iloc[0]['depth']
    this_area = these_details.iloc[0]['specific_area']
    
    all_depths.append(this_depth)
    all_areas.append(this_area)
    
stoll_amyg['depth'] = all_depths

all_depths = []
all_areas = []

for unit in range(len(stoll_lai)):
    
    this_unit = stoll_lai.iloc[unit]
    
    unit_id = this_unit.unit
    
    these_details = lai_details[lai_details.unit==unit_id]
    
    this_depth = these_details.iloc[0]['depth']
    this_area = these_details.iloc[0]['specific_area']
    
    all_depths.append(this_depth)
    all_areas.append('LAI')
    
stoll_lai['depth'] = all_depths
stoll_lai['specific_area'] = all_areas

# %%

stoll_ofc.to_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/separate_fred_ofc/fred_ofc.csv')

stoll_amyg.to_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/separate_fred_ofc/fred_amyg.csv')

stoll_lai.to_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/separate_fred_ofc/fred_lai.csv')

# %%
