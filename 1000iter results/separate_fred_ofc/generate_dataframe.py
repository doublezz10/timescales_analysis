#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from scipy.io import loadmat

plt.style.use('seaborn')

#%% Load in data, filter

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data_with_lai_vl.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

stoll_amyg = fred_data[fred_data.brain_area == 'amygdala']

stoll_amyg = stoll_amyg[stoll_amyg.dataset=='stoll']

stoll_ofc = fred_data[(fred_data.brain_area == 'ofc') | (fred_data.brain_area == 'orb')]

stoll_ofc = stoll_ofc[stoll_ofc.dataset=='stoll']

stoll_lai = fred_data[fred_data.brain_area == 'LAI']

stoll_vl = fred_data[fred_data.brain_area == 'vlPFC']

#%% Import cell_info to get which Brodmann's area
#   Put together matching units along with which area

ofc_details = []
amyg_details = []
lai_details = []
vl_details = []

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
    
mat = loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Stoll - monkey1/stoll_vlPFC.mat',simplify_cells=True)
vl_cell_info = mat['cell_info']

for unit in range(len(vl_cell_info)):
    
    unit_id = unit + 1
    
    depth = vl_cell_info[unit]['depth']
    
    area = vl_cell_info[unit]['area']
    
    vl_details.append((unit_id,'vlPFC',area,depth))
    
ofc_details = pd.DataFrame(ofc_details,columns=['unit','brain_region','specific_area','depth'])
amyg_details = pd.DataFrame(amyg_details,columns=['unit','brain_region','specific_area','depth'])
lai_details = pd.DataFrame(lai_details,columns=['unit','brain_region','specific_area','depth'])
vl_details = pd.DataFrame(vl_details,columns=['unit','brain_region','specific_area','depth'])

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

all_depths = []
all_areas = []

for unit in range(len(stoll_vl)):
    
    this_unit = stoll_vl.iloc[unit]
    
    unit_id = this_unit.unit
    
    these_details = vl_details[vl_details.unit==unit_id]
    
    this_depth = these_details.iloc[0]['depth']
    this_area = these_details.iloc[0]['specific_area']
    
    all_depths.append(this_depth)
    all_areas.append(this_area)
    
stoll_vl['depth'] = all_depths
stoll_vl['specific_area'] = all_areas

# %%

stoll_ofc.to_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/fred_ofc_isi.csv')

stoll_amyg.to_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/fred_amyg_isi.csv')

stoll_lai.to_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/fred_lai_isi.csv')

stoll_vl.to_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/fred_vl_isi.csv')
# %%
