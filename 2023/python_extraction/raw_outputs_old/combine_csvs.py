#%%

import pandas as pd
import numpy as np

set1 = pd.read_csv('python_timescales_dataset1.csv')
set2 = pd.read_csv('python_timescales_dataset2.csv')
set3 = pd.read_csv('python_timescales_dataset3.csv')
set4 = pd.read_csv('python_timescales_dataset4.csv')
set5 = pd.read_csv('python_timescales_dataset5.csv')
set6 = pd.read_csv('python_timescales_dataset6.csv')

keep6 = pd.DataFrame()

for dataset in set6.dataset.unique():
    
    this_dataset = set6[set6.dataset==dataset]
    
    for unit in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit==unit]
        
        keep = this_unit[this_unit.fit_err==np.nanmin(this_unit.fit_err)].iloc[0]
        
        keep6 = pd.concat((keep6,keep),axis=1)
        
keep6 = keep6.T

set6 = keep6

#%%

combined = pd.concat((set1,set2,set3,set4,set5,set6))
data = combined.drop(["Unnamed: 0"],axis=1)

combined.to_csv('combined_python_timescales.csv',index=False)
# %%

# process data

mouse = data[data.dataset=='stein']
monkey = data[(data.dataset=='fontanier') | (data.dataset=='meg') | (data.dataset=='stoll') | (data.dataset=='stoll2') | (data.dataset=='wirth') | (data.dataset=='wirth2')]
human = data[(data.dataset=='minxha') | (data.dataset=='chandravadia')]

mouse_g = mouse.assign(species='mouse')
monkey_g = monkey.assign(species='monkey')
human_g = human.assign(species='human')

data = pd.concat((mouse_g,monkey_g,human_g))

amyg = data[(data.area=='amygdala') | (data.area=='bla')]
hc = data[(data.area=='hippocampus') | (data.area=='dg') | (data.area=='ca1') | (data.area=='ca2') | (data.area=='ca3')]
acc = data[(data.area=='mcc') | (data.area=='dACC') | (data.area=='aca')]
mpfc = data[(data.area=='scACC') | (data.area=='ila') | (data.area=='pl')]
ofc = data[(data.area=='OFC') | (data.area=='orb')]

acc_g = acc.assign(brain_region = 'ACC')
amyg_g = amyg.assign(brain_region = 'Amygdala')
hc_g = hc.assign(brain_region = 'Hippocampus')
mpfc_g = mpfc.assign(brain_region = 'mPFC')
ofc_g = ofc.assign(brain_region = 'OFC')

grouped_data = pd.concat((acc_g,amyg_g,hc_g,mpfc_g,ofc_g))

brain_regions = ['Hippocampus','Amygdala','OFC','mPFC','ACC']

grouped_data['brain_region'] = pd.Categorical(grouped_data['brain_region'], categories = brain_regions , ordered = True)

grouped_data['r2'] = grouped_data['r2'] * -1

grouped_data.to_csv('processed_data.csv',index=False)
# %%
