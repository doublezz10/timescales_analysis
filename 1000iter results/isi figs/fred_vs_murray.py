#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress, pearsonr

plt.rcParams['font.size'] = '7'

plt.style.use('seaborn')

old_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/results.csv')

old_data = old_data[old_data.tau < 1000]
old_data = old_data[old_data.r2 > 0.5]


#%% Load in Fred's data

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

fred_data['unit'] = fred_data['unit'] - 1

#%%

matching_units = []

for dataset in old_data.dataset.unique():
    
    this_dataset = old_data[old_data.dataset == dataset]
    
    fred_dataset = fred_data[fred_data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]
        
        fred_these_data = fred_dataset[fred_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n ]
            
            fred_unit = fred_these_data[fred_these_data.unit == unit_n]
            
            if len(this_unit)==0:
                
                pass
            
            elif len(fred_unit) == 0:
                
                pass
            
            elif fred_unit['keep'].values[0] == 0:
                
                pass
            
            elif fred_unit['r2'].values[0] <= 0.5:
                
                pass
            
            else:
                
                species = this_unit.iloc[0]['species']
                
                zach_tau = this_unit['tau'].values[0]
                zach_fr = this_unit['fr'].values[0]
                zach_r2 = this_unit['r2'].values[0]
                
                fred_tau = fred_unit['tau'].values[0]
                fred_fr = fred_unit['FR'].values[0]
                fred_r2 = fred_unit['r2'].values[0]
                
                tau_diff = zach_tau - fred_tau
                
                if dataset == 'meg':
                    
                    dataset = 'young/mosher'
                
                matching_units.append((dataset,species,brain_area,unit_n,zach_tau,zach_fr,zach_r2,fred_tau,fred_fr,fred_r2,tau_diff))
            
matching_units = pd.DataFrame(matching_units,columns=['dataset','species','brain_area','unit','murray_tau','murray_fr','murray_r2','fred_tau','fred_fr','fred_r2','tau_diff'])

listofspecies = ['mouse','monkey','human']

matching_units['species'] = pd.Categorical(matching_units['species'], categories = listofspecies , ordered = True)

#%% Separate by brain area

acc = matching_units[(matching_units.brain_area == 'acc') | (matching_units.brain_area == 'dACC') | (matching_units.brain_area == 'aca') | (matching_units.brain_area == 'mcc')]

amyg = matching_units[(matching_units.brain_area == 'amygdala') | (matching_units.brain_area == 'central') | (matching_units.brain_area == 'bla')]

hc = matching_units[(matching_units.brain_area == 'hc') | (matching_units.brain_area == 'ca1') | (matching_units.brain_area == 'ca2') | (matching_units.brain_area == 'ca3') | (matching_units.brain_area == 'dg')]

mpfc = matching_units[(matching_units.brain_area == 'mpfc') | (matching_units.brain_area == 'pl') | (matching_units.brain_area == 'ila') | (matching_units.brain_area == 'scACC')]

ofc = matching_units[(matching_units.brain_area == 'ofc') | (matching_units.brain_area == 'orb')]

# New dataframe with all brain regions stacked

acc2 = acc.assign(brain_region='ACC')
amyg2 = amyg.assign(brain_region='Amygdala')
hc2 = hc.assign(brain_region='Hippocampus')
mpfc2 = mpfc.assign(brain_region='mPFC')
ofc2 = ofc.assign(brain_region='OFC')

brain_region_data = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2))

# %%

corrs = []

for species in listofspecies:
    
    this_species = brain_region_data[brain_region_data.species==species]

    for region in this_species.brain_region.unique():
        
        fred_taus = this_species[this_species.brain_region == region]['fred_tau']
        murray_taus = this_species[this_species.brain_region == region]['murray_tau']
        
        corr = np.corrcoef(fred_taus,murray_taus)[0,1]
        
        p = pearsonr(fred_taus, murray_taus)[1]
        
        corrs.append((species,region,corr,p))
        
        
corrs = pd.DataFrame(corrs,columns=['species','area','corr','pval'])

corrs_ = corrs.pivot('species','area','corr')

pvals = corrs.pivot('species','area','pval')

#%%
        
plt.figure(figsize=(3,3))

ax = sns.heatmap(corrs_,vmin=0,vmax=1,annot = pvals)

cbar = ax.collections[0].colorbar

cbar.ax.tick_params(labelsize=7)
cbar.ax.set_ylabel('correlation',size=7)

plt.tick_params(axis='x',rotation=45,labelsize=7)
plt.tick_params(axis='y',labelsize=7)

plt.xlabel('')
plt.ylabel('')

plt.show()
#%%

sns.lmplot(data=brain_region_data[brain_region_data.brain_region=='Amygdala'],x='fred_tau',y='murray_tau',hue='species',legend=False,scatter_kws={'s':6,'alpha':0.7},height=3.2,aspect=1)

plt.xlabel('ISI timescale (ms)',fontsize=7)
plt.ylabel('One-fit timescale (ms)',fontsize=7)
plt.tick_params(axis='x',labelsize=7)
plt.tick_params(axis='y',labelsize=7)
plt.legend(prop={'size': 7})

plt.xlim(0,1000)
plt.ylim(0,1000)

plt.show()
# %%
