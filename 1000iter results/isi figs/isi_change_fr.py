#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress, pearsonr

plt.rcParams['font.size'] = '7'

plt.style.use('seaborn')

#%% Load in data, filter

corrs = []

for fr in [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5]:

    raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

    data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

    data = data[(data.r2 >= 0.5)]
    
    data = data[data.fr >= fr]

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

    #%% Load in Fred's data

    fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')
    fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
    fred_data = fred_data[fred_data.dataset != 'faraut']

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
    
    fred_data = fred_data[fred_data.FR >= fr]

    fred_data = fred_data[(fred_data.tau >=10) & (fred_data.tau <= 1000)]

    all_means['brain_area'] = all_means['brain_area'].str.replace('hippocampus','hc')
    
    # Loop through and pull out matching taus

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
                    fred_fr = fred_unit['FR'].values[0]
                    fred_r2 = fred_unit['r2'].values[0]
                    
                    tau_diff = zach_tau - fred_tau
                    
                    if dataset == 'meg':
                        
                        dataset = 'young/mosher'
                    
                    matching_units.append((dataset,species,brain_area,unit_n,zach_tau,zach_tau_sd,zach_fr,zach_n,zach_r2,fred_tau,fred_fr,fred_r2,tau_diff))
                
    matching_units = pd.DataFrame(matching_units,columns=['dataset','species','brain_area','unit','zach_tau','zach_tau_sd','zach_fr','zach_n','zach_r2','fred_tau','fred_fr','fred_r2','tau_diff'])

    listofspecies = ['mouse','monkey','human']

    matching_units['species'] = pd.Categorical(matching_units['species'], categories = listofspecies , ordered = True)


    # Separate by brain area

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

    #

    for species in listofspecies:
        
        this_species = brain_region_data[brain_region_data.species==species]

        for region in this_species.brain_region.unique():
            
            fred_taus = this_species[this_species.brain_region == region]['fred_tau']
            zach_taus = this_species[this_species.brain_region == region]['zach_tau']
            
            corr = np.corrcoef(fred_taus,zach_taus)[0,1]
            p = pearsonr(fred_taus,zach_taus)[1]
            
            n = len(fred_taus)
            
            corrs.append((species,region,corr,p,fr,n,np.mean(fred_taus),np.mean(zach_taus)))
            
corrs = pd.DataFrame(corrs,columns=['species','brain_region','corr','pval','fr','n','mean_f','mean_z'])

# %%

fig,ax1 = plt.subplots()

mouse_amyg = corrs[corrs.brain_region=='Amygdala']
mouse_amyg = mouse_amyg[mouse_amyg.species=='mouse']

ax1 = sns.scatterplot(data=mouse_amyg,x='fr',y='n',color='green')

ax1.set_xlabel('FR thresh (hz)')

ax2 = ax1.twinx()
ax2 = sns.scatterplot(data=mouse_amyg,x='fr',y='corr')

mean_corr = np.mean(corrs[corrs.fr==1]['corr'])
plt.axhline(mean_corr,color='black',linestyle='dashed',alpha=0.4)

plt.grid(False)

plt.show()

# %%

sns.scatterplot(data=mouse_amyg,x='fr',y='mean_f',color='blue',label='isi')
sns.scatterplot(data=mouse_amyg,x='fr',y='mean_z',color='green',label='item')

# mean_taus = corrs[corrs.fr==1].groupby('species',as_index=False).mean()

# plt.axhline(mean_taus.iloc[0].mean_f,label='human mean - ISI',color='red')
# plt.axhline(mean_taus.iloc[1].mean_f,label='monkey mean - ISI',color='purple')

# plt.axhline(mean_taus.iloc[0].mean_z,label='human mean - ITEM',color='red',linestyle='dashed')
# plt.axhline(mean_taus.iloc[1].mean_z,label='monkey mean - ITEM',color='purple',linestyle='dashed')

plt.legend(bbox_to_anchor=(1.05, 1.0))

plt.xlabel('FR thresh (hz)')

plt.ylabel('mean tau')

plt.show()

#%%