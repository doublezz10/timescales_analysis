#%% Imports, get filenames
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io as spio

import bottleneck as bn

plt.style.use('seaborn')

directory = '/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/Spiketimes only/'
 
datasets = []

for root, dirs, files in os.walk(directory):
    
    for filename in files:
        
        if filename.startswith('ISI'):
            pass
        
        elif filename.startswith('stoll2'):
            pass
        
        elif filename.endswith('.mat'):
            
            datasets.append(os.path.join(root, filename))
            
del directory, dirs, files, filename, root

#%% for each dataset, calculate stuff about spiketimes

all_spike_stats = []
     
for ds in range(len(datasets)):
            
    mat = spio.loadmat(datasets[ds],simplify_cells=True, squeeze_me=True)

    spikes = mat['spikes']
    cell_info = mat['cell_info']
    
    if type(cell_info) == list:
        
        cell_info = cell_info[0]
        
    elif type(cell_info) == np.ndarray:
        
        cell_info = {'dataset': cell_info[0][0],'species': cell_info[0][1],'brain_area':cell_info[0][2]}
    
    for cell in range(len(spikes)):
        
        spiketimes = spikes[cell]
        
        if len(spiketimes) == 0:
            
            pass
        
        else:
        
            # 0 align for ease :)
            
            spiketimes = spiketimes - np.min(spiketimes)
            
            n_spikes = len(spiketimes)
            
            # mean_fr over whole timeseries
            
            mean_fr = len(spiketimes)/np.max(spiketimes)
            
            # mean and var of 1sec sliding windows - z-scored
            
            bins = np.arange(0,np.max(spiketimes),step=0.5) # bin at 500ms
            binned_spikes, edges = np.histogram(spiketimes,bins=bins)
            
            slide = bn.move_sum(binned_spikes, window=2) # sliding window over 1sec (2 windows)
            
            z_slide = slide/mean_fr
            
            mean_z = np.mean(z_slide[1:])
            
            var_z = np.var(z_slide[1:])
            
            # fano factor
            
            fano = var_z / mean_z
            
            # burstiness - from Ko et al J Neurosci Methods 2012
            
            isis = np.diff(spiketimes)
            
            log_isi = np.log10(isis)
            
            norm_log_isi = log_isi - np.mean(log_isi)
            
            burst_thresh = np.percentile(norm_log_isi,99.5)
            pause_thresh = np.percentile(norm_log_isi,0.5)   
            
            prop_burst = sum(1 for i in norm_log_isi if i > burst_thresh)/len(norm_log_isi)
            prop_pause = sum(1 for i in norm_log_isi if i < pause_thresh)/len(norm_log_isi)
            
            # assemble
            
            try:    
            
                brain_area = cell_info['brain_area']
                
            except KeyError:
                
                try:
                
                    brain_area = cell_info['Brain_area']
                
                except KeyError:
                
                    brain_area = cell_info['Brain_Area']
            
            if brain_area == 'MCC':
                
                dataset = 'Fontanier'
                species = 'monkey'
                
            else:
                
                try:
            
                    dataset = cell_info['dataset']
                    species = cell_info['species']
                
                except KeyError:
                    
                    dataset = cell_info['Dataset']
                    species = cell_info['Species']

            all_spike_stats.append((dataset,species,brain_area,cell,n_spikes,mean_fr,mean_z,var_z,fano,prop_burst,prop_pause))

# combine into df  
      
all_spike_stats = pd.DataFrame(all_spike_stats,columns=['dataset','species','brain_area','unit','n_spikes','mean_fr','mean_norm_fr','var_norm_fr','fano','prop_burst','prop_pause'])


#%% Clean up

# areas we don't care about

all_spike_stats = all_spike_stats[all_spike_stats.brain_area != 'IFG']
all_spike_stats = all_spike_stats[all_spike_stats.brain_area != 'dlPFC']
all_spike_stats = all_spike_stats[all_spike_stats.brain_area != 'PMd']
all_spike_stats = all_spike_stats[all_spike_stats.brain_area != 'caudate']
all_spike_stats = all_spike_stats[all_spike_stats.brain_area != 'putamen']
all_spike_stats = all_spike_stats[all_spike_stats.brain_area != 'preSMA']
all_spike_stats = all_spike_stats[all_spike_stats.brain_area != 'ventralStriatum']

all_spike_stats = all_spike_stats.replace('Macaque','monkey')

all_spike_stats.loc[all_spike_stats['dataset']=='Wirth2','brain_area'] = 'hc2'

all_spike_stats['dataset'] = all_spike_stats['dataset'].str.replace('Wirth2','wirth')

# rename areas to match other dataframes

all_spike_stats['brain_area'] = all_spike_stats['brain_area'].str.replace('MCC','mcc')
all_spike_stats['brain_area'] = all_spike_stats['brain_area'].str.replace('dorsal ACC','dACC')
all_spike_stats['brain_area'] = all_spike_stats['brain_area'].str.replace('OFC','ofc')
all_spike_stats['brain_area'] = all_spike_stats['brain_area'].str.replace('hippocampus','hc')



acc = all_spike_stats[(all_spike_stats.brain_area=='MCC') | (all_spike_stats.brain_area=='aca') | (all_spike_stats.brain_area=='dACC')]
amyg = all_spike_stats[(all_spike_stats.brain_area=='bla') | (all_spike_stats.brain_area=='amygdala')]
hc = all_spike_stats[(all_spike_stats.brain_area=='hc') | (all_spike_stats.brain_area=='ca1') | (all_spike_stats.brain_area=='ca2') | (all_spike_stats.brain_area=='ca3') | (all_spike_stats.brain_area=='dg')]
mpfc = all_spike_stats[(all_spike_stats.brain_area=='pl') | (all_spike_stats.brain_area=='ila') | (all_spike_stats.brain_area=='scACC') | (all_spike_stats.brain_area=='mPFC')]
ofc = all_spike_stats[(all_spike_stats.brain_area=='ofc') | (all_spike_stats.brain_area=='orb')]
vlpfc = all_spike_stats[(all_spike_stats.brain_area=='vlPFC')]
ai = all_spike_stats[(all_spike_stats.brain_area=='LAI')]

acc = acc.assign(brain_region='ACC')
amyg = amyg.assign(brain_region='Amygdala')
hc = hc.assign(brain_region='Hippocampus')
mpfc = mpfc.assign(brain_region='mPFC')
ofc = ofc.assign(brain_region='OFC')
vlpfc = vlpfc.assign(brain_region='vlPFC')
ai = ai.assign(brain_region='AI')

brain_regions = pd.concat((acc,amyg,hc,mpfc,ofc,vlpfc,ai))

brain_regions['dataset'] = brain_regions['dataset'].str.lower()

#%% load in freds data

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')
fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})

fred_data = fred_data[fred_data.species != 'rat']
fred_data = fred_data[fred_data.dataset != 'faraut']
fred_data = fred_data[fred_data.dataset != 'froot']

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

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

fred_data = fred_data[fred_data.keep == 1]

#%%

new_df = pd.DataFrame()

for dataset in brain_regions.dataset.unique():
    
    this_dataset = brain_regions[brain_regions.dataset==dataset]
    fred_dataset = fred_data[fred_data.dataset==dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        this_area = this_dataset[this_dataset.brain_area==brain_area]
        fred_area = fred_dataset[fred_dataset.brain_area==brain_area]

        for unit in this_area.unit.unique():
            
            this_unit = this_area[this_area.unit==unit]
            fred_unit = fred_area[fred_area.unit==unit]
            
            try:
            
                tau = fred_unit['tau'].values[0]
                lat = fred_unit['lat'].values[0]
                r2 = fred_unit['r2'].values[0]
                
                this_unit = this_unit.assign(tau=tau)
                this_unit = this_unit.assign(lat=lat)
                this_unit = this_unit.assign(r2=r2)
                
                new_df = pd.concat((new_df,this_unit),ignore_index=True)
                
            except:
                
                pass

# %%

new_df.to_csv('spike_stats_1percent.csv')
# %%
