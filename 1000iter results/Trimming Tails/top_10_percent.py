#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

import seaborn as sns

plt.style.use('seaborn')

#%% Load in data, filter, plot before trimming

raw_data = pd.read_csv('F:/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

acc = data[(data.brain_area == 'acc') | (data.brain_area == 'dACC') | (data.brain_area == 'aca') | (data.brain_area == 'mcc')]

amyg = data[(data.brain_area == 'amygdala') | (data.brain_area == 'central') | (data.brain_area == 'bla')]

hc = data[(data.brain_area == 'hc') | (data.brain_area == 'ca1') | (data.brain_area == 'ca2') | (data.brain_area == 'ca3') | (data.brain_area == 'dg')]

mpfc = data[(data.brain_area == 'mpfc') | (data.brain_area == 'pl') | (data.brain_area == 'ila') | (data.brain_area == 'scACC')]

ofc = data[(data.brain_area == 'ofc') | (data.brain_area == 'orb')]
           
striatum = data[(data.brain_area == 'vStriatum') | (data.brain_area == 'putamen') | (data.brain_area == 'caudate')]

acc['brain_region'] = 'acc'
amyg['brain_region'] = 'amygdala'
hc['brain_region'] = 'hippocampus'
mpfc['brain_region'] = 'mpfc'
ofc['brain_region'] = 'ofc'
striatum['brain_region'] = 'striatum'

data = pd.concat((acc,amyg,hc,mpfc,ofc,striatum))

plt.figure(figsize=(8,6))

sns.catplot(data=data,x='species',y='tau',col='brain_region',col_wrap=3,ci='sd',kind='violin')

plt.suptitle('No trimming')

plt.show()

#%% Trim tails by 10% of successful iterations at top and bottomm

no_tails = []

for dataset in data.dataset.unique():
    
    this_dataset = data[data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            if len(this_unit) < 100:
                
                pass
            
            else:
                
                n = len(this_unit)
                
                head_n = int(0.9 * n)
                
                tail_n = int(0.1 * n)
                
                no_tail_unit = this_unit.iloc[tail_n:head_n]
            
                species = no_tail_unit.iloc[0]['species']
                
                brain_area = no_tail_unit.iloc[0]['brain_area']
                
                mean_tau = np.mean(no_tail_unit['tau'])
                
                sd_tau = np.std(no_tail_unit['tau'])
                
                mean_r2 = np.mean(no_tail_unit['r2'])
                
                sd_r2 = np.std(no_tail_unit['r2'])
                
                mean_fr = np.mean(no_tail_unit['fr'])
                
                sd_fr = np.std(no_tail_unit['fr'])
                
                try:
                
                    no_tails.append((dataset,species,brain_area,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                    
                except:
                    
                    pass
    
no_tails = pd.DataFrame(no_tails,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

no_tails['species'] = pd.Categorical(no_tails['species'], categories=listofspecies, ordered=True)

acc = no_tails[(no_tails.brain_area == 'acc') | (no_tails.brain_area == 'dACC') | (no_tails.brain_area == 'aca') | (no_tails.brain_area == 'mcc')]

amyg = no_tails[(no_tails.brain_area == 'amygdala') | (no_tails.brain_area == 'central') | (no_tails.brain_area == 'bla')]

hc = no_tails[(no_tails.brain_area == 'hc') | (no_tails.brain_area == 'ca1') | (no_tails.brain_area == 'ca2') | (no_tails.brain_area == 'ca3') | (no_tails.brain_area == 'dg')]

mpfc = no_tails[(no_tails.brain_area == 'mpfc') | (no_tails.brain_area == 'pl') | (no_tails.brain_area == 'ila') | (no_tails.brain_area == 'scACC')]

ofc = no_tails[(no_tails.brain_area == 'ofc') | (no_tails.brain_area == 'orb')]
           
striatum = no_tails[(no_tails.brain_area == 'vStriatum') | (no_tails.brain_area == 'putamen') | (no_tails.brain_area == 'caudate')]

acc['brain_region'] = 'acc'
amyg['brain_region'] = 'amygdala'
hc['brain_region'] = 'hippocampus'
mpfc['brain_region'] = 'mpfc'
ofc['brain_region'] = 'ofc'
striatum['brain_region'] = 'striatum'

no_tails = pd.concat((acc,amyg,hc,mpfc,ofc,striatum))

# Plot!

plt.figure(figsize=(8,6))

sns.catplot(data=no_tails,x='species',y='tau',col='brain_region',col_wrap=3,ci='sd',kind='violin')

plt.suptitle('Trimming top and bottom 10%')

plt.show()


#%% Repeat with 5%

no_tails = []

for dataset in data.dataset.unique():
    
    this_dataset = data[data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            if len(this_unit) < 100:
                
                pass
            
            else:
                
                n = len(this_unit)
                
                head_n = int(0.95 * n)
                
                tail_n = int(0.05 * n)
                
                no_tail_unit = this_unit.iloc[tail_n:head_n]
            
                species = no_tail_unit.iloc[0]['species']
                
                brain_area = no_tail_unit.iloc[0]['brain_area']
                
                mean_tau = np.mean(no_tail_unit['tau'])
                
                sd_tau = np.std(no_tail_unit['tau'])
                
                mean_r2 = np.mean(no_tail_unit['r2'])
                
                sd_r2 = np.std(no_tail_unit['r2'])
                
                mean_fr = np.mean(no_tail_unit['fr'])
                
                sd_fr = np.std(no_tail_unit['fr'])
                
                try:
                
                    no_tails.append((dataset,species,brain_area,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                    
                except:
                    
                    pass
    
no_tails = pd.DataFrame(no_tails,columns=['dataset','species','brain_area','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

no_tails['species'] = pd.Categorical(no_tails['species'], categories=listofspecies, ordered=True)

acc = no_tails[(no_tails.brain_area == 'acc') | (no_tails.brain_area == 'dACC') | (no_tails.brain_area == 'aca') | (no_tails.brain_area == 'mcc')]

amyg = no_tails[(no_tails.brain_area == 'amygdala') | (no_tails.brain_area == 'central') | (no_tails.brain_area == 'bla')]

hc = no_tails[(no_tails.brain_area == 'hc') | (no_tails.brain_area == 'ca1') | (no_tails.brain_area == 'ca2') | (no_tails.brain_area == 'ca3') | (no_tails.brain_area == 'dg')]

mpfc = no_tails[(no_tails.brain_area == 'mpfc') | (no_tails.brain_area == 'pl') | (no_tails.brain_area == 'ila') | (no_tails.brain_area == 'scACC')]

ofc = no_tails[(no_tails.brain_area == 'ofc') | (no_tails.brain_area == 'orb')]
           
striatum = no_tails[(no_tails.brain_area == 'vStriatum') | (no_tails.brain_area == 'putamen') | (no_tails.brain_area == 'caudate')]

acc['brain_region'] = 'acc'
amyg['brain_region'] = 'amygdala'
hc['brain_region'] = 'hippocampus'
mpfc['brain_region'] = 'mpfc'
ofc['brain_region'] = 'ofc'
striatum['brain_region'] = 'striatum'

no_tails = pd.concat((acc,amyg,hc,mpfc,ofc,striatum))

# Plot!

plt.figure(figsize=(8,6))

sns.catplot(data=no_tails,x='species',y='tau',col='brain_region',col_wrap=3,ci='sd',kind='violin')

plt.suptitle('Trimming top and bottom 10%')

plt.show()

# %%
