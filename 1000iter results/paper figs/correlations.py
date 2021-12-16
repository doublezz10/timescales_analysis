#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import ptitprince as pt

import seaborn as sns

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

data = data[data.species != 'rat']

data['species'] = pd.Categorical(data['species'], categories = listofspecies , ordered = True)

print('Proportion of units surviving filtering:', len(data)/len(raw_data))

#%% Separate by brain_region

# raw data

raw_acc = raw_data[(raw_data.brain_area == 'acc') | (raw_data.brain_area == 'dACC') | (raw_data.brain_area == 'aca') | (raw_data.brain_area == 'mcc')]

raw_amyg = raw_data[(raw_data.brain_area == 'amygdala') | (raw_data.brain_area == 'central') | (raw_data.brain_area == 'bla')]

raw_hc = raw_data[(raw_data.brain_area == 'hc') | (raw_data.brain_area == 'hc2') | (raw_data.brain_area == 'ca1') | (raw_data.brain_area == 'ca2') | (raw_data.brain_area == 'ca3') | (raw_data.brain_area == 'dg')]

raw_mpfc = raw_data[(raw_data.brain_area == 'mpfc') | (raw_data.brain_area == 'pl') | (raw_data.brain_area == 'ila') | (raw_data.brain_area == 'scACC')]

raw_ofc = raw_data[(raw_data.brain_area == 'ofc') | (raw_data.brain_area == 'orb')]
           
raw_striatum = raw_data[(raw_data.brain_area == 'vStriatum') | (raw_data.brain_area == 'putamen') | (raw_data.brain_area == 'caudate')]

# filtered data

acc = data[(data.brain_area == 'acc') | (data.brain_area == 'dACC') | (data.brain_area == 'aca') | (data.brain_area == 'mcc')]

amyg = data[(data.brain_area == 'amygdala') | (data.brain_area == 'central') | (data.brain_area == 'bla')]

hc = data[(data.brain_area == 'hc') | (data.brain_area == 'ca1') | (data.brain_area == 'ca2') | (data.brain_area == 'ca3') | (data.brain_area == 'dg')]

mpfc = data[(data.brain_area == 'mpfc') | (data.brain_area == 'pl') | (data.brain_area == 'ila') | (data.brain_area == 'scACC')]

ofc = data[(data.brain_area == 'ofc') | (data.brain_area == 'orb')]
           
striatum = data[(data.brain_area == 'vStriatum') | (data.brain_area == 'putamen') | (data.brain_area == 'caudate')]

brain_regions = pd.concat((acc,amyg,hc,mpfc,ofc,striatum))

#%% Plot of surviving iterations by brain area

prop_surviving = [len(acc)/len(raw_acc),len(amyg)/len(raw_amyg),len(hc)/len(raw_hc),len(mpfc)/len(raw_mpfc),len(ofc)/len(raw_ofc),len(striatum)/len(raw_striatum)]

plt.bar(range(6),prop_surviving)
plt.xticks(range(6),['ACC','amygdala','hippocampus','mPFC','OFC','striatum'])
plt.xlabel('brain_area')
plt.ylabel('prop surviving')
plt.title('$R^2 > 0.5$ and $10 < tau < 1000$')
plt.show()

#%% Get mean values over all iterations

all_means = []

for dataset in data.dataset.unique():
    
    this_dataset = data[data.dataset == dataset]
    
    for brain_area in this_dataset.brain_area.unique():
        
        these_data = this_dataset[this_dataset.brain_area == brain_area]

        for unit_n in these_data.unit.unique():
    
            this_unit = these_data[these_data.unit == unit_n]
            
            if len(this_unit) < 100:
                
                pass
            
            else:
            
                species = this_unit.iloc[0]['species']
                
                mean_tau = np.mean(this_unit['tau'])
                
                sd_tau = np.std(this_unit['tau'])
                
                mean_r2 = np.mean(this_unit['r2'])
                
                sd_r2 = np.std(this_unit['r2'])
                
                mean_fr = np.mean(this_unit['fr'])
                
                sd_fr = np.std(this_unit['fr'])
                
                n = len(this_unit)
                
                try:
                
                    all_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                    
                except:
                    
                    pass
    
all_means = pd.DataFrame(all_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

all_means['species'] = pd.Categorical(all_means['species'], categories=listofspecies, ordered=True)

#%% ACC

acc_means = []

for dataset in acc.dataset.unique():
    
    these_data = acc[acc.dataset == dataset]

    for unit_n in these_data.unit.unique():

        this_unit = these_data[these_data.unit == unit_n]
        
        if len(this_unit) < 100:
                
                pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                acc_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass
                
acc_means = pd.DataFrame(acc_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

acc_means['species'] = pd.Categorical(acc_means['species'], categories=listofspecies, ordered=True)

# Amygdala

amyg_means = []

for dataset in amyg.dataset.unique():
    
    these_data = amyg[amyg.dataset == dataset]

    for unit_n in these_data.unit.unique():

        this_unit = these_data[these_data.unit == unit_n]
        
        if len(this_unit) < 100:
                
            pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            amyg_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))

amyg_means = pd.DataFrame(amyg_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

amyg_means['species'] = pd.Categorical(amyg_means['species'], categories=listofspecies, ordered=True)

# Hippocampus

hc_means = []

for dataset in hc.dataset.unique():
    
    these_data = hc[hc.dataset == dataset]

    for unit_n in these_data.unit.unique():
        
        this_unit = these_data[these_data.unit == unit_n]

        if len(this_unit) < 100:
                
                pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                hc_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass

hc_means = pd.DataFrame(hc_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

hc_means['species'] = pd.Categorical(hc_means['species'], categories=listofspecies, ordered=True)

# mPFC

mpfc_means = []

for dataset in mpfc.dataset.unique():
    
    these_data = mpfc[mpfc.dataset == dataset]

    for unit_n in these_data.unit.unique():
        
        this_unit = these_data[these_data.unit == unit_n]

        if len(this_unit) < 100:
                
            pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                mpfc_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass

mpfc_means = pd.DataFrame(mpfc_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

mpfc_means['species'] = pd.Categorical(mpfc_means['species'], categories=listofspecies, ordered=True)

# OFC

ofc_means = []

for dataset in ofc.dataset.unique():
    
    these_data = ofc[ofc.dataset == dataset]

    for unit_n in these_data.unit.unique():
        
        this_unit = these_data[these_data.unit == unit_n]
        
        if len(this_unit) < 100:
                
            pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                ofc_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass

ofc_means = pd.DataFrame(ofc_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

ofc_means['species'] = pd.Categorical(ofc_means['species'], categories=listofspecies, ordered=True)

# Striatum

striatum_means = []

for dataset in striatum.dataset.unique():
    
    these_data = striatum[striatum.dataset == dataset]

    for unit_n in these_data.unit.unique():
        
        this_unit = these_data[these_data.unit == unit_n]

        if len(this_unit) < 100:
                
            pass
            
        else:
            
            species = this_unit.iloc[0]['species']
            
            mean_tau = np.mean(this_unit['tau'])
            
            sd_tau = np.std(this_unit['tau'])
            
            mean_r2 = np.mean(this_unit['r2'])
            
            sd_r2 = np.std(this_unit['r2'])
            
            mean_fr = np.mean(this_unit['fr'])
            
            sd_fr = np.std(this_unit['fr'])
            
            n = len(this_unit)
            
            try:
            
                striatum_means.append((dataset,species,unit_n,mean_tau,sd_tau,np.log10(mean_tau),mean_r2,sd_r2,mean_fr,sd_fr,n))
                
            except:
                
                pass
    
striatum_means = pd.DataFrame(striatum_means,columns=['dataset','species','unit','tau','sd_tau','log_tau','mean_r2','sd_r2','mean_fr','sd_fr','n'])

striatum_means['species'] = pd.Categorical(striatum_means['species'], categories=listofspecies, ordered=True)

#%%

acc_means['brain_region'] = 'acc'
amyg_means['brain_region'] = 'amygdala'
hc_means['brain_region'] = 'hippocampus'
mpfc_means['brain_region'] = 'mpfc'
ofc_means['brain_region'] = 'ofc'

all_means = pd.concat((acc_means,amyg_means,hc_means,mpfc_means,ofc_means))

#%% 

acc_means2 = acc_means.assign(brain_region = 'acc')
amyg_means2 = amyg_means.assign(brain_region = 'amygdala')
hc_means2 = hc_means.assign(brain_region = 'hippocampus')
mpfc_means2 = mpfc_means.assign(brain_region = 'mpfc')
ofc_means2 = ofc_means.assign(brain_region = 'ofc')

brain_region_data = pd.concat((acc_means2,amyg_means2,hc_means2,mpfc_means2,ofc_means2))

#%%

fig = plt.figure(figsize=(11,8.5))

ax1 = fig.add_subplot(121)

f = sns.scatterplot(ax=ax1,data=brain_region_data,x='mean_r2',y='tau',hue='species',s=8,alpha=0.6)

f.set_xlabel('goodness of fit (R$^2$)')
f.set_ylabel('timescale (ms)')

ax2 = fig.add_subplot(122,sharey=ax1)

f = sns.scatterplot(ax=ax2,data=brain_region_data,x='mean_fr',y='tau',hue='species',s=8,alpha=0.6)

f.set_xlabel('mean firing rate (hz)')

plt.show()
# %% model

r2_mod = (smf.ols('tau ~ mean_r2',data=brain_region_data)).fit()

print(r2_mod.summary())

#%%

fr_mod = (smf.ols('tau ~ mean_fr',data=brain_region_data)).fit()

print(fr_mod.summary())
# %%
