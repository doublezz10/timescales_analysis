#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from scipy.stats import levene

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

#%%

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data_with_lai_vl.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

fred_brain_region_data = fred_data

#%%

fig,axs = plt.subplots(2,3,figsize=(6.5,3.75),sharex=True,sharey=True)

for region, ax in zip(fred_brain_region_data.brain_region.unique(),axs.ravel()):
        
    this_region = fred_brain_region_data[fred_brain_region_data.brain_region==region]
    
    g=sns.histplot(ax=ax,data=this_region,x='tau',hue='species',element='step',stat='percent',fill=False,log_scale=True,binwidth=0.1,common_norm=False,alpha=0.7)
    
    ax.set_title(region + ' (n=%i)'%(len(this_region)),fontsize=8)
    
    ax.set_xlabel('timescale (ms)',fontsize=7)
    ax.set_ylabel('percent',fontsize=7)
    
    ax.tick_params(axis='x',labelsize=7)
    ax.tick_params(axis='y',labelsize=7)
    
    plt.setp(g.get_legend().get_texts(), fontsize='7') 
    leg = ax.get_legend()
    leg.set_title('')

    if region!='ACC':
        
        ax.get_legend().remove()
    
plt.tight_layout()
plt.show()

#%%

for brain_region in fred_brain_region_data.brain_region.unique():
    
    if brain_region in ['LAI', 'vlPFC']:
        
        pass
    
    elif brain_region == 'mPFC':
        
        this_region = fred_brain_region_data[fred_brain_region_data.brain_region==brain_region]
    
        p = levene(this_region[this_region.species=='mouse'].tau,this_region[this_region.species=='monkey'].tau)
    
        print(brain_region)
        print(p)
    else:
    
        this_region = fred_brain_region_data[fred_brain_region_data.brain_region==brain_region]
        
        p = levene(this_region[this_region.species=='mouse'].tau,this_region[this_region.species=='monkey'].tau,this_region[this_region.species=='human'].tau)
        
        print(brain_region)
        print(p)

#%%

fig,axs = plt.subplots(2,3,figsize=(6.5,3.75),sharex=True,sharey=True)

for region, ax in zip(fred_brain_region_data.brain_region.unique(),axs.ravel()):
    
    this_region = fred_brain_region_data[fred_brain_region_data.brain_region==region]
    
    g=sns.histplot(ax=ax,data=this_region,x='lat',hue='species',element='step',stat='percent',fill=False,common_norm=False,alpha=0.7,legend=True)
    
    ax.set_title(region + ' (n=%i)'%(len(this_region)),fontsize=8)
    
    ax.set_xlabel('latency (ms)',fontsize=7)
    ax.set_ylabel('percent',fontsize=7)
    
    ax.tick_params(axis='x',labelsize=7)
    ax.tick_params(axis='y',labelsize=7)
    
    plt.setp(g.get_legend().get_texts(), fontsize='7') 
    leg = ax.get_legend()
    leg.set_title('')
    
    if region!='ACC':
        
        ax.get_legend().remove()
    
plt.tight_layout()
plt.show()

#%%

for brain_region in fred_brain_region_data.brain_region.unique():
    
    if brain_region in ['LAI', 'vlPFC']:
        
        pass
    
    elif brain_region == 'mPFC':
        
        this_region = fred_brain_region_data[fred_brain_region_data.brain_region==brain_region]
    
        p = levene(this_region[this_region.species=='mouse'].lat,this_region[this_region.species=='monkey'].lat)
    
        print(brain_region)
        print(p)
    else:
    
        this_region = fred_brain_region_data[fred_brain_region_data.brain_region==brain_region]
        
        p = levene(this_region[this_region.species=='mouse'].lat,this_region[this_region.species=='monkey'].lat,this_region[this_region.species=='human'].lat)
        
        print(brain_region)
        print(p)

# %% how do variances compare?

from scipy.stats import levene

for brain_region in fred_brain_region_data.brain_region.unique():
    
    this_region = fred_brain_region_data[fred_brain_region_data.brain_region==brain_region]
    
    mouse = this_region[this_region.species=='mouse'].tau.to_numpy()
    monkey = this_region[this_region.species=='monkey'].tau.to_numpy()
    
    if brain_region == 'mPFC':
        
        stat, p = levene(mouse,monkey,center='median')
        
        print(brain_region)
        
        print('F(%i,%i) = %.2f' %(1,len(mouse)+len(monkey),stat))
        
        print('tau p-val = %.3f' %p)
        
    elif brain_region in ['LAI','vlPFC']:
        
        pass
        
    else:
        
        human = this_region[this_region.species=='human'].tau.to_numpy()
        
        stat, p = levene(mouse,monkey,human,center='median')
        
        print(brain_region)
        print('F(%i,%i) = %.2f' %(2,len(mouse)+len(monkey)+len(human),stat))
        print('tau p-val = %.3f' %p)
        
#%%

for brain_region in fred_brain_region_data.brain_region.unique():
    
    this_region = fred_brain_region_data[fred_brain_region_data.brain_region==brain_region]
    
    mouse = this_region[this_region.species=='mouse'].lat.to_numpy()
    monkey = this_region[this_region.species=='monkey'].lat.to_numpy()
    
    if brain_region == 'mPFC':
        
        stat, p = levene(mouse,monkey,center='median')
        
        print(brain_region)
        print('F(%i,%i) = %.2f' %(1,len(mouse)+len(monkey),stat))
        print('lat p-val = %.3f' %p)
        
    elif brain_region in ['LAI','vlPFC']:
        
        pass
        
    else:
        
        human = this_region[this_region.species=='human'].lat.to_numpy()
        
        stat, p = levene(mouse,monkey,human,center='median')
        
        print(brain_region)
        print('F(%i,%i) = %.2f' %(2,len(mouse)+len(monkey)+len(human),stat))
        print('lat p-val = %.3f' %p)
        
#%%
