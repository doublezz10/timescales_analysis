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

data = pd.read_csv('processed_data.csv')

data['species'] = pd.Categorical(data['species'], categories = ['mouse','monkey','human'] , ordered = True)


filt_data = data[np.logical_and(data.tau < 1000, data.r2 > 0.5)]
filt_data = filt_data[filt_data.tau > 10]

brain_regions = ['Hippocampus','OFC','Amygdala','mPFC','ACC']

filt_data['brain_region'] = pd.Categorical(filt_data['brain_region'], categories = brain_regions , ordered = True)


#%%

fig,axs = plt.subplots(1,5,figsize=(7,2),sharex=True,sharey=True)

for region, ax in zip(brain_regions,axs.ravel()):
        
    this_region = filt_data[filt_data.brain_region==region]
    
    g=sns.kdeplot(ax=ax,data=this_region,x='tau',hue='species',fill=False,log_scale=True,common_norm=False,alpha=0.7)
    
    ax.set_title(region + ' (n=%i)'%(len(this_region)),fontsize=8)
    
    ax.set_xlabel('timescale (ms)',fontsize=7)
    ax.set_ylabel('density',fontsize=7)
    
    ax.tick_params(axis='x',labelsize=7)
    ax.tick_params(axis='y',labelsize=7)
    #ax.xaxis.set_label_position('top')
    
    plt.setp(g.get_legend().get_texts(), fontsize='7') 
    leg = ax.get_legend()
    leg.set_title('')

    if region!='ACC':
        
        ax.get_legend().remove()
    
plt.tight_layout()
plt.show()

#%%

for brain_region in filt_data.brain_region.unique():
    
    if brain_region in ['LAI', 'vlPFC']:
        
        pass
    
    elif brain_region == 'mPFC':
        
        this_region = filt_data[filt_data.brain_region==brain_region]
    
        p = levene(this_region[this_region.species=='mouse'].tau,this_region[this_region.species=='monkey'].tau)
    
        print(brain_region)
        print(p)
    else:
    
        this_region = filt_data[filt_data.brain_region==brain_region]
        
        p = levene(this_region[this_region.species=='mouse'].tau,this_region[this_region.species=='monkey'].tau,this_region[this_region.species=='human'].tau)
        
        print(brain_region)
        print(p)

#%%

fig,axs = plt.subplots(1,5,figsize=(7,2),sharex=True,sharey=True)

for region, ax in zip(brain_regions,axs.ravel()):
    
    this_region = filt_data[filt_data.brain_region==region]
    
    g=sns.kdeplot(ax=ax,data=this_region,x='lat',hue='species',fill=False,common_norm=False,alpha=0.7)
    
    ax.set_title(region + ' (n=%i)'%(len(this_region)),fontsize=8)
    
    ax.set_xlabel('latency (ms)',fontsize=7)
    ax.set_ylabel('density',fontsize=7)
    
    #ax.xaxis.set_label_position('top')
    
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

for brain_region in filt_data.brain_region.unique():
    
    if brain_region in ['LAI', 'vlPFC']:
        
        pass
    
    elif brain_region == 'mPFC':
        
        this_region = filt_data[filt_data.brain_region==brain_region]
    
        p = levene(this_region[this_region.species=='mouse'].lat,this_region[this_region.species=='monkey'].lat)
    
        print(brain_region)
        print(p)
    else:
    
        this_region = filt_data[filt_data.brain_region==brain_region]
        
        p = levene(this_region[this_region.species=='mouse'].lat,this_region[this_region.species=='monkey'].lat,this_region[this_region.species=='human'].lat)
        
        print(brain_region)
        print(p)

# %% how do variances compare?

from scipy.stats import levene

for brain_region in filt_data.brain_region.unique():
    
    this_region = filt_data[filt_data.brain_region==brain_region]
    
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

for brain_region in filt_data.brain_region.unique():
    
    this_region = filt_data[filt_data.brain_region==brain_region]
    
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
