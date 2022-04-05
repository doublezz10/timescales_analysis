#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

plt.style.use('seaborn')

data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/spiking_stats/spike_stats.csv')

#%% restrict to only steinmetz amyg

amyg = data[data.dataset=='steinmetz']
amyg = amyg[amyg.brain_area=='bla']

amyg_lower = amyg[amyg.tau<20]

amyg_low = amyg[amyg.tau<50]
amyg_low = amyg_low[amyg_low.tau>20]

amyg_rest = amyg[amyg.tau>50]

amyg_lower = amyg_lower.assign(group='tau < 20')
amyg_low = amyg_low.assign(group='20 < tau < 50')
amyg_rest = amyg_rest.assign(group='50 < tau')

all_amyg = pd.concat((amyg_lower,amyg_low,amyg_rest))

all_amyg['group'] = pd.Categorical(all_amyg['group'],categories=['tau < 20','20 < tau < 50','50 < tau'],ordered=True)

#%%

fig,ax = plt.subplots(1,1,figsize=(6,6))

sns.histplot(ax=ax,data=all_amyg,x='tau',element='step',fill=False,stat='percent',common_norm=False,log_scale=True)

plt.show()

#%%

for col in ['n_spikes','mean_norm_fr','fano','prop_burst','prop_pause','r2']:
    
    fig,ax = plt.subplots(1,1,figsize=(6,6))

    sns.histplot(ax=ax,data=all_amyg,x=col,hue='group',element='step',fill=False,stat='percent',common_norm=False)

    plt.title('Steinmetz Amygdala')

    plt.show()

#%%

model = smf.ols('prop_burst ~ group',data=all_amyg)

res = model.fit()

print(res.summary())

#%% compare to monkey amygdala

amyg = data[data.species=='monkey']
amyg = amyg[amyg.brain_area=='amygdala']

amyg_lower = amyg[amyg.tau<20]

amyg_low = amyg[amyg.tau<50]
amyg_low = amyg_low[amyg_low.tau>20]

amyg_rest = amyg[amyg.tau>50]

amyg_lower = amyg_lower.assign(group='tau < 20')
amyg_low = amyg_low.assign(group='20 < tau < 50')
amyg_rest = amyg_rest.assign(group='50 < tau')

all_amyg = pd.concat((amyg_lower,amyg_low,amyg_rest))

all_amyg['group'] = pd.Categorical(all_amyg['group'],categories=['tau < 20','20 < tau < 50','50 < tau'],ordered=True)

#%%

fig,ax = plt.subplots(1,1,figsize=(6,6))

sns.histplot(ax=ax,data=all_amyg,x='tau',element='step',fill=False,stat='percent',common_norm=False,log_scale=True)

plt.show()

#%%

for col in ['n_spikes','mean_norm_fr','fano','prop_burst','prop_pause','r2']:
    
    fig,ax = plt.subplots(1,1,figsize=(6,6))

    sns.histplot(ax=ax,data=all_amyg,x=col,hue='group',element='step',fill=False,stat='percent',common_norm=False)

    plt.title('Monkey Amygdala')

    plt.show()

#%%

model = smf.ols('prop_burst ~ group',data=all_amyg)

res = model.fit()

print(res.summary())

#%%

filtdata = data[data.mean_fr > 1]

sns.lmplot(data=filtdata[filtdata.brain_region!='vlPFC'],x='fano',y='tau',hue='species',col='brain_region',col_wrap=3,scatter_kws={'s':2,'alpha':0.7})

plt.show()
# %%

model = smf.ols('tau ~ fano + brain_region + species',data=filtdata)

res = model.fit()

print(res.summary())
# %%

model = smf.ols('tau ~ fano',data=filtdata)

res = model.fit()

print(res.summary())

# %%
