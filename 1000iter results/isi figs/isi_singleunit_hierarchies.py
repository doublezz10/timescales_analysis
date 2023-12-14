#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/filtered_isi_data.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

fred_brain_region_data = fred_data
fred_brain_region_data2 = fred_data

#%% First fig (no LAI)

brain_regions2 = ['Hippocampus','OFC','Amygdala','mPFC','ACC']

fred_brain_region_data2['brain_region'] = pd.Categorical(fred_brain_region_data2['brain_region'], categories = brain_regions2 , ordered = True)

fig, axs = plt.subplots(1,2,figsize=(4.475,2.75))

sns.lineplot(ax=axs[0],data=fred_brain_region_data2[fred_brain_region_data2.brain_region != 'LAI'],x='brain_region',y='tau',hue='species',ci=95,markers=True,legend=True,estimator=np.mean)
sns.pointplot(ax=axs[0],data=fred_brain_region_data2[fred_brain_region_data2.brain_region != 'LAI'],x='brain_region',y='tau',hue='species',ci=None,connect=False,scale=0.4,legend=False)

axs[0].set_xlabel(None)
axs[0].tick_params(axis='x', rotation=90,labelsize=7)
axs[0].tick_params(axis='y',labelsize=7)
axs[0].set_ylabel('timescale (ms)',fontsize=7)

axs[0].legend(['mouse','monkey','human'],title='',prop={'size': 7})

sns.lineplot(ax=axs[1],data=fred_brain_region_data2[fred_brain_region_data2.brain_region != 'LAI'],x='brain_region',y='lat',hue='species',ci=95,markers=True,legend=False,estimator=np.mean)
sns.pointplot(ax=axs[1],data=fred_brain_region_data2[fred_brain_region_data2.brain_region != 'LAI'],x='brain_region',y='lat',hue='species',ci=None,connect=False,scale=0.4)


axs[1].set_xlabel(None)
axs[1].tick_params(axis='x', rotation=90,labelsize=7)
axs[1].tick_params(axis='y',labelsize=7)
axs[1].set_ylabel('latency (ms)',fontsize=7)

axs[1].get_legend().remove()

plt.tight_layout()

plt.show()


#%% Is latency correlated with fred's tau?

sns.lmplot(data=fred_brain_region_data,x='lat',y='tau',hue='species',ci=95,legend=True)

plt.xlabel('latency (ms)')
plt.ylabel('timescale (ms)')

plt.show()

#%%