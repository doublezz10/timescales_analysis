#%%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

data = pd.read_csv('processed_data.csv')

data['species'] = pd.Categorical(data['species'], categories = ['mouse','monkey','human'] , ordered = True)


filt_data = data[np.logical_and(data.tau < 1000, data.fr > 1)]
filt_data = filt_data[filt_data.tau > 10]

brain_regions = ['Hippocampus','OFC','Amygdala','mPFC','ACC']

filt_data['brain_region'] = pd.Categorical(filt_data['brain_region'], categories = brain_regions , ordered = True)

#%%

fig, axs = plt.subplots(1,2,figsize=(4.475,2.75))

sns.lineplot(ax=axs[0],data=filt_data,x='brain_region',y='tau',hue='species',ci=95,markers=True,legend=True,estimator=np.mean)
g = sns.pointplot(ax=axs[0],data=filt_data,x='brain_region',y='tau',hue='species',ci=None,connect=False,scale=0.4,legend=False)

axs[0].set_xlabel(None)
axs[0].tick_params(axis='x', rotation=90,labelsize=7)
axs[0].tick_params(axis='y',labelsize=7)
axs[0].set_ylabel('timescale (ms)',fontsize=7)

handles, labels = g.get_legend_handles_labels()
axs[0].legend(handles[:3], labels[:3],loc='upper right',prop={'size':7})

sns.lineplot(ax=axs[1],data=filt_data,x='brain_region',y='peak_lat',hue='species',ci=95,markers=True,legend=False,estimator=np.mean)
sns.pointplot(ax=axs[1],data=filt_data,x='brain_region',y='peak_lat',hue='species',ci=None,connect=False,scale=0.4)


axs[1].set_xlabel(None)
axs[1].tick_params(axis='x', rotation=90,labelsize=7)
axs[1].tick_params(axis='y',labelsize=7)
axs[1].set_ylabel('latency (ms)',fontsize=7)

axs[1].get_legend().remove()

plt.tight_layout()

plt.show()

#%%