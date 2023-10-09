#%%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

data = pd.read_csv('processed_data.csv')

data['species'] = pd.Categorical(data['species'], categories = ['mouse','monkey','human'] , ordered = True)


filt_data = data[np.logical_and(data.tau < 1000, data.fr > 1)]
#filt_data = filt_data[filt_data.tau > 10]

brain_regions = ['Hippocampus','OFC','Amygdala','mPFC','ACC']

filt_data['brain_region'] = pd.Categorical(filt_data['brain_region'], categories = brain_regions , ordered = True)

#%%

# brain region data

plt.figure(figsize=(3,3))

f = sns.countplot(data=data,x='brain_region',hue='species',alpha=0.2,order=['Hippocampus','Amygdala','OFC','mPFC','ACC'])
g = sns.countplot(data=filt_data,x='brain_region',hue='species',order=['Hippocampus','Amygdala','OFC','mPFC','ACC'])

handles, labels = g.get_legend_handles_labels()
plt.legend(handles[3:], labels[3:],loc='upper right',prop={'size':7})
plt.xlabel('')
plt.ylabel('count',fontsize=7)
plt.xticks(fontsize=7,rotation=45)
plt.yticks(fontsize=7)

plt.show()

# %%

plt.figure(figsize=(3,3))

f = sns.countplot(data=data,x='fit',hue='species',alpha=0.2,order=['mono','dip1','dip2'])
g = sns.countplot(data=filt_data,x='fit',hue='species',order=['mono','dip1','dip2'])

handles, labels = g.get_legend_handles_labels()
plt.legend(handles[3:], labels[3:],loc='upper right',prop={'size':7})
plt.xlabel('')
plt.ylabel('count',fontsize=7)
plt.xticks(fontsize=7,rotation=45)
plt.yticks(fontsize=7)

plt.show()

#%%