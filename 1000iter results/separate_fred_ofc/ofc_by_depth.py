#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from scipy.io import loadmat

plt.style.use('seaborn')

#%%

ofc = pd.read_csv('fred_ofc.csv')

#%%

fig,axs = plt.subplots(1,2,figsize=(11,8.5))

sns.scatterplot(ax=axs[0],data=ofc,x='depth',y='zach_tau',hue='specific_area',size=0.5,alpha=0.6)
sns.scatterplot(ax=axs[1],data=ofc,x='depth',y='fred_tau',hue='specific_area',size=0.5,alpha=0.6)

axs[0].set_ylim(0,1000)
axs[1].set_ylim(0,1000)

axs[0].set_ylabel('iterative tau (ms)')
axs[1].set_ylabel('ISI tau (ms)')

axs[0].set_xlabel('depth (mm)')
axs[1].set_xlabel('depth (mm)')

plt.suptitle('Morbier OFC')

plt.show
# %%
