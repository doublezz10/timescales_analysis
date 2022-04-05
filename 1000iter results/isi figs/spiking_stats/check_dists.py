#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/spiking_stats/spike_stats.csv')

#%%

sns.displot(data=data,x='fano',col='brain_region',hue='species',col_wrap=3,kind='hist',height=6)

plt.show()

#%%