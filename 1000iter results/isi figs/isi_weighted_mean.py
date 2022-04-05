#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

fred_brain_region_data2 = fred_data

brain_regions2 = ['Hippocampus','Amygdala','OFC','ACC','mPFC']

fred_brain_region_data2['brain_region'] = pd.Categorical(fred_brain_region_data2['brain_region'], categories = brain_regions2 , ordered = True)


def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

fred_brain_region_data2.groupby(['brain_region','species']).apply(w_avg, 'tau', 'r2').unstack().plot()
# %%
