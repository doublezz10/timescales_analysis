#%%

"""
Created on Tue Apr 13 10:57:16 2021

@author: zachz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

#%% Load in Fred's data

listofspecies = ['mouse','rat','monkey','human']


fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')
fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

# rename columns to match

fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus','hc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('mPFC','mpfc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('ventralStriatum','vStriatum')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('AMG','amygdala')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('Cd','caudate')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('OFC','ofc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('PUT','putamen')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus2','hc2')

fred_data['dataset'] = fred_data['dataset'].str.replace('stein','steinmetz')

fred_data = fred_data[fred_data.r2 >= 0.5]

fred_data = fred_data[(fred_data.tau >=10) & (fred_data.tau <= 1000)]

low_tau = fred_data[fred_data.tau <= 25]

#%%

sns.scatterplot(x='FR',y='tau',data=low_tau,hue='species',s=3,alpha=0.7)

plt.title('Fred data \n 10 < tau < 25')

plt.show()

#%%