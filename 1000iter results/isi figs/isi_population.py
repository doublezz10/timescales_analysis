#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_pop_data.csv')

fred_data = fred_data[fred_data.species != 'rat']

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

fred_data = fred_data[fred_data.keep == 1]

acc = fred_data[(fred_data.brain_area == 'acc') | (fred_data.brain_area == 'dACC') | (fred_data.brain_area == 'aca') | (fred_data.brain_area == 'mcc')]

amyg = fred_data[(fred_data.brain_area == 'amygdala') | (fred_data.brain_area == 'central') | (fred_data.brain_area == 'bla')]

hc = fred_data[(fred_data.brain_area == 'hc') | (fred_data.brain_area == 'ca1') | (fred_data.brain_area == 'ca2') | (fred_data.brain_area == 'ca3') | (fred_data.brain_area == 'dg')]

mpfc = fred_data[(fred_data.brain_area == 'mpfc') | (fred_data.brain_area == 'pl') | (fred_data.brain_area == 'ila') | (fred_data.brain_area == 'scACC')]

ofc = fred_data[(fred_data.brain_area == 'ofc') | (fred_data.brain_area == 'orb')]

lai = fred_data[fred_data.brain_area == 'LAI']


acc2 = acc.assign(brain_region='ACC')
amyg2 = amyg.assign(brain_region='Amygdala')
hc2 = hc.assign(brain_region='Hippocampus')
mpfc2 = mpfc.assign(brain_region='mPFC')
ofc2 = ofc.assign(brain_region='OFC')
lai2 = lai.assign(brain_region='LAI')


fred_brain_region_data = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2))

fred_brain_region_data_lai = pd.concat((fred_brain_region_data,lai2))

#%%

brain_regions2 = ['Hippocampus','Amygdala','OFC','mPFC','ACC']

fred_brain_region_data['brain_region'] = pd.Categorical(fred_brain_region_data['brain_region'], categories = brain_regions2 , ordered = True)

fig= plt.figure(figsize=(3.4,2.6))

sns.lineplot(data=fred_brain_region_data,x='brain_region',y='tau',hue='species',ci=None,markers=True,legend=False)
sns.scatterplot(data=fred_brain_region_data,x='brain_region',y='tau',hue='species',s=25)

plt.xlabel(None)
plt.tick_params(axis='x', rotation=0,labelsize=7)
plt.tick_params(axis='y',labelsize=7)
plt.ylabel('timescale (ms)',fontsize=7)

plt.legend(title='species',prop={'size': 7})

plt.grid(False)

plt.show()

#%%

fig= plt.figure(figsize=(3.2,2.6))

sns.lineplot(data=fred_brain_region_data,x='brain_region',y='lat',hue='species',ci=None,markers=True,legend=False)
sns.scatterplot(data=fred_brain_region_data,x='brain_region',y='lat',hue='species',legend=False,s=25)

plt.xlabel(None)
plt.tick_params(axis='x', rotation=0,labelsize=7)
plt.tick_params(axis='y',labelsize=7)
plt.ylabel('latency (ms)',fontsize=7)

plt.grid(False)


plt.show()

# %%
