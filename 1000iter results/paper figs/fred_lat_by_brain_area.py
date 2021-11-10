
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

listofspecies = ['mouse','monkey','human']

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

fred_data = fred_data[fred_data.keep == 1]

acc = fred_data[(fred_data.brain_area == 'acc') | (fred_data.brain_area == 'dACC') | (fred_data.brain_area == 'aca') | (fred_data.brain_area == 'mcc')]

amyg = fred_data[(fred_data.brain_area == 'amygdala') | (fred_data.brain_area == 'central') | (fred_data.brain_area == 'bla')]

hc = fred_data[(fred_data.brain_area == 'hc') | (fred_data.brain_area == 'ca1') | (fred_data.brain_area == 'ca2') | (fred_data.brain_area == 'ca3') | (fred_data.brain_area == 'dg')]

mpfc = fred_data[(fred_data.brain_area == 'mpfc') | (fred_data.brain_area == 'pl') | (fred_data.brain_area == 'ila') | (fred_data.brain_area == 'scACC')]

ofc = fred_data[(fred_data.brain_area == 'ofc') | (fred_data.brain_area == 'orb')]


acc2 = acc.assign(brain_region='ACC')
amyg2 = amyg.assign(brain_region='Amygdala')
hc2 = hc.assign(brain_region='Hippocampus')
mpfc2 = mpfc.assign(brain_region='mPFC')
ofc2 = ofc.assign(brain_region='OFC')

fred_brain_region_data = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2))

brain_regions = ['Hippocampus','Amygdala','OFC','mPFC','ACC']

fred_brain_region_data['brain_region'] = pd.Categorical(fred_brain_region_data['brain_region'], categories = brain_regions , ordered = True)

#%%

sns.lineplot(data=fred_brain_region_data,x='brain_region',y='lat',hue='species',ci=95,markers=True,legend=True)

plt.xlabel('brain region')
plt.ylabel('latency (ms)')

plt.show()


# %%
