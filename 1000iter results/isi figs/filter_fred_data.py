#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from scipy.io import loadmat

plt.style.use('seaborn')

#%% Load in data, filter

listofspecies = ['mouse','monkey','human']


# Load in Fred's data

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data_new.csv')
fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})

fred_data = fred_data[fred_data.species != 'rat']
fred_data = fred_data[fred_data.dataset != 'faraut']
fred_data = fred_data[fred_data.dataset != 'froot']

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus','hc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('mPFC','mpfc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('ventralStriatum','vStriatum')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('Cd','caudate')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('OFC','ofc')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('PUT','putamen')
fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus2','hc2')

fred_data = fred_data.replace(['amyg','AMG'],'amygdala')

fred_data['dataset'] = fred_data['dataset'].str.replace('stein','steinmetz')

fred_data = fred_data[fred_data.r2 >= 0.5]

fred_data = fred_data[(fred_data.tau >=10) & (fred_data.tau <= 1000)]

fred_data = fred_data[fred_data.keep == 1]

#%%

acc = fred_data[(fred_data.brain_area == 'acc') | (fred_data.brain_area == 'dACC') | (fred_data.brain_area == 'aca') | (fred_data.brain_area == 'mcc')]

amyg = fred_data[(fred_data.brain_area == 'amygdala') | (fred_data.brain_area == 'central') | (fred_data.brain_area == 'bla')]

hc = fred_data[(fred_data.brain_area == 'hc') | (fred_data.brain_area == 'hc2') | (fred_data.brain_area == 'ca1') | (fred_data.brain_area == 'ca2') | (fred_data.brain_area == 'ca3') | (fred_data.brain_area == 'dg')]

mpfc = fred_data[(fred_data.brain_area == 'mpfc') | (fred_data.brain_area == 'pl') | (fred_data.brain_area == 'ila') | (fred_data.brain_area == 'scACC')]

ofc = fred_data[(fred_data.brain_area == 'ofc') | (fred_data.brain_area == 'orb')]

lai = fred_data[fred_data.brain_area == 'LAI']

vlpfc = fred_data[fred_data.brain_area == 'vlPFC']


acc2 = acc.assign(brain_region='ACC')
amyg2 = amyg.assign(brain_region='Amygdala')
hc2 = hc.assign(brain_region='Hippocampus')
mpfc2 = mpfc.assign(brain_region='mPFC')
ofc2 = ofc.assign(brain_region='OFC')
lai2 = lai.assign(brain_region='LAI')
vlpfc2 = vlpfc.assign(brain_region='vlPFC')

fred_brain_region_data = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2))

fred_brain_region_data_lai = pd.concat((fred_brain_region_data,lai2,vlpfc2))

fred_brain_region_data.to_csv('filtered_isi_data.csv')
fred_brain_region_data_lai.to_csv('filtered_isi_data_with_lai_vl.csv')

#%%