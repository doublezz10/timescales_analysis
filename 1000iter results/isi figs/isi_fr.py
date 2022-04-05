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

#%% Load single-unit data

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

fred_brain_region_data = fred_data

fred_data_lai = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data_with_lai.csv')

fred_data_lai['species'] = pd.Categorical(fred_data_lai['species'], categories=listofspecies, ordered=True)

#%% Pop hierarchy

listofspecies = ['mouse','monkey','human']

fred_pop_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_pop_data.csv')

fred_pop_data = fred_pop_data[fred_pop_data.species != 'rat']

fred_pop_data = fred_pop_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
fred_pop_data['species'] = pd.Categorical(fred_pop_data['species'], categories=listofspecies, ordered=True)

# rename columns to match

fred_pop_data['brain_area'] = fred_pop_data['brain_area'].str.replace('hippocampus','hc')
fred_pop_data['brain_area'] = fred_pop_data['brain_area'].str.replace('mPFC','mpfc')
fred_pop_data['brain_area'] = fred_pop_data['brain_area'].str.replace('ventralStriatum','vStriatum')
fred_pop_data['brain_area'] = fred_pop_data['brain_area'].str.replace('AMG','amygdala')
fred_pop_data['brain_area'] = fred_pop_data['brain_area'].str.replace('Cd','caudate')
fred_pop_data['brain_area'] = fred_pop_data['brain_area'].str.replace('OFC','ofc')
fred_pop_data['brain_area'] = fred_pop_data['brain_area'].str.replace('PUT','putamen')
fred_pop_data['brain_area'] = fred_pop_data['brain_area'].str.replace('hippocampus2','hc2')

fred_pop_data['dataset'] = fred_pop_data['dataset'].str.replace('stein','steinmetz')

fred_pop_data = fred_pop_data[fred_pop_data.r2 >= 0.5]

fred_pop_data = fred_pop_data[(fred_pop_data.tau >=10) & (fred_pop_data.tau <= 1000)]

fred_pop_data = fred_pop_data[fred_pop_data.keep == 1]

acc = fred_pop_data[(fred_pop_data.brain_area == 'acc') | (fred_pop_data.brain_area == 'dACC') | (fred_pop_data.brain_area == 'aca') | (fred_pop_data.brain_area == 'mcc')]

amyg = fred_pop_data[(fred_pop_data.brain_area == 'amygdala') | (fred_pop_data.brain_area == 'central') | (fred_pop_data.brain_area == 'bla')]

hc = fred_pop_data[(fred_pop_data.brain_area == 'hc') | (fred_pop_data.brain_area == 'ca1') | (fred_pop_data.brain_area == 'ca2') | (fred_pop_data.brain_area == 'ca3') | (fred_pop_data.brain_area == 'dg')]

mpfc = fred_pop_data[(fred_pop_data.brain_area == 'mpfc') | (fred_pop_data.brain_area == 'pl') | (fred_pop_data.brain_area == 'ila') | (fred_pop_data.brain_area == 'scACC')]

ofc = fred_pop_data[(fred_pop_data.brain_area == 'ofc') | (fred_pop_data.brain_area == 'orb')]

lai = fred_pop_data[fred_pop_data.brain_area == 'LAI']


acc2 = acc.assign(brain_region='ACC')
amyg2 = amyg.assign(brain_region='Amygdala')
hc2 = hc.assign(brain_region='Hippocampus')
mpfc2 = mpfc.assign(brain_region='mPFC')
ofc2 = ofc.assign(brain_region='OFC')
lai2 = lai.assign(brain_region='LAI')


fred_pop_brain_region_data = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2))

#%% whole dataset plot by fr

sns.lmplot(data=fred_brain_region_data,x='FR',y='tau',hue='species',col='brain_region',col_wrap=3,scatter_kws={'s':5, 'alpha': 0.5})

plt.ylim(0,1000)

plt.show()
# %% z-score FR by species

zscore = lambda x: (x - x.mean()) / x.std()

fred_brain_region_data.insert(14, 'zscore_fr', fred_brain_region_data.groupby(['species'])['FR'].transform(zscore))

sns.lmplot(data=fred_brain_region_data,x='zscore_fr',y='tau',hue='species',col='brain_region',col_wrap=3,scatter_kws={'s':5, 'alpha': 0.5})

plt.ylim(0,1000)

plt.show()

# %%

sns.lmplot(data=fred_brain_region_data,x='FR',y='lat',hue='species',col='brain_region',col_wrap=3,scatter_kws={'s':5, 'alpha': 0.5})

plt.ylim(0,1000)

plt.show()

# %%

sns.lmplot(data=fred_brain_region_data,x='zscore_fr',y='lat',hue='species',col='brain_region',col_wrap=3,scatter_kws={'s':5, 'alpha': 0.5})

plt.ylim(0,1000)

plt.show()

#%% z-score within dataset

fred_brain_region_data.insert(14, 'zscore_fr_ds', fred_brain_region_data.groupby(['dataset'])['FR'].transform(zscore))

sns.lmplot(data=fred_brain_region_data,x='zscore_fr_ds',y='tau',hue='species',col='brain_region',col_wrap=3,scatter_kws={'s':5, 'alpha': 0.5})

plt.ylim(0,1000)

plt.show()

sns.lmplot(data=fred_brain_region_data,x='zscore_fr_ds',y='lat',hue='species',col='brain_region',col_wrap=3,scatter_kws={'s':5, 'alpha': 0.5})

plt.ylim(0,1000)

plt.show()
# %%

by_individual = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/all_individuals.csv')

by_individual['individual'] = pd.factorize(by_individual.individual)[0] + 1

#%%

sns.catplot(data=by_individual[by_individual.dataset=='steinmetz'],x='individual',y='tau',col='brain_area',col_wrap=3,kind='box')

plt.xticks([])

plt.show()

sns.catplot(data=by_individual[by_individual.dataset=='chandravadia'],x='individual',y='tau',col='brain_area',kind='box')

plt.xticks([])

plt.show()


# %%
