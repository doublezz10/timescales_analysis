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

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')

fred_data = fred_data[fred_data.species != 'rat']

fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
fred_data = fred_data[fred_data.dataset != 'faraut']
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

#%% Pop hierarchy

listofspecies = ['mouse','monkey','human']

fred_pop_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_pop_data.csv')

fred_pop_data = fred_pop_data[fred_pop_data.species != 'rat']

fred_pop_data = fred_pop_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
fred_data = fred_data[fred_data.dataset != 'faraut']
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

#%%

model = smf.ols('tau ~ species + brain_region',data=fred_pop_brain_region_data)

res = model.fit()

print(res.summary())

# %% Amyg only

model = smf.ols('tau ~ species',data=fred_pop_brain_region_data[fred_pop_brain_region_data.brain_region=='Amygdala'])

res = model.fit()

print(res.summary())

#%% Single-unit hierarchy

model = smf.ols('tau ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

model2 = smf.ols('tau ~ species + brain_region + FR',data=fred_brain_region_data)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

#%% Single-unit hierarchy in lat

model = smf.ols('lat ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

model2 = smf.ols('lat ~ species + brain_region + FR',data=fred_brain_region_data)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

# %% Amyg only

model = smf.ols('tau ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='Amygdala'])

res = model.fit()

print(res.summary())

# %% Amyg only

model = smf.ols('lat ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='Amygdala'])

res = model.fit()

print(res.summary())
# %% Individual

by_individual = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/all_individuals.csv')

by_individual['individual'] = pd.factorize(by_individual.individual)[0] + 1

model = smf.ols('tau ~ individual + brain_area',data=by_individual[by_individual.dataset=='steinmetz'])

res = model.fit()

print(res.summary())

#%%

by_individual['individual'] = pd.factorize(by_individual.individual)[0] + 1

model = smf.ols('tau ~ individual + FR + brain_area',data=by_individual[by_individual.dataset=='steinmetz'])

res = model.fit()

print(res.summary())

#%%

model = smf.ols('tau ~ individual + brain_area',data=by_individual[by_individual.dataset=='chandravadia'])

res = model.fit()

print(res.summary())

#%%

ofc = pd.read_csv('fred_ofc.csv')
lai = pd.read_csv('fred_lai.csv')

ofc_lai = pd.concat((ofc,lai),ignore_index=True)

model = smf.ols('fred_tau ~ specific_area', data = ofc_lai[ofc_lai.specific_area != '13b'])

res = model.fit()

print(res.summary())

#%%

model = smf.ols('fred_lat ~ specific_area', data = ofc_lai[ofc_lai.specific_area != '13b'])

res = model.fit()

print(res.summary())
# %% what if you z-score fr

zscore = lambda x: (x - x.mean()) / x.std()

fred_brain_region_data.insert(14, 'zscore_fr', fred_brain_region_data.groupby(['species'])['FR'].transform(zscore))


model = smf.ols('tau ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

model2 = smf.ols('tau ~ species + brain_region + zscore_fr',data=fred_brain_region_data)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

#%% Single-unit hierarchy in lat

model = smf.ols('lat ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

model2 = smf.ols('lat ~ species + brain_region + zscore_fr',data=fred_brain_region_data)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

#%%

fred_brain_region_data.insert(14, 'zscore_fr_ds', fred_brain_region_data.groupby(['dataset'])['FR'].transform(zscore))

model = smf.ols('tau ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

model2 = smf.ols('tau ~ species + brain_region + zscore_fr_ds',data=fred_brain_region_data)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

#%%

model = smf.ols('lat ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

model2 = smf.ols('lat ~ species + brain_region + zscore_fr_ds',data=fred_brain_region_data)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))
# %%
