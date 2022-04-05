#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from scipy import stats

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

#%% Load single-unit data

listofspecies = ['monkey','mouse','human']

fred_data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

fred_brain_region_data = fred_data

fred_data_lai = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data_with_lai_vl.csv')

fred_data_lai['species'] = pd.Categorical(fred_data_lai['species'], categories=listofspecies, ordered=True)

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

# %% Amyg only tau

model = smf.ols('tau ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='Amygdala'])

res = model.fit()

print(res.summary())

# %% Amyg only lat

model = smf.ols('lat ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='Amygdala'])

res = model.fit()

print(res.summary())

# %% OFC only tau

model = smf.ols('tau ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='OFC'])

res = model.fit()

print(res.summary())

# %% OFC only lat

model = smf.ols('lat ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='OFC'])

res = model.fit()

print(res.summary())

#%% Within OFC

ofc = pd.read_csv('fred_ofc_isi.csv')
lai = pd.read_csv('fred_lai_isi.csv')
vl = pd.read_csv('fred_vl_isi.csv')

ofc_lai_vl = pd.concat((ofc,lai,vl),ignore_index=True)

ofc_lai_vl['brain_area'] = ofc_lai_vl['brain_area'].str.replace('LAI','AI')
ofc_lai_vl['specific_area'] = ofc_lai_vl['specific_area'].str.replace('LAI','AI')

ofc_lai_vl.loc[ofc_lai_vl['specific_area'] == '11m', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area']== '11l', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area']== '12m', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area']== '12l', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area']== '12o', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area']== '45', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area']== '12r', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area'] =='13m', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area'] == '13l', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['specific_area'] =='AI', 'granularity'] = 'agranular'

ofc_lai_vl['granularity'] = pd.Categorical(ofc_lai_vl['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)

ofc_lai_vl = ofc_lai_vl[ofc_lai_vl.specific_area!='13b']
ofc_lai_vl = ofc_lai_vl[ofc_lai_vl.specific_area!='45']

#%%

model = smf.ols('tau ~ granularity', data = ofc_lai_vl)

res = model.fit()

print(res.summary())


#%%

model = smf.ols('tau ~ specific_area', data = ofc_lai_vl)

res2 = model.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

#%%

model = smf.ols('lat ~ granularity', data = ofc_lai_vl)

res = model.fit()

print(res.summary())


#%%

model = smf.ols('lat ~ specific_area', data = ofc_lai_vl)

res2 = model.fit()

print(res2.summary())
#%%

print(anova_lm(res,res2))
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
