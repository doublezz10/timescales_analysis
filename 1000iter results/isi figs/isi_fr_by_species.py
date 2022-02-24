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

#%% 

taus = []

for species in listofspecies:
    
    this_species = fred_brain_region_data[fred_brain_region_data.species==species]
    
    model = smf.ols('tau ~ species + brain_region',data=this_species)

    res = model.fit()

    model2 = smf.ols('tau ~ species + brain_region + FR',data=this_species)

    res2 = model2.fit()
    
    taus.append((species,res2.params['FR'],res2.pvalues['FR'],'tau'))
        
    model = smf.ols('lat ~ species + brain_region',data=this_species)

    res = model.fit()

    model2 = smf.ols('lat ~ species + brain_region + FR',data=this_species)

    res2 = model2.fit()
    
    taus.append((species,res2.params['FR'],res2.pvalues['FR'],'lat'))
    
taus = pd.DataFrame(taus,columns=['species','fr_beta','fr_pval','tau/lat'])

taus = taus.assign(logp = -1 * np.log10(taus['fr_pval']))

#%%

plt.figure(figsize=(6,3))

sns.barplot(data=taus,x='species',y='logp',hue='tau/lat',palette='Set2')

plt.axhline(-1 * np.log10(0.05),linestyle='dashed',color='#e78ac3')

plt.ylabel('-log(p)',fontsize=7)
plt.xlabel('')

plt.tick_params(axis='x', rotation=0,labelsize=7)
plt.tick_params(axis='y',labelsize=7)

plt.legend(title='parameter',prop={'size': 7})

plt.show()

#%%
    
s = sns.lmplot(x='FR',y='tau',col='species',hue='species',data=fred_brain_region_data,scatter_kws={'s':5, 'alpha': 0.5},palette=None,height=3)

s.set_titles("{col_name}",size=7)
s.set_axis_labels(x_var='Firing rate (hz)',y_var='Timescale (ms)',fontsize=7)
s.set(xlim=(0,50),ylim=(0,1000))
s.set_yticklabels([0,200,400,600,800,1000],size = 7)
s.set_xticklabels([0,10,20,30,40,50],size = 7)

plt.show()
    
s = sns.lmplot(x='FR',y='lat',col='species',hue='species',data=fred_brain_region_data,scatter_kws={'s':5, 'alpha': 0.5},palette=None,height=3)

plt.xlabel('')
s.set(xlim=(0,50),ylim=(0,400))
s.set_yticklabels([0,100,200,300,400],size = 7)
s.set_xticklabels([0,10,20,30,40,50],size = 7)

s.set_titles("{col_name}",size=7)
s.set_axis_labels(x_var='Firing rate (hz)',y_var='Latency (ms)',fontsize=7)

plt.show()
    
#%% what happens if we restrict the fr space?
# Not a lot -- only makes relationships more significant

restrict_fr = fred_brain_region_data[fred_brain_region_data.FR < 10]

for species in listofspecies:
    
    this_species = restrict_fr[restrict_fr.species==species]
    
    model = smf.ols('tau ~ species + brain_region',data=this_species)

    res = model.fit()

    model2 = smf.ols('tau ~ species + brain_region + FR',data=this_species)

    res2 = model2.fit()
    
    print(species, 'tau')

    print(anova_lm(res,res2))
        
    model = smf.ols('lat ~ species + brain_region',data=this_species)

    res = model.fit()

    model2 = smf.ols('lat ~ species + brain_region + FR',data=this_species)

    res2 = model2.fit()
    
    print(species, 'lat')

    print(anova_lm(res,res2))
    
#%%
    
sns.lmplot(x='FR',y='tau',col='species',data=restrict_fr,scatter_kws={'s':5, 'alpha': 0.5})

plt.ylim(0,1000)

plt.show()

    
sns.lmplot(x='FR',y='lat',col='species',data=restrict_fr,scatter_kws={'s':5, 'alpha': 0.5})

plt.show()
# %%
